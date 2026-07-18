"""Train CNN or TCN EEG models with reproducible patient-aware evaluation."""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_majority_class_baseline,
    evaluate_model,
    find_best_threshold,
    get_labels_from_split,
)
from src.evaluation.reporting import (  # noqa: E402
    log_evaluation_metrics_to_mlflow,
    save_cnn_baseline_results_table,
    save_confusion_matrix_plot,
    save_split_manifest,
    save_test_metrics_report,
    save_threshold_sweep_table,
)
from src.evaluation.splits import split_dataset, split_dataset_by_patient  # noqa: E402
from src.models.cnn import SeizureCNN  # noqa: E402
from src.models.tcn import SeizureTCN  # noqa: E402


N_CLASSES = 3
CLASS_NAMES = ["Interictal", "Pre-Ictal", "Ictal"]
DEFAULT_SEED = 42


def _set_reproducible_seed(seed=DEFAULT_SEED, deterministic=True):
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic backend settings."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def _discover_npz_files(data_path, max_patients=None):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Expected an .npz file, got: {path}")
        files = [path]
    else:
        files = sorted(path.rglob("*.npz"))

    if max_patients is not None:
        if max_patients < 1:
            raise ValueError("--max-patients must be at least 1.")
        files = files[:max_patients]

    if not files:
        raise FileNotFoundError(f"No .npz files found under: {path}")
    return files


def _read_processed_npz(npz_file):
    with np.load(npz_file, allow_pickle=False) as data:
        feature_key = "epochs" if "epochs" in data.files else "X" if "X" in data.files else None
        label_key = "labels" if "labels" in data.files else "y" if "y" in data.files else None
        if feature_key is None or label_key is None:
            raise KeyError(
                f"{npz_file} must contain epochs/labels or X/y. Found keys: {data.files}"
            )

        features = np.asarray(data[feature_key], dtype=np.float32)
        labels = np.asarray(data[label_key], dtype=np.int64).reshape(-1)
        if "sfreq" in data.files:
            sampling_rate = float(np.asarray(data["sfreq"]).reshape(-1)[0])
        elif "sampling_rate" in data.files:
            sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
        else:
            sampling_rate = 128.0

    if features.ndim == 4 and features.shape[1] == 1:
        features = features[:, 0, :, :]
    if features.ndim != 3:
        raise ValueError(
            f"Expected shape (windows, channels, timepoints), got {features.shape} in {npz_file}."
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Window/label mismatch in {npz_file}: {features.shape[0]} vs {labels.shape[0]}."
        )
    if features.shape[0] == 0:
        raise ValueError(f"Processed file contains zero windows: {npz_file}")
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError(
            f"Labels in {npz_file} must be in [0, {N_CLASSES - 1}], "
            f"found [{labels.min()}, {labels.max()}]."
        )
    return features, labels, sampling_rate


def _normalize_features(features):
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (tensor - mean) / std


def _print_split_distribution(name, dataset):
    labels = get_labels_from_split(dataset)
    counts = np.bincount(labels, minlength=N_CLASSES)
    total = len(labels)

    print(f"\n{name} class distribution")
    print("-" * 40)
    for class_index, class_name in enumerate(CLASS_NAMES):
        percentage = counts[class_index] / total * 100 if total else 0.0
        print(f"{class_name:<12}: {counts[class_index]:7d} ({percentage:6.2f}%)")
    print(f"{'Total':<12}: {total:7d}")


def _prepare_datasets(data_path, max_patients=None, seed=DEFAULT_SEED):
    files = _discover_npz_files(data_path, max_patients=max_patients)
    print(f"[INFO] Discovered {len(files)} processed subject file(s).")

    patient_datasets = []
    patient_ids = []
    sampling_rates = []
    feature_shape = None

    for npz_file in files:
        features, labels, sampling_rate = _read_processed_npz(npz_file)
        print(f"   [DATA] {npz_file} | X={features.shape} | y={labels.shape}")
        if feature_shape is None:
            feature_shape = features.shape[1:]
        elif features.shape[1:] != feature_shape:
            raise ValueError(
                "All subjects must have the same channel/time shape. "
                f"Expected {feature_shape}, got {features.shape[1:]} in {npz_file}."
            )
        patient_datasets.append((features, labels))
        patient_ids.append(npz_file.stem)
        sampling_rates.append(sampling_rate)

    sampling_rate = sampling_rates[0]
    if not all(np.isclose(rate, sampling_rate) for rate in sampling_rates):
        raise ValueError(f"Sampling-rate mismatch across subjects: {sampling_rates}")

    if len(patient_datasets) >= 3:
        datasets, selection = split_dataset_by_patient(
            patient_datasets,
            seed=seed,
            patient_ids=patient_ids,
            return_selection=True,
        )
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
        train_dataset = TensorDataset(_normalize_features(X_train), torch.tensor(y_train).long())
        val_dataset = TensorDataset(_normalize_features(X_val), torch.tensor(y_val).long())
        test_dataset = TensorDataset(_normalize_features(X_test), torch.tensor(y_test).long())
        split_strategy = "patient_level_balanced_search"
        split_patient_ids = {
            "train": [patient_ids[index] for index in selection.train_indices],
            "validation": [patient_ids[index] for index in selection.val_indices],
            "test": [patient_ids[index] for index in selection.test_indices],
        }
        split_score = float(selection.score)
    else:
        print(
            "[WARNING] Fewer than 3 subjects supplied. Using a window-level split "
            "for smoke testing only."
        )
        all_features = np.concatenate([item[0] for item in patient_datasets], axis=0)
        all_labels = np.concatenate([item[1] for item in patient_datasets], axis=0)
        full_dataset = TensorDataset(
            _normalize_features(all_features),
            torch.tensor(all_labels, dtype=torch.long),
        )
        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, seed=seed)
        split_strategy = "window_level_smoke_test"
        split_patient_ids = {"train": patient_ids, "validation": [], "test": []}
        split_score = float("nan")

    source_path = Path(data_path)
    dataset_name = source_path.stem if source_path.is_file() else f"{source_path.name}-multi"
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "sampling_rate": sampling_rate,
        "patient_count": len(files),
        "split_strategy": split_strategy,
        "split_patient_ids": split_patient_ids,
        "split_score": split_score,
        "run_dataset_name": dataset_name,
        "input_files": files,
    }


def _calculate_class_weights(train_dataset, device):
    labels = get_labels_from_split(train_dataset)
    class_counts = np.bincount(labels, minlength=N_CLASSES)
    present_mask = class_counts > 0
    if not np.all(present_mask):
        missing = [CLASS_NAMES[index] for index in np.flatnonzero(~present_mask)]
        print("[WARNING] Training split missing classes: " + ", ".join(missing))

    weights = np.zeros(N_CLASSES, dtype=np.float32)
    present_class_count = int(present_mask.sum())
    weights[present_mask] = len(labels) / (
        present_class_count * class_counts[present_mask]
    )
    return torch.tensor(weights, dtype=torch.float32, device=device), class_counts, weights


def train(
    data_path="data/processed/tusz",
    epochs=20,
    lr=0.001,
    batch_size=32,
    max_patients=None,
    window_overlap_frac=0.5,
    model_name="cnn",
    seed=DEFAULT_SEED,
    deterministic=True,
    run_name_suffix=None,
):
    """Train, validate, tune alarm threshold, and evaluate one experiment."""
    _set_reproducible_seed(seed=seed, deterministic=deterministic)
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Compute device: {device}")
    print(f"[INFO] Random seed: {seed} | deterministic={deterministic}")

    prepared = _prepare_datasets(data_path, max_patients=max_patients, seed=seed)
    train_dataset = prepared["train_dataset"]
    val_dataset = prepared["val_dataset"]
    test_dataset = prepared["test_dataset"]
    sampling_rate = prepared["sampling_rate"]
    dataset_name = prepared["run_dataset_name"]

    _print_split_distribution("Train", train_dataset)
    _print_split_distribution("Validation", val_dataset)
    _print_split_distribution("Test", test_dataset)

    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_features, _ = train_dataset[0]
    n_channels = int(sample_features.shape[-2])
    n_timepoints = int(sample_features.shape[-1])
    window_duration_s = n_timepoints / sampling_rate
    window_step_s = window_duration_s * (1.0 - window_overlap_frac)

    print("  Normalization   : per-window, per-channel z-score")
    print(f"  Input shape     : channels={n_channels}, timepoints={n_timepoints}")
    print(f"  Window duration : {window_duration_s:.3f} seconds")
    print(f"  Window step     : {window_step_s:.3f} seconds")

    if model_name == "tcn":
        model = SeizureTCN(n_channels, n_timepoints, n_classes=N_CLASSES).to(device)
        model_display_name = "SeizureTCN"
    else:
        model = SeizureCNN(n_channels, n_timepoints, n_classes=N_CLASSES).to(device)
        model_display_name = "SeizureCNN"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights_tensor, train_class_counts, class_weights = _calculate_class_weights(
        train_dataset, device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("  Class imbalance : weighted CrossEntropyLoss")
    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"    {class_name:<12} count={train_class_counts[index]:7d} "
            f"weight={class_weights[index]:.4f}"
        )

    best_val_f1 = -1.0
    best_epoch = 0
    best_model_state = None

    run_name = f"{dataset_name}_{model_name}"
    if run_name_suffix:
        run_name = f"{run_name}_{run_name_suffix}"

    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("EEG-Seizure-Detection")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "sampling_rate_hz": sampling_rate,
                "window_duration_s": window_duration_s,
                "window_overlap_frac": window_overlap_frac,
                "window_step_s": window_step_s,
                "channels": n_channels,
                "timepoints": n_timepoints,
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "model": model_name,
                "dataset": dataset_name,
                "patient_count": prepared["patient_count"],
                "split_strategy": prepared["split_strategy"],
                "split_balance_score": prepared["split_score"],
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "n_classes": N_CLASSES,
                "random_seed": seed,
                "split_seed": seed,
                "dataloader_seed": seed,
                "deterministic_algorithms": deterministic,
                "primary_prediction_rule": "multiclass_argmax",
                "alarm_threshold_source": "validation_alarm_f1",
            }
        )
        mlflow.set_tags(
            {
                "train_patient_ids": ",".join(prepared["split_patient_ids"]["train"]),
                "validation_patient_ids": ",".join(prepared["split_patient_ids"]["validation"]),
                "test_patient_ids": ",".join(prepared["split_patient_ids"]["test"]),
            }
        )

        for class_index in range(N_CLASSES):
            mlflow.log_param(f"train_class_{class_index}_count", int(train_class_counts[class_index]))
            mlflow.log_param(f"class_{class_index}_weight", float(class_weights[class_index]))

        for epoch in range(epochs):
            model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_samples = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                current_batch_size = xb.size(0)
                train_loss_total += loss.item() * current_batch_size
                train_correct += (outputs.argmax(dim=1) == yb).sum().item()
                train_samples += current_batch_size

            train_loss = train_loss_total / train_samples
            train_accuracy = train_correct / train_samples
            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                window_step_s=window_step_s,
                decision_threshold=0.5,
                smoothing_window=0,
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                print(
                    f"      [BEST] epoch={best_epoch}, "
                    f"val_multiclass_macro_f1={best_val_f1:.4f}"
                )

            mlflow.log_metrics(
                {"train_loss": train_loss, "train_accuracy": train_accuracy},
                step=epoch + 1,
            )
            log_evaluation_metrics_to_mlflow(
                prefix="val_primary",
                metrics=val_metrics,
                step=epoch + 1,
                class_names=CLASS_NAMES,
            )
            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_multiclass_macro_f1={val_metrics['f1']:.4f}"
            )

        if best_model_state is None:
            raise RuntimeError("Training completed without a validation checkpoint.")
        model.load_state_dict(best_model_state)
        print(
            f"\n[PASS] Restored best validation model from epoch {best_epoch} "
            f"with multiclass macro F1={best_val_f1:.4f}"
        )

        threshold_result = find_best_threshold(
            model=model,
            loader=val_loader,
            criterion=criterion,
            window_step_s=window_step_s,
            metric_name="alarm_f1",
            smoothing_window=0,
        )
        best_threshold = float(threshold_result["best_threshold"])
        best_val_alarm_metrics = threshold_result["best_metrics"]
        print(
            f"[PASS] Validation-selected alarm threshold: {best_threshold:.2f} "
            f"(validation alarm F1={threshold_result['best_score']:.4f})"
        )
        mlflow.log_metric("selected_alarm_threshold", best_threshold)
        log_evaluation_metrics_to_mlflow(
            prefix="val_tuned_alarm",
            metrics=best_val_alarm_metrics,
            class_names=CLASS_NAMES,
        )

        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            window_step_s=window_step_s,
            decision_threshold=best_threshold,
            smoothing_window=0,
        )
        baseline_metrics = compute_majority_class_baseline(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            window_step_s=window_step_s,
        )

        evaluation_dir = PROJECT_ROOT / "reports" / "evaluation"
        table_dir = PROJECT_ROOT / "reports" / "tables"
        report_path = save_test_metrics_report(
            test_metrics, evaluation_dir, run_name, baseline_metrics, CLASS_NAMES
        )
        confusion_png, confusion_csv = save_confusion_matrix_plot(
            test_metrics, evaluation_dir, run_name, CLASS_NAMES
        )
        table_csv, table_md = save_cnn_baseline_results_table(
            test_metrics, baseline_metrics, table_dir, run_name, model_display_name
        )
        sweep_csv, sweep_md = save_threshold_sweep_table(
            threshold_result["sweep_rows"], table_dir, run_name
        )
        split_manifest_csv = save_split_manifest(
            prepared["split_patient_ids"], table_dir, run_name
        )

        log_evaluation_metrics_to_mlflow(
            prefix="test_primary_and_alarm",
            metrics=test_metrics,
            class_names=CLASS_NAMES,
        )
        log_evaluation_metrics_to_mlflow(
            prefix="baseline",
            metrics=baseline_metrics,
            class_names=CLASS_NAMES,
        )
        mlflow.log_metric("best_val_multiclass_f1", best_val_f1)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("training_elapsed_seconds", time.perf_counter() - start_time)

        checkpoint_filename = "seizure_tcn.pt" if model_name == "tcn" else "seizure_cnn.pt"
        checkpoint_path = PROJECT_ROOT / "models" / checkpoint_filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_display_name,
                "model_state_dict": model.state_dict(),
                "n_channels": n_channels,
                "n_timepoints": n_timepoints,
                "n_classes": N_CLASSES,
                "alarm_decision_threshold": best_threshold,
                "best_epoch": best_epoch,
                "best_val_multiclass_f1": best_val_f1,
                "sampling_rate": sampling_rate,
                "window_overlap_frac": window_overlap_frac,
                "class_weights": class_weights.tolist(),
                "class_names": CLASS_NAMES,
                "split_strategy": prepared["split_strategy"],
                "split_balance_score": prepared["split_score"],
                "split_patient_ids": prepared["split_patient_ids"],
                "patient_count": prepared["patient_count"],
                "random_seed": seed,
                "split_seed": seed,
                "dataloader_seed": seed,
                "deterministic_algorithms": deterministic,
                "primary_prediction_rule": "multiclass_argmax",
            },
            checkpoint_path,
        )

        artifacts = [
            report_path,
            confusion_png,
            confusion_csv,
            table_csv,
            table_md,
            sweep_csv,
            sweep_md,
            split_manifest_csv,
            checkpoint_path,
        ]
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="final_outputs")

        try:
            mlflow.pytorch.log_model(
                model,
                name=f"seizure_{model_name}",
                serialization_format="pickle",
            )
        except TypeError:
            mlflow.pytorch.log_model(
                model,
                artifact_path=f"seizure_{model_name}",
                serialization_format="pickle",
            )

        print("\nFinal Held-Out Test Results")
        print("---------------------------")
        print("Primary three-class task (argmax; threshold-independent)")
        print(f"  Accuracy                 : {test_metrics['accuracy']:.4f}")
        print(f"  Balanced accuracy        : {test_metrics['balanced_accuracy']:.4f}")
        print(f"  Macro precision          : {test_metrics['precision']:.4f}")
        print(f"  Macro recall             : {test_metrics['recall_sensitivity']:.4f}")
        print(f"  Macro specificity        : {test_metrics['specificity']:.4f}")
        print(f"  Macro F1                 : {test_metrics['f1']:.4f}")
        print(
            "  Macro AUC                : "
            + (f"{test_metrics['auc']:.4f}" if np.isfinite(test_metrics["auc"]) else "N/A")
        )
        for index, class_name in enumerate(CLASS_NAMES):
            print(
                f"  {class_name:<12} recall={test_metrics['per_class_recall'][index]:.4f} "
                f"f1={test_metrics['per_class_f1'][index]:.4f}"
            )
        print("Secondary alarm task (validation-selected threshold)")
        print(f"  Threshold                : {best_threshold:.2f}")
        print(f"  Alarm F1                 : {test_metrics['alarm_f1']:.4f}")
        print(f"  Alarm sensitivity        : {test_metrics['alarm_recall_sensitivity']:.4f}")
        print(f"  Alarm specificity        : {test_metrics['alarm_specificity']:.4f}")
        print(f"  False-positive windows/hr: {test_metrics['false_alarms_per_hour']:.4f}")
        print(f"Checkpoint                 : {checkpoint_path}")
        print(f"Report                     : {report_path}")
        print(f"Threshold sweep            : {sweep_csv}")
        print(f"Patient split manifest     : {split_manifest_csv}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EEG seizure model.")
    parser.add_argument("--data", default="data/processed/tusz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--window-overlap-frac", type=float, default=0.5)
    parser.add_argument("--model", choices=["cnn", "tcn"], default="cnn")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic PyTorch algorithms (default: enabled).",
    )
    parser.add_argument(
        "--run-name-suffix",
        default=None,
        help="Optional suffix to prevent report/checkpoint artifact-name collisions.",
    )
    args = parser.parse_args()

    sys.exit(
        train(
            data_path=args.data,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_patients=args.max_patients,
            window_overlap_frac=args.window_overlap_frac,
            model_name=args.model,
            seed=args.seed,
            deterministic=args.deterministic,
            run_name_suffix=args.run_name_suffix,
        )
    )
