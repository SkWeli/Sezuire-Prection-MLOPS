"""
Baseline seizure detection training script with MLflow tracking.
"""

import argparse
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

from src.models.cnn import SeizureCNN
from src.models.tcn import SeizureTCN

from src.evaluation.metrics import (
    evaluate_model,
    find_best_threshold,
    get_labels_from_split,
    compute_majority_class_baseline,
)
from src.evaluation.reporting import (
    log_evaluation_metrics_to_mlflow,
    save_test_metrics_report,
    save_cnn_baseline_results_table,
)
from src.evaluation.splits import split_dataset, split_dataset_by_patient

def train(data_path=None, epochs=20, lr=0.001, batch_size=32, max_patients=None, window_overlap_frac=0.5, model_name="cnn"):
    start_time = time.perf_counter()
    patient_level_split = False  

    # Define GPU device targets dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Host Cluster Compute Device Identified: {device}")

    if max_patients is not None:
        processed_dir = Path("data/processed/tusz")
        npz_files = sorted(processed_dir.rglob("*.npz"))[:max_patients]
        print(f"🚀 Loading {len(npz_files)} patients (max {max_patients})")

        patient_datasets = []
        sampling_rate = None
        for npz_file in npz_files:
            print(f"   📂 {npz_file.name}")
            data = np.load(npz_file)
            if sampling_rate is None:
                sampling_rate = float(data["sfreq"]) if "sfreq" in data.files else 128.0
            patient_datasets.append((data["epochs"], data["labels"]))

        data_path_str = "tusz-multi"

        if len(patient_datasets) >= 3:
            (X_train_np, y_train_np), (X_val_np, y_val_np), (X_test_np, y_test_np) = \
                split_dataset_by_patient(patient_datasets)
            patient_level_split = True

            X = torch.tensor(np.concatenate([p[0] for p in patient_datasets])).float().unsqueeze(1)
            y = torch.tensor(np.concatenate([p[1] for p in patient_datasets])).long()
        else:
            print("  ⚠️ Fewer than 3 patients loaded — falling back to window-level split.")
            all_X = [p[0] for p in patient_datasets]
            all_y = [p[1] for p in patient_datasets]
            X = torch.tensor(np.concatenate(all_X)).float().unsqueeze(1)
            y = torch.tensor(np.concatenate(all_y)).long()
    else:
        data_path = Path(data_path)
        print(f"📂 Loading: {data_path}")
        data = np.load(data_path)

        X = torch.tensor(data["epochs"]).float().unsqueeze(1)
        y = torch.tensor(data["labels"]).long()
        data_path_str = data_path.stem
        sampling_rate = float(data["sfreq"]) if "sfreq" in data.files else 128.0

    if not patient_level_split:
        mean = X.mean(dim=-1, keepdim=True)   
        std = X.std(dim=-1, keepdim=True) + 1e-6  
        X = (X - mean) / std

    print(f"  Normalization   : per-window z-score (mean=0, std=1 per channel)")
    print(f"  X: {X.shape} | y: {y.shape}")

    n_channels = X.shape[2]
    n_timepoints = X.shape[-1]

    window_duration_s = n_timepoints / sampling_rate
    window_step_s = window_duration_s * (1.0 - window_overlap_frac)

    if patient_level_split:
        def _normalize(X_np):
            X_t = torch.tensor(X_np).float().unsqueeze(1)  
            mean = X_t.mean(dim=-1, keepdim=True)
            std = X_t.std(dim=-1, keepdim=True) + 1e-6
            return (X_t - mean) / std

        X_train_t = _normalize(X_train_np)
        X_val_t = _normalize(X_val_np)
        X_test_t = _normalize(X_test_np)

        y_train_t = torch.tensor(y_train_np).long()
        y_val_t = torch.tensor(y_val_np).long()
        y_test_t = torch.tensor(y_test_np).long()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
    else:
        dataset = TensorDataset(X, y)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model and immediately push onto the CUDA processing core
    if model_name == "tcn":
        model = SeizureTCN(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=3).to(device)
    else:
        model = SeizureCNN(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=3).to(device)

    print(f"  Model architecture bound to target hardware : {model_name.upper()} -> {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_labels = get_labels_from_split(train_dataset)
    train_class_counts = np.bincount(train_labels, minlength=3)

    total_samples = len(train_labels)
    n_classes = 3
    class_weights = total_samples / (n_classes * np.maximum(train_class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("  Class imbalance handling: weighted CrossEntropyLoss (Ternary balance)")
    print(f"  Train class weights calculated: Interictal={class_weights[0]:.3f}, Pre-Ictal={class_weights[1]:.3f}, Ictal={class_weights[2]:.3f}")
    
    MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
    MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlartifacts"
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("EEG-Seizure-Detection")

    with mlflow.start_run():
        mlflow.log_params({
            "sampling_rate_hz": sampling_rate,
            "window_duration_s": window_duration_s,
            "window_overlap_frac": window_overlap_frac,
            "window_step_s": window_step_s,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "model": model_name,
            "dataset": data_path_str,
            "n_classes": 3
        })

        for epoch in range(epochs):
            model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_samples = 0

            for xb, yb in train_loader:
                # Push data batches to the GPU engine safely
                xb, yb = xb.to(device), yb.to(device)
                
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                batch_size_current = xb.size(0)
                train_loss_total += loss.item() * batch_size_current
                train_correct += (outputs.argmax(1) == yb).sum().item()
                train_samples += batch_size_current

            train_loss = train_loss_total / train_samples
            train_acc = train_correct / train_samples

            val_metrics = evaluate_model(
                model=model, loader=val_loader, criterion=criterion,
                window_step_s=window_step_s, decision_threshold=0.5
            )

            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch + 1)
            log_evaluation_metrics_to_mlflow(prefix="val", metrics=val_metrics, step=epoch + 1)

            print(
                f"   Epoch {epoch + 1}/{epochs} "
                f"— train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} "
                f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.3f} val_f1: {val_metrics['f1']:.3f}"
            )

        threshold_result = find_best_threshold(
            model=model, loader=val_loader, criterion=criterion,
            window_step_s=window_step_s, metric_name="f1", smoothing_window=5
        )
        best_threshold = threshold_result["best_threshold"]
        best_val_threshold_metrics = threshold_result["best_metrics"]

        test_metrics = evaluate_model(
            model=model, loader=test_loader, criterion=criterion,
            window_step_s=window_step_s, decision_threshold=best_threshold, smoothing_window=5
        )
        
        baseline_metrics = compute_majority_class_baseline(
            train_dataset=train_dataset, test_dataset=test_dataset, window_step_s=window_step_s
        )

        table_csv_path, table_md_path = save_cnn_baseline_results_table(
            test_metrics=test_metrics, baseline_metrics=baseline_metrics,
            output_dir=PROJECT_ROOT / "reports" / "tables", run_name=f"{data_path_str}_{model_name}"
        )

        log_evaluation_metrics_to_mlflow(prefix="test", metrics=test_metrics)
        log_evaluation_metrics_to_mlflow(prefix="baseline", metrics=baseline_metrics)

        mlflow.log_metric("final_test_accuracy", test_metrics["accuracy"])
        
        checkpoint_filename = "seizure_tcn.pt" if model_name == "tcn" else "seizure_cnn.pt"
        model_checkpoint_path = PROJECT_ROOT / "models" / checkpoint_filename
        model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_name": "SeizureTCN" if model_name == "tcn" else "SeizureCNN",
                "model_state_dict": model.state_dict(),
                "n_channels": int(n_channels),
                "n_timepoints": int(n_timepoints),
                "n_classes": 3,
                "decision_threshold": best_threshold,
            },
            model_checkpoint_path
        )

        mlflow.pytorch.log_model(model, name=f"seizure_{model_name}")
        print(f"✅ Model successfully trained and saved onto node: {model_checkpoint_path}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed/tusz/aaaaaajy/aaaaaajy.npz")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--window-overlap-frac", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "tcn"])
    args = parser.parse_args()
    sys.exit(train(args.data, args.epochs, args.lr, args.batch_size, args.max_patients, args.window_overlap_frac, args.model))