"""
Standalone evaluation script for trained EEG seizure detection models.

Usage:
    python src/evaluation/evaluate.py \
        --data data/processed/tusz/aaaaaajy/aaaaaajy.npz \
        --model models/seizure_cnn.pt

Why this script exists:
- Training and evaluation should be possible separately.
- A saved model can be evaluated again without retraining.
- Later, this script can be reused for CHB-MIT, cross-dataset testing,
  robustness testing, and quantized model comparison.
"""

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Make project imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.cnn import SeizureCNN
from src.evaluation.splits import split_dataset
from src.evaluation.metrics import evaluate_model, compute_majority_class_baseline
from src.evaluation.reporting import (
    log_evaluation_metrics_to_mlflow,
    save_test_metrics_report,
    save_cnn_baseline_results_table,
)


def load_processed_npz(data_path):
    """
    Load processed EEG windows from an .npz file.

    Expected arrays:
        epochs -> EEG windows with shape (N, channels, timepoints)
        labels -> binary labels where 0 = non-seizure, 1 = seizure

    The model expects input shape:
        (N, 1, channels, timepoints)
    """

    data_path = Path(data_path)
    print(f"Loading evaluation data: {data_path}")

    data = np.load(data_path)

    X = torch.tensor(data["epochs"]).float().unsqueeze(1)
    y = torch.tensor(data["labels"]).long()

    # Sampling rate is needed for false alarms per hour.
    # If missing, use 128 Hz because our preprocessing resamples to 128 Hz.
    sampling_rate = float(data["sfreq"]) if "sfreq" in data.files else 128.0

    return X, y, sampling_rate, data_path.stem


def load_seizure_cnn_checkpoint(model_path):
    """
    Load a saved SeizureCNN checkpoint.

    The checkpoint stores:
    - model weights
    - input channel count
    - timepoint count
    - window information
    """

    model_path = Path(model_path)
    print(f"Loading model checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    model = SeizureCNN(
        n_channels=int(checkpoint.get("n_channels", 20)),
        n_timepoints=int(checkpoint.get("n_timepoints", 512)),
        n_classes=int(checkpoint.get("n_classes", 2)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint


def evaluate_saved_model(data_path, model_path, batch_size=32, window_overlap_frac=0.5):
    """
    Evaluate a saved CNN model on the held-out test split.

    The same 70/15/15 split logic is used here so that evaluation is consistent
    with the training script.
    """

    X, y, sampling_rate, run_name = load_processed_npz(data_path)

    print(f"  X: {X.shape} | y: {y.shape} | seizures: {y.sum().item()}")

    # Recalculate window duration and step for false alarm analysis.
    n_timepoints = X.shape[-1]
    window_duration_s = n_timepoints / sampling_rate
    window_step_s = window_duration_s * (1.0 - window_overlap_frac)

    print(f"  Sampling rate   : {sampling_rate:.1f} Hz")
    print(f"  Window duration : {window_duration_s:.2f} seconds")
    print(f"  Window step     : {window_step_s:.2f} seconds")

    dataset = TensorDataset(X, y)

    # Use the same split behavior as training.
    # This gives the same held-out test split when the same seed is used.
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_dataset)}")
    print(f"  Test samples : {len(test_dataset)}")

    model, checkpoint = load_seizure_cnn_checkpoint(model_path)
    criterion = nn.CrossEntropyLoss()

    # Evaluate model on the held-out test set.
    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        window_step_s=window_step_s
    )

    # Majority-class baseline uses training labels and predicts one class for test data.
    baseline_metrics = compute_majority_class_baseline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        window_step_s=window_step_s
    )

    final_reported_accuracy = test_metrics["accuracy"]

    # Save evaluation report and result table.
    report_path = save_test_metrics_report(
        metrics=test_metrics,
        output_dir=PROJECT_ROOT / "reports" / "evaluation",
        run_name=f"{run_name}_standalone",
        baseline_metrics=baseline_metrics
    )

    table_csv_path, table_md_path = save_cnn_baseline_results_table(
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        output_dir=PROJECT_ROOT / "reports" / "evaluation",
        run_name=f"{run_name}_standalone"
    )

    # Log standalone evaluation to MLflow.
    mlflow_db = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("EEG-Seizure-Detection")

    with mlflow.start_run(run_name=f"evaluate_{run_name}"):
        mlflow.log_params({
            "script": "src/evaluation/evaluate.py",
            "model_path": str(model_path),
            "data_path": str(data_path),
            "model_name": checkpoint.get("model_name", "SeizureCNN"),
            "batch_size": batch_size,
            "sampling_rate_hz": sampling_rate,
            "window_duration_s": window_duration_s,
            "window_overlap_frac": window_overlap_frac,
            "window_step_s": window_step_s,
            "evaluation_split": "held_out_test_set",
        })

        log_evaluation_metrics_to_mlflow(
            prefix="test",
            metrics=test_metrics
        )

        log_evaluation_metrics_to_mlflow(
            prefix="baseline",
            metrics=baseline_metrics
        )

        mlflow.log_metric("final_test_accuracy", final_reported_accuracy)

        mlflow.log_artifact(
            str(report_path),
            artifact_path="reports/evaluation"
        )

        mlflow.log_artifact(
            str(table_csv_path),
            artifact_path="reports/evaluation"
        )

        mlflow.log_artifact(
            str(table_md_path),
            artifact_path="reports/evaluation"
        )

    print("\nStandalone Evaluation Results")
    print("-----------------------------")
    print(f"Test accuracy                : {test_metrics['accuracy']:.3f}")
    print(f"Final reported accuracy      : {final_reported_accuracy:.3f}")
    print(f"Precision                    : {test_metrics['precision']:.3f}")
    print(f"Recall / Sensitivity         : {test_metrics['recall_sensitivity']:.3f}")
    print(f"Specificity                  : {test_metrics['specificity']:.3f}")
    print(f"F1-score                     : {test_metrics['f1']:.3f}")
    print(f"False alarms per hour        : {test_metrics['false_alarms_per_hour']:.2f}")

    if np.isfinite(test_metrics["auc"]):
        print(f"AUC                          : {test_metrics['auc']:.3f}")
    else:
        print("AUC                          : Not available")

    print("\nMajority-Class Baseline")
    print("-----------------------")
    print(
        f"Majority class                : "
        f"{baseline_metrics['majority_class']} "
        f"(0=non-seizure, 1=seizure)"
    )
    print(f"Baseline accuracy             : {baseline_metrics['accuracy']:.3f}")
    print(f"Baseline F1-score             : {baseline_metrics['f1']:.3f}")
    print(f"Baseline false alarms/hour    : {baseline_metrics['false_alarms_per_hour']:.2f}")

    print("\nSaved Evaluation Artifacts")
    print("--------------------------")
    print(f"Report table    : {report_path}")
    print(f"CSV table       : {table_csv_path}")
    print(f"Markdown table  : {table_md_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved EEG seizure detection model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to processed .npz EEG data."
    )

    parser.add_argument(
        "--model",
        default="models/seizure_cnn.pt",
        help="Path to saved PyTorch checkpoint."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation."
    )

    parser.add_argument(
        "--window-overlap-frac",
        type=float,
        default=0.5,
        help="Window overlap fraction used during preprocessing."
    )

    args = parser.parse_args()

    sys.exit(
        evaluate_saved_model(
            data_path=args.data,
            model_path=args.model,
            batch_size=args.batch_size,
            window_overlap_frac=args.window_overlap_frac
        )
    )