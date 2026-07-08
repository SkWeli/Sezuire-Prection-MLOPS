"""
Baseline seizure detection training script with MLflow tracking.

This file handles:
- loading processed EEG windows
- splitting data into train/validation/test
- training the baseline CNN
- evaluating model performance
- logging metrics and artifacts to MLflow

The CNN model architecture is stored separately in:
    src/models/cnn.py
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
from torch.utils.data import DataLoader, TensorDataset, random_split


# Project root is needed so this file works when run directly:
# python src/training/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.cnn import SeizureCNN
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
from src.evaluation.splits import split_dataset

def train(data_path=None, epochs=20, lr=0.001, batch_size=32, max_patients=None, window_overlap_frac=0.5):
    """
    Train the baseline seizure detection model.

    window_overlap_frac:
        The overlap used during EEG window creation.
        Example:
            0.5 means 50% overlap.

    Why this is needed:
        False alarms per hour depends on how much EEG time each test window represents.
    """
    
    import time
    from pathlib import Path

    start_time = time.perf_counter()

    if max_patients is not None:
        processed_dir = Path("data/processed/tusz")
        npz_files = sorted(processed_dir.rglob("*.npz"))[:max_patients]
        print(f"🚀 Loading {len(npz_files)} patients (max {max_patients})")
        all_X, all_y = [], []
        sampling_rate = None
        for npz_file in npz_files:
            print(f"   📂 {npz_file.name}")
            data = np.load(npz_file)
            if sampling_rate is None:
                # If the metadata is missing, safely use 128.0 as the default.
                sampling_rate = float(data["sfreq"]) if "sfreq" in data.files else 128.0
            all_X.append(data["epochs"])
            all_y.append(data["labels"])
            print(f"     {len(data['labels'])} windows, {data['n_seizures']} seizures")

        data_path_str = "tusz-multi"
        X = torch.tensor(np.concatenate(all_X)).float().unsqueeze(1)  # (N,1,C,T)
        y = torch.tensor(np.concatenate(all_y)).long()
    else:
        data_path = Path(data_path)
        print(f"📂 Loading: {data_path}")

        data = np.load(data_path)

        X = torch.tensor(data["epochs"]).float().unsqueeze(1)
        y = torch.tensor(data["labels"]).long()

        data_path_str = data_path.stem

        # Sampling rate is used to calculate the time duration represented by windows.
        sampling_rate = float(data["sfreq"]) if "sfreq" in data.files else 128.0

    print(f"  X: {X.shape} | y: {y.shape} | seizures: {y.sum().item()}")

    # Each EEG window has shape: (channels, timepoints).
    # X shape is: (N, 1, channels, timepoints)
    n_timepoints = X.shape[-1]

    # Window duration in seconds.
    # Example: 512 timepoints / 128 Hz = 4 seconds.
    window_duration_s = n_timepoints / sampling_rate

    # Because windows overlap, each new window does not represent the full duration.
    # With 50% overlap, a 4-second window moves forward by 2 seconds.
    window_step_s = window_duration_s * (1.0 - window_overlap_frac)

    print(f"  Sampling rate   : {sampling_rate:.1f} Hz")
    print(f"  Window duration : {window_duration_s:.2f} seconds")
    print(f"  Window step     : {window_step_s:.2f} seconds")

    dataset = TensorDataset(X, y)

    # Split the full dataset before training.
    # This prevents reporting accuracy on the same data used for learning.
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Training loader is shuffled because the model should see batches in random order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_dataset)}")
    print(f"  Test samples : {len(test_dataset)}")

    model = SeizureCNN() # Create the CNN model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam optimizer updates model weights during training.
    # Calculate class weights from the training split only.
    # This prevents the model from being biased toward the majority class.
    train_labels = get_labels_from_split(train_dataset)
    train_class_counts = np.bincount(train_labels, minlength=2)

    # Weighted CrossEntropyLoss gives more importance to the minority class.
    # Example:
    # If non-seizure windows are fewer, class 0 receives a higher weight.
    class_weights = len(train_labels) / (2.0 * np.maximum(train_class_counts, 1))

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("  Class imbalance handling: weighted CrossEntropyLoss")
    print(f"  Train class counts      : non-seizure={train_class_counts[0]}, seizure={train_class_counts[1]}")
    print(f"  Class weights           : non-seizure={class_weights[0]:.3f}, seizure={class_weights[1]:.3f}")

    MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
    MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlartifacts"
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("EEG-Seizure-Detection")

    with mlflow.start_run():
        mlflow.log_params({
        # EEG window and evaluation settings
        "sampling_rate_hz": sampling_rate,
        "window_duration_s": window_duration_s,
        "window_overlap_frac": window_overlap_frac,
        "window_step_s": window_step_s,
        "evaluation_level": "window_level",
        "false_alarm_definition": "false_positive_windows_per_evaluated_hour",

        # Training configuration
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,

        # Model and dataset information
        "model": "SeizureCNN",
        "dataset": data_path_str,
        "n_samples": len(X),
        "n_seizures": int(y.sum()),

        # Data split configuration
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,

        # Actual number of samples in each split.
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),

        # This makes it clear that final accuracy is reported from the test set,
        # not from the training set.
        "reported_accuracy_source": "held_out_test_set",
        "reported_accuracy_metric": "final_test_accuracy",
    })

        for epoch in range(epochs):
            # Put model in training mode.
            model.train()

            train_loss_total = 0.0
            train_correct = 0
            train_samples = 0

            for xb, yb in train_loader:
                # Clear old gradients from the previous batch.
                optimizer.zero_grad()

                # Forward pass:
                # The model predicts seizure/non-seizure for the EEG batch.
                outputs = model(xb)

                # Calculate classification loss.
                loss = criterion(outputs, yb)

                # Backward pass:
                # Calculate gradients based only on training data.
                loss.backward()

                # Update model weights.
                optimizer.step()

                batch_size_current = xb.size(0)

                # Accumulate loss and accuracy information for this epoch.
                train_loss_total += loss.item() * batch_size_current
                train_correct += (outputs.argmax(1) == yb).sum().item()
                train_samples += batch_size_current

            # Average training loss and accuracy for the full epoch.
            train_loss = train_loss_total / train_samples
            train_acc = train_correct / train_samples

            # Evaluate on validation data after each epoch.
            # Validation metrics show how well the model performs on unseen data
            # during training, without updating the model weights.
            # During training, validation is shown using the default 0.5 threshold.
            # After training, we tune the threshold using the full validation split.
            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                window_step_s=window_step_s,
                decision_threshold=0.5
            )
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            # Log both training and validation metrics to MLflow.
            # This allows comparison between learning performance and generalization.
            # Log training metrics for this epoch.
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
            }, step=epoch + 1)

            # Log all validation evaluation metrics with the prefix "val".
            # This includes accuracy, precision, recall, specificity, F1, AUC,
            # confusion matrix values, and false alarms per hour.
            log_evaluation_metrics_to_mlflow(
                prefix="val",
                metrics=val_metrics,
                step=epoch + 1
            )

            # AUC can be NaN if the split contains only one class.
            # MLflow should log it only when it is valid.
            if not np.isnan(val_metrics["auc"]):
                mlflow.log_metric("val_auc", val_metrics["auc"], step=epoch + 1)

            print(
                f"   Epoch {epoch + 1}/{epochs} "
                f"— train_loss: {train_loss:.4f} "
                f"train_acc: {train_acc:.3f} "
                f"val_loss: {val_loss:.4f} "
                f"val_acc: {val_metrics['accuracy']:.3f} "
                f"val_f1: {val_metrics['f1']:.3f}"
            )
        # Final test evaluation.
        # Tune the decision threshold using validation data.
        # We select the threshold that gives the best balanced accuracy.
        threshold_result = find_best_threshold(
            model=model,
            loader=val_loader,
            criterion=criterion,
            window_step_s=window_step_s,
            metric_name="balanced_accuracy"
        )

        best_threshold = threshold_result["best_threshold"]
        best_val_threshold_metrics = threshold_result["best_metrics"]

        print("\nThreshold Tuning")
        print("----------------")
        print(f"Selection metric             : {threshold_result['metric_name']}")
        print(f"Best validation threshold    : {best_threshold:.2f}")
        print(f"Best validation balanced acc : {threshold_result['best_score']:.3f}")
        
        # Final test evaluation uses the threshold selected from validation data.
        # The test set is not used to choose the threshold.
        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            window_step_s=window_step_s,
            decision_threshold=best_threshold
        )
        
        final_reported_accuracy = test_metrics["accuracy"]

        # Majority-class baseline comparison.
        # This baseline uses only the training labels to decide the most common class,
        # then predicts that class for every test window.
        baseline_metrics = compute_majority_class_baseline(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            window_step_s=window_step_s
        )

        # Save a simple CNN vs majority-class baseline result table.
        # This table is useful for the thesis and final evaluation discussion.
        table_csv_path, table_md_path = save_cnn_baseline_results_table(
            test_metrics=test_metrics,
            baseline_metrics=baseline_metrics,
            output_dir=PROJECT_ROOT / "reports" / "tables",
            run_name=data_path_str
        )

        # Log all final test metrics with the prefix "test".
        # These are the official final evaluation metrics.
        log_evaluation_metrics_to_mlflow(
            prefix="test",
            metrics=test_metrics
        )

        # Log validation threshold tuning result.
        log_evaluation_metrics_to_mlflow(
            prefix="val_best_threshold",
            metrics=best_val_threshold_metrics
        )

        mlflow.log_metric("best_decision_threshold", best_threshold)
        mlflow.log_metric(
            "best_val_balanced_accuracy",
            threshold_result["best_score"]
        )

        # Log majority-class baseline metrics.
        # These allow direct comparison between the CNN and a simple baseline.
        log_evaluation_metrics_to_mlflow(
            prefix="baseline",
            metrics=baseline_metrics
        )

        mlflow.log_params({
            "baseline_type": "majority_class",
            "baseline_majority_class": baseline_metrics["majority_class"],
            "baseline_train_non_seizure_count": baseline_metrics["train_non_seizure_count"],
            "baseline_train_seizure_count": baseline_metrics["train_seizure_count"],

            # Class imbalance handling
            "class_imbalance_handling": "weighted_cross_entropy",
            "class_0_weight_non_seizure": float(class_weights[0]),
            "class_1_weight_seizure": float(class_weights[1]),
            "train_non_seizure_count": int(train_class_counts[0]),
            "train_seizure_count": int(train_class_counts[1]),

            # Threshold tuning
            "threshold_tuning": "validation_set_grid_search",
            "threshold_selection_metric": "balanced_accuracy",
        })

        # Log model improvement over the baseline.
        mlflow.log_metrics({
            "model_vs_baseline_accuracy_delta": (
                test_metrics["accuracy"] - baseline_metrics["accuracy"]
            ),
            "model_vs_baseline_f1_delta": (
                test_metrics["f1"] - baseline_metrics["f1"]
            ),
            "model_vs_baseline_false_alarm_delta": (
                test_metrics["false_alarms_per_hour"]
                - baseline_metrics["false_alarms_per_hour"]
            ),
        })

        # Log final reported accuracy separately for easy visibility in MLflow.
        mlflow.log_metric("final_test_accuracy", final_reported_accuracy)

        # Save a readable test report and attach it to the MLflow run.
        report_path = save_test_metrics_report(
            metrics=test_metrics,
            output_dir=PROJECT_ROOT / "reports" / "metrics",
            run_name=data_path_str,
            baseline_metrics=baseline_metrics
        )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="reports"
        )

        # Log result tables as MLflow artifacts.
        # These can be downloaded later from the MLflow run page.
        mlflow.log_artifact(
            str(table_csv_path),
            artifact_path="reports/tables"
        )

        mlflow.log_artifact(
            str(table_md_path),
            artifact_path="reports/tables"
        )
        
        # AUC can be NaN if the test split has only one class.
        # We log it only when it is mathematically valid.
        if not np.isnan(test_metrics["auc"]):
            mlflow.log_metric("test_auc", test_metrics["auc"])

        print("\nFinal Test Results")
        print("------------------")
        print(f"Test loss                    : {test_metrics['loss']:.4f}")
        print(f"Test accuracy                : {test_metrics['accuracy']:.3f}")
        print(f"Final reported accuracy      : {final_reported_accuracy:.3f}")
        print(f"Balanced accuracy            : {test_metrics['balanced_accuracy']:.3f}")
        print(f"Decision threshold           : {test_metrics['decision_threshold']:.2f}")
        print(f"Precision                    : {test_metrics['precision']:.3f}")
        print(f"Recall / Sensitivity         : {test_metrics['recall_sensitivity']:.3f}")
        print(f"Specificity                  : {test_metrics['specificity']:.3f}")
        print(f"F1-score                     : {test_metrics['f1']:.3f}")
        print(f"Evaluated test duration hrs  : {test_metrics['evaluated_duration_hours']:.4f}")
        print(f"False alarms per hour        : {test_metrics['false_alarms_per_hour']:.2f}")

        if not np.isnan(test_metrics["auc"]):
            print(f"AUC                          : {test_metrics['auc']:.3f}")
        else:
            print("AUC                          : Not available, only one class in test split")

        print("\nConfusion Matrix")
        print("----------------")
        print("                 Predicted")
        print("               Non-Seizure  Seizure")
        print(f"Actual Non-Seizure   {test_metrics['tn']:>6}   {test_metrics['fp']:>6}")
        print(f"Actual Seizure       {test_metrics['fn']:>6}   {test_metrics['tp']:>6}")

        print("\nMajority-Class Baseline")
        print("-----------------------")
        print(
            f"Majority class                : "
            f"{baseline_metrics['majority_class']} "
            f"(0=non-seizure, 1=seizure)"
        )
        print(f"Baseline accuracy             : {baseline_metrics['accuracy']:.3f}")
        print(f"Baseline precision            : {baseline_metrics['precision']:.3f}")
        print(f"Baseline recall/sensitivity   : {baseline_metrics['recall_sensitivity']:.3f}")
        print(f"Baseline specificity          : {baseline_metrics['specificity']:.3f}")
        print(f"Baseline F1-score             : {baseline_metrics['f1']:.3f}")

        if np.isfinite(baseline_metrics["auc"]):
            print(f"Baseline AUC                  : {baseline_metrics['auc']:.3f}")
        else:
            print("Baseline AUC                  : Not available")

        print(f"Baseline false alarms/hour    : {baseline_metrics['false_alarms_per_hour']:.2f}")

        print("\nModel vs Baseline")
        print("-----------------")
        print(
            f"Accuracy difference           : "
            f"{test_metrics['accuracy'] - baseline_metrics['accuracy']:+.3f}"
        )
        print(
            f"F1-score difference           : "
            f"{test_metrics['f1'] - baseline_metrics['f1']:+.3f}"
        )
        print(
            f"False alarms/hour difference  : "
            f"{test_metrics['false_alarms_per_hour'] - baseline_metrics['false_alarms_per_hour']:+.2f}"
        )

        print("\nSaved Result Tables")
        print("-------------------")
        print(f"CSV table      : {table_csv_path}")
        print(f"Markdown table : {table_md_path}")

        print("\nReported accuracy source     : held-out test set")

        # Save a local PyTorch checkpoint.
        # This allows evaluate.py to load the trained model later without retraining.
        model_output_dir = PROJECT_ROOT / "models"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_checkpoint_path = model_output_dir / "seizure_cnn.pt"

        torch.save(
            {
                "model_name": "SeizureCNN",
                "model_state_dict": model.state_dict(),

                # Store input shape details so evaluate.py can rebuild the model correctly.
                "n_channels": int(X.shape[2]),
                "n_timepoints": int(X.shape[3]),
                "n_classes": 2,

                # Store window information used for false alarm calculation.
                "sampling_rate_hz": sampling_rate,
                "window_duration_s": window_duration_s,
                "window_overlap_frac": window_overlap_frac,
                "window_step_s": window_step_s,

                # Store tuned threshold so evaluate.py can use the same decision rule.
                "decision_threshold": best_threshold,
                "threshold_selection_metric": "balanced_accuracy",
            },
            model_checkpoint_path
        )

        # Log the checkpoint as an MLflow artifact as well.
        mlflow.log_artifact(
            str(model_checkpoint_path),
            artifact_path="models"
        )

        mlflow.pytorch.log_model(model, name="seizure_cnn")
        print("✅ Model logged to MLflow!")
        print(f"✅ Local model checkpoint saved: {model_checkpoint_path}")
        print(f"✅ Run: mlflow ui → http://127.0.0.1:5000")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed/tusz/aaaaaajy/aaaaaajy.npz")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--max-patients", type=int, default=None, help="Max number of patients (npz files) to load; None uses --data single file")
    parser.add_argument("--window-overlap-frac", type=float, default=0.5, help="Window overlap fraction used during preprocessing. Default 0.5 means 50% overlap.")
    args = parser.parse_args()
    sys.exit(train(args.data, args.epochs, args.lr, args.batch_size, args.max_patients, args.window_overlap_frac))