"""
Baseline seizure detection model with MLflow tracking.
Usage: python src/training/train.py --data data/processed/tusz/aaaaaajy/aaaaaajy.npz
"""
import argparse
import sys


import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

class SeizureCNN(nn.Module):
    """Lightweight CNN for EEG seizure detection (quantization-friendly)"""
    def __init__(self, n_channels=20, n_timepoints=512, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 25), padding=(0, 12))
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(n_channels, 1))
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.AdaptiveAvgPool2d((1, 32))
        self.fc1   = nn.Linear(32 * 32, 64)
        self.fc2   = nn.Linear(64, n_classes)
        self.drop  = nn.Dropout(0.5)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

def split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split the full EEG window dataset into train, validation, and test sets.

    The split used here is:
    - 70% training
    - 15% validation
    - 15% testing
    """

    total_size = len(dataset)

    # Calculate train and validation sizes using the given ratios.
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Calculate test size using the remaining samples.
    # This avoids losing samples due to rounding.
    test_size = total_size - train_size - val_size

    # A fixed random seed makes the split reproducible.
    # This means the same samples go into train/val/test every time we run.
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def compute_binary_auc(y_true, y_score):
    """
    Calculate binary AUC without adding an extra sklearn dependency.

    y_true:
        True labels, where 0 = non-seizure and 1 = seizure.

    y_score:
        Predicted probability for class 1, seizure.

    Why we need this:
    - AUC shows how well the model separates seizure and non-seizure windows.
    - It is useful when the dataset is imbalanced.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # AUC cannot be calculated if only one class exists in the split.
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank predicted scores.
    # Higher seizure probability should ideally receive higher rank.
    order = np.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = np.empty(len(y_score), dtype=float)

    # Average ranks are used for tied prediction scores.
    i = 0
    while i < len(sorted_scores):
        j = i

        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1

        # Ranks are 1-based, so we add 2 when averaging i and j.
        average_rank = (i + j + 2) / 2.0
        ranks[order[i:j + 1]] = average_rank

        i = j + 1

    positive_rank_sum = np.sum(ranks[y_true == 1])

    auc = (
        positive_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    ) / (n_pos * n_neg)

    return float(auc)


def evaluate_model(model, loader, criterion, window_step_s=2.0):
    """
    Evaluate the model and return classification metrics.

    This includes:
    - accuracy
    - precision
    - recall/sensitivity
    - specificity
    - F1-score
    - AUC
    - confusion matrix values
    - false alarms per hour

    false alarms per hour:
        Calculated as false positive windows per evaluated EEG hour.

    Note:
        This is currently a window-level approximation.
        Later, this can be improved into event-level false alarm counting.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []
    all_scores = []

    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)

            batch_size = xb.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Convert logits into class probabilities.
            # Class 1 is seizure, so this score is used for AUC.
            probabilities = torch.softmax(outputs, dim=1)
            seizure_scores = probabilities[:, 1]

            predictions = outputs.argmax(1)

            all_true.extend(yb.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_scores.extend(seizure_scores.cpu().numpy())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_score = np.asarray(all_scores)

    # Binary confusion matrix values.
    # Positive class = seizure.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc = compute_binary_auc(y_true, y_score)
    avg_loss = total_loss / total_samples

    # Estimate how much EEG time was evaluated.
    # Example:
    # 350 test windows * 2 second step = 700 seconds = 0.194 hours.
    evaluated_duration_seconds = total_samples * window_step_s
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0

    # False alarms are false positives.
    # We divide by evaluated hours to get false alarms per hour.
    false_alarms_per_hour = (
        fp / evaluated_duration_hours
        if evaluated_duration_hours > 0
        else 0.0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,

        # Confusion matrix values
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        # False alarm analysis
        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }

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
    criterion = nn.CrossEntropyLoss() # Two-class classification: class 0 = non-seizure, 1 = seizure

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
    MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlartifacts"
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("EEG-Seizure-Detection")

    with mlflow.start_run():
        mlflow.log_params({
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
            val_metrics = evaluate_model(model, val_loader, criterion, window_step_s=window_step_s)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            # Log both training and validation metrics to MLflow.
            # This allows comparison between learning performance and generalization.
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,

                # Validation metrics are logged every epoch.
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall_sensitivity": val_metrics["recall_sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "val_f1": val_metrics["f1"],
                "val_false_alarms_per_hour": val_metrics["false_alarms_per_hour"],
            }, step=epoch + 1)

            # AUC can be NaN if the split contains only one class.
            # MLflow should log it only when it is valid.
            if not np.isnan(val_metrics["auc"]):
                mlflow.log_metric("val_auc", val_metrics["auc"], step=epoch + 1)

            print(
                f"   Epoch {epoch + 1}/{epochs} "
                f"— train_loss: {train_loss:.4f} "
                f"train_acc: {train_acc:.3f} "
                f"val_loss: {val_loss:.4f} "
                f"val_f1: {val_metrics['f1']:.3f}"
            )
        # Final test evaluation.
        test_metrics = evaluate_model(model, test_loader, criterion, window_step_s=window_step_s)

        final_reported_accuracy = test_metrics["accuracy"]

        # Log final test metrics.
        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "final_test_accuracy": final_reported_accuracy,
            "test_precision": test_metrics["precision"],
            "test_recall_sensitivity": test_metrics["recall_sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_f1": test_metrics["f1"],

            # Confusion matrix values
            "test_tp": test_metrics["tp"],
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],

            # False alarm analysis
            "test_evaluated_duration_hours": test_metrics["evaluated_duration_hours"],
            "test_false_alarms_per_hour": test_metrics["false_alarms_per_hour"],
        })

        # AUC can be NaN if the test split has only one class.
        # We log it only when it is mathematically valid.
        if not np.isnan(test_metrics["auc"]):
            mlflow.log_metric("test_auc", test_metrics["auc"])

        print("\nFinal Test Results")
        print("------------------")
        print(f"Test loss                    : {test_metrics['loss']:.4f}")
        print(f"Test accuracy                : {test_metrics['accuracy']:.3f}")
        print(f"Final reported accuracy      : {final_reported_accuracy:.3f}")
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

        print("\nReported accuracy source     : held-out test set")


        mlflow.pytorch.log_model(model, name="seizure_cnn")
        print("✅ Model logged to MLflow!")
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