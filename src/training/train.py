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


def evaluate_model(model, loader, criterion):
    """
    Evaluate the model on validation or test data.

    Important:
    - This function does not train the model.
    - It does not update weights.
    - It only calculates average loss and accuracy.

    This function is used for:
    - validation after each epoch
    - final test evaluation after training is complete
    """

    # Set model to evaluation mode.
    # This disables training-specific behavior such as dropout.
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    # torch.no_grad() saves memory and speeds up evaluation.
    # We do not need gradients because we are not doing backpropagation here.
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)

            batch_size = xb.size(0)

            # Multiply loss by batch size because batches may not all be equal.
            # The final batch can be smaller than the others.
            total_loss += loss.item() * batch_size

            # outputs.argmax(1) gives the predicted class:
            # 0 = non-seizure, 1 = seizure
            correct += (outputs.argmax(1) == yb).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy

def train(data_path=None, epochs=20, lr=0.001, batch_size=32, max_patients=None):
    import time
    from pathlib import Path

    start_time = time.perf_counter()

    if max_patients is not None:
        processed_dir = Path("data/processed/tusz")
        npz_files = sorted(processed_dir.rglob("*.npz"))[:max_patients]
        print(f"🚀 Loading {len(npz_files)} patients (max {max_patients})")
        all_X, all_y = [], []
        for npz_file in npz_files:
            print(f"   📂 {npz_file.name}")
            data = np.load(npz_file)
            all_X.append(data["epochs"])
            all_y.append(data["labels"])
            print(f"     {len(data['labels'])} windows, {data['n_seizures']} seizures")

        data_path_str = "tusz-multi"
        X = torch.tensor(np.concatenate(all_X)).float().unsqueeze(1)  # (N,1,C,T)
        y = torch.tensor(np.concatenate(all_y)).long()
    else:
        data_path = Path(data_path)   # ← convert early
        print(f"📂 Loading: {data_path}")
        data = np.load(data_path)
        X = torch.tensor(data["epochs"]).float().unsqueeze(1)
        y = torch.tensor(data["labels"]).long()
        data_path_str = data_path.stem

    print(f"  X: {X.shape} | y: {y.shape} | seizures: {y.sum().item()}")

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
            # Validation data is not used to update weights.
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)

            # Log both training and validation metrics to MLflow.
            # This allows comparison between learning performance and generalization.
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch + 1)

            print(
                f"   Epoch {epoch + 1}/{epochs} "
                f"— train_loss: {train_loss:.4f} "
                f"train_acc: {train_acc:.3f} "
                f"val_loss: {val_loss:.4f} "
                f"val_acc: {val_acc:.3f}"
            )
        # Final test evaluation.
        # This is the most important accuracy value because the model has not trained on test data.
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)

        # These values should be used when reporting final model performance.
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        })

        print("\nFinal Test Results")
        print("------------------")
        print(f"Test loss    : {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.3f}")


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
    args = parser.parse_args()
    sys.exit(train(args.data, args.epochs, args.lr, args.batch_size, args.max_patients))