"""
Baseline seizure detection model with MLflow tracking.
Usage: python src/training/train.py --data data/processed/tusz/aaaaaajy/aaaaaajy.npz
"""
import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def train(data_path, epochs=5, lr=0.001, batch_size=32):
    data_path = Path(data_path)
    print(f"📂 Loading: {data_path}")

    data   = np.load(data_path)
    X      = torch.tensor(data["epochs"]).float().unsqueeze(1)  # (N,1,C,T)
    y      = torch.tensor(data["labels"]).long()

    print(f"   X: {X.shape} | y: {y.shape} | seizures: {y.sum().item()}")

    dataset    = TensorDataset(X, y)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model      = SeizureCNN()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    mlflow.set_experiment("EEG-Seizure-Detection")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "model": "SeizureCNN",
            "dataset": data_path.stem,
            "n_samples": len(X),
            "n_seizures": int(y.sum()),
        })

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct    = 0

            for xb, yb in loader:
                optimizer.zero_grad()
                out  = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct    += (out.argmax(1) == yb).sum().item()

            acc = correct / len(X)
            avg_loss = total_loss / len(loader)
            mlflow.log_metrics({"loss": avg_loss, "accuracy": acc}, step=epoch)
            print(f"   Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}  acc: {acc:.3f}")

        mlflow.pytorch.log_model(model, "seizure_cnn")
        print("✅ Model logged to MLflow!")
        print(f"✅ Run: mlflow ui → http://127.0.0.1:5000")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed/tusz/aaaaaajy/aaaaaajy.npz")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch-size", type=int,   default=32)
    args = parser.parse_args()
    sys.exit(train(args.data, args.epochs, args.lr, args.batch_size))