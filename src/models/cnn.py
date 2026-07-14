"""
CNN model architecture for EEG seizure detection.

This file contains only model definitions.
Training logic is kept separately in src/training/train.py.

Why this separation is useful:
- train.py becomes cleaner.
- model architecture can be reused for evaluation, ONNX export, and quantization.
- future models such as EEGNet or TCN can be added in this same models folder.
"""

import torch
import torch.nn as nn


class SeizureCNN(nn.Module):
    """
    Lightweight CNN for binary EEG seizure detection.

    Input shape:
        (batch_size, 1, n_channels, n_timepoints)

    Example:
        (32, 1, 20, 512)

    Output shape:
        (batch_size, 2)

    Output classes:
        class 0 = non-seizure
        class 1 = seizure

    Why this model is used:
        This is the baseline CNN model. It is simple, lightweight,
        and suitable for later edge deployment experiments.
    """

    def __init__(self, n_channels=20, n_timepoints=512, n_classes=3):
        super().__init__()

        # First convolution learns temporal patterns inside each EEG channel.
        # Kernel size (1, 25) means it looks across time but not across channels yet.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(1, 25),
            padding=(0, 12)
        )
        self.bn1 = nn.BatchNorm2d(16)

        # Second convolution learns spatial relationships across EEG channels.
        # Kernel size (n_channels, 1) combines information from all channels.
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(n_channels, 1)
        )
        self.bn2 = nn.BatchNorm2d(32)

        # Adaptive pooling reduces the feature map to a fixed size.
        # This keeps the fully connected layer size stable.
        self.pool = nn.AdaptiveAvgPool2d((1, 32))

        # Fully connected layers perform final classification.
        self.fc1 = nn.Linear(32 * 32, 64)
        self.fc2 = nn.Linear(64, n_classes)

        # Dropout helps reduce overfitting during training.
        self.drop = nn.Dropout(0.5)

        # ReLU adds non-linearity after convolution and dense layers.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the CNN.

        x:
            EEG batch with shape (batch_size, 1, channels, timepoints)
        """

        # Learn temporal features.
        x = self.relu(self.bn1(self.conv1(x)))

        # Learn spatial/channel features.
        x = self.relu(self.bn2(self.conv2(x)))

        # Reduce feature map size before classification.
        x = self.pool(x)

        # Flatten features for the fully connected layers.
        x = x.view(x.size(0), -1)

        # Classifier head.
        x = self.drop(self.relu(self.fc1(x)))

        return self.fc2(x)