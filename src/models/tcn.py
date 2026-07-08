"""
Temporal Convolutional Network (TCN) for EEG seizure detection.

This file contains only model definitions.
Training logic stays in src/training/train.py.

Why TCN:
- Uses dilated causal convolutions to capture long-range temporal
  dependencies in EEG signals without the vanishing-gradient issues of RNNs.
- Lightweight compared to LSTMs/Transformers, making it suitable for
  edge deployment (ONNX export, quantization) later in the pipeline.
- Same input/output interface as SeizureCNN, so it works directly with
  evaluate_model(), find_best_threshold(), and MLflow logging.
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Removes extra right-side padding introduced by causal convolution,
    so the output length matches the input length exactly.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    One residual block of the TCN: two dilated causal convolutions
    with weight normalization, ReLU, dropout, and a residual connection.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2
        )

        # 1x1 conv to match channel dimensions for the residual connection.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SeizureTCN(nn.Module):
    """
    TCN for binary EEG seizure detection.

    Input shape:
        (batch_size, 1, n_channels, n_timepoints)
    Example:
        (32, 1, 20, 512)

    Output shape:
        (batch_size, 2)

    Output classes:
        class 0 = non-seizure
        class 1 = seizure

    Design note:
    EEG channels are flattened into the input feature dimension of a 1D
    TCN (n_channels acts like "input channels" for Conv1d), and the
    network processes the time axis with dilated convolutions.
    """

    def __init__(self, n_channels=20, n_timepoints=512, n_classes=2,
                 num_channels=(32, 32, 64), kernel_size=7, dropout=0.2):
        super().__init__()

        layers = []
        in_channels = n_channels
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size,
                              dilation=dilation, dropout=dropout)
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        # Global average pooling over the time axis collapses variable-length
        # sequences into a fixed-size vector for classification.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        """
        x: EEG batch with shape (batch_size, 1, channels, timepoints)
        """
        # Drop the singleton dimension so channels become the Conv1d
        # "in_channels" axis: (batch, channels, timepoints).
        x = x.squeeze(1)

        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)

        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)