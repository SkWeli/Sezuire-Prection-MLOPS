"""
Dataset split utilities.

This file keeps train/validation/test splitting logic reusable.
Both train.py and evaluate.py need the same split behavior so that
evaluation is consistent and reproducible.
"""

import torch
from torch.utils.data import random_split


def split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split the full EEG window dataset into train, validation, and test sets.

    Train set:
        Used to update model weights.

    Validation set:
        Used during training to monitor performance.

    Test set:
        Used only for final evaluation.

    A fixed seed is used so the split is the same across training and evaluation.
    """

    total_size = len(dataset)

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Test size is calculated from the remaining samples.
    # This avoids losing samples due to rounding.
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )