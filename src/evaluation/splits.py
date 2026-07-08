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

import numpy as np


def split_dataset_by_patient(patient_datasets, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split EEG data into train/validation/test sets at the PATIENT level,
    not the window level.

    Why this is needed:
        Standard random_split() shuffles all windows from all patients
        together before splitting. Because EEG windows overlap (e.g. 50%),
        and because random_split() has no concept of "which patient this
        window came from", the same patient's data can end up in both the
        training and test sets. This lets the model partially memorize
        patient-specific noise/amplitude characteristics rather than
        learning genuine, generalizable seizure patterns, and produces
        test metrics that look better (or worse) than the model would
        actually achieve on a truly unseen patient.

        Splitting by patient first guarantees that no patient's windows
        appear in more than one split, giving an honest estimate of how
        well the model generalizes to new patients.

    patient_datasets:
        A list of (X_patient, y_patient) tuples, one per patient/npz file.
        X_patient shape: (n_windows, 1, channels, timepoints)
        y_patient shape: (n_windows,)

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    n_patients = len(patient_datasets)

    if n_patients < 3:
        # With fewer than 3 patients, patient-level splitting isn't
        # meaningful (can't hold out separate patients for val AND test).
        # Caller should fall back to window-level split in this case.
        raise ValueError(
            f"Patient-level split requires at least 3 patients, got {n_patients}. "
            "Use split_dataset() (window-level) for single-patient runs instead."
        )

    rng = np.random.default_rng(seed)
    patient_indices = np.arange(n_patients)
    rng.shuffle(patient_indices)

    n_train = max(1, int(train_ratio * n_patients))
    n_val = max(1, int(val_ratio * n_patients))
    # Remaining patients go to test; guarantees every patient is used exactly once.
    n_val = min(n_val, n_patients - n_train - 1) if n_patients - n_train > 1 else n_val
    n_test = n_patients - n_train - n_val

    train_idx = patient_indices[:n_train]
    val_idx = patient_indices[n_train:n_train + n_val]
    test_idx = patient_indices[n_train + n_val:]

    def _concat(indices):
        Xs = [patient_datasets[i][0] for i in indices]
        ys = [patient_datasets[i][1] for i in indices]
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _concat(train_idx)
    X_val, y_val = _concat(val_idx)
    X_test, y_test = _concat(test_idx)

    print(f"  Patient-level split      : {n_patients} patients total")
    print(f"  Train patients ({len(train_idx)}) : {sorted(train_idx.tolist())}")
    print(f"  Val patients   ({len(val_idx)}) : {sorted(val_idx.tolist())}")
    print(f"  Test patients  ({len(test_idx)}) : {sorted(test_idx.tolist())}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)