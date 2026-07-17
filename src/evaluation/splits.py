"""Reusable train/validation/test split utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import random_split


def split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split a window-level dataset reproducibly into train/validation/test subsets."""
    if len(dataset) < 3:
        raise ValueError(
            f"At least 3 windows are required for a three-way split, got {len(dataset)}."
        )

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    total_size = len(dataset)
    train_size = min(max(1, int(train_ratio * total_size)), total_size - 2)
    val_size = min(max(1, int(val_ratio * total_size)), total_size - train_size - 1)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )


def _patient_split_counts(n_patients, train_ratio, val_ratio):
    """Return split counts while guaranteeing at least one patient per split."""
    if n_patients < 3:
        raise ValueError(
            f"Patient-level split requires at least 3 patients, got {n_patients}."
        )

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    n_train = min(max(1, int(train_ratio * n_patients)), n_patients - 2)
    remaining_after_train = n_patients - n_train
    n_val = min(max(1, int(val_ratio * n_patients)), remaining_after_train - 1)
    n_test = n_patients - n_train - n_val

    if min(n_train, n_val, n_test) < 1:
        raise RuntimeError(
            "Failed to create non-empty patient-level train/validation/test splits: "
            f"train={n_train}, val={n_val}, test={n_test}."
        )

    return n_train, n_val, n_test


def split_dataset_by_patient(
    patient_datasets,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    patient_ids=None,
):
    """
    Split EEG arrays at patient level, preventing a patient's windows from
    appearing in more than one split.

    patient_datasets is a list of (X_patient, y_patient) tuples.
    """
    n_patients = len(patient_datasets)
    n_train, n_val, n_test = _patient_split_counts(
        n_patients=n_patients,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    if patient_ids is None:
        patient_ids = [str(index) for index in range(n_patients)]

    if len(patient_ids) != n_patients:
        raise ValueError("patient_ids length must match patient_datasets length.")

    reference_feature_shape = None
    for index, (features, labels) in enumerate(patient_datasets):
        features = np.asarray(features)
        labels = np.asarray(labels)

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Patient {patient_ids[index]} has {features.shape[0]} windows but "
                f"{labels.shape[0]} labels."
            )
        if features.shape[0] == 0:
            raise ValueError(f"Patient {patient_ids[index]} contains zero windows.")

        if reference_feature_shape is None:
            reference_feature_shape = features.shape[1:]
        elif features.shape[1:] != reference_feature_shape:
            raise ValueError(
                "All patient files must use the same channel/time dimensions. "
                f"Expected {reference_feature_shape}, got {features.shape[1:]} "
                f"for patient {patient_ids[index]}."
            )

    rng = np.random.default_rng(seed)
    patient_indices = np.arange(n_patients)
    rng.shuffle(patient_indices)

    train_idx = patient_indices[:n_train]
    val_idx = patient_indices[n_train:n_train + n_val]
    test_idx = patient_indices[n_train + n_val:]

    def _concat(indices):
        features = [np.asarray(patient_datasets[index][0]) for index in indices]
        labels = [np.asarray(patient_datasets[index][1]) for index in indices]
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    X_train, y_train = _concat(train_idx)
    X_val, y_val = _concat(val_idx)
    X_test, y_test = _concat(test_idx)

    def _names(indices):
        return [patient_ids[index] for index in indices]

    print(f"  Patient-level split      : {n_patients} patients total")
    print(f"  Train patients ({n_train}) : {_names(train_idx)}")
    print(f"  Val patients   ({n_val}) : {_names(val_idx)}")
    print(f"  Test patients  ({n_test}) : {_names(test_idx)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
