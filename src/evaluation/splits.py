"""Reusable train/validation/test split utilities.

Patient-level splitting uses a deterministic multi-start search to reduce
class-distribution drift between train, validation, and test splits while
keeping every patient in exactly one split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import random_split


@dataclass(frozen=True)
class PatientSplitSelection:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    score: float


def split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split a window-level dataset reproducibly into three non-empty subsets."""
    if len(dataset) < 3:
        raise ValueError(
            f"At least 3 windows are required for a three-way split, got {len(dataset)}."
        )

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )

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
    """Return patient counts while guaranteeing at least one patient per split."""
    if n_patients < 3:
        raise ValueError(
            f"Patient-level split requires at least 3 patients, got {n_patients}."
        )

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )

    # Small patient cohorts need more than one validation/test subject; a
    # single subject makes threshold selection and final metrics extremely
    # sensitive to that person's seizure burden.
    min_eval_patients = 2 if n_patients >= 6 else 1

    n_val = max(min_eval_patients, int(round(val_ratio * n_patients)))
    n_test = max(
        min_eval_patients,
        int(round((1.0 - train_ratio - val_ratio) * n_patients)),
    )
    n_train = n_patients - n_val - n_test

    if n_train < 1:
        raise RuntimeError(
            "Not enough patients to satisfy the requested evaluation split sizes: "
            f"patients={n_patients}, val={n_val}, test={n_test}."
        )

    if min(n_train, n_val, n_test) < 1:
        raise RuntimeError(
            "Failed to create non-empty patient-level splits: "
            f"train={n_train}, val={n_val}, test={n_test}."
        )

    return n_train, n_val, n_test


def _validate_patient_datasets(patient_datasets, patient_ids):
    reference_feature_shape = None
    class_count = 0

    for index, (features, labels) in enumerate(patient_datasets):
        features = np.asarray(features)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Patient {patient_ids[index]} has {features.shape[0]} windows but "
                f"{labels.shape[0]} labels."
            )
        if features.shape[0] == 0:
            raise ValueError(f"Patient {patient_ids[index]} contains zero windows.")
        if labels.min() < 0:
            raise ValueError(f"Patient {patient_ids[index]} contains negative labels.")

        class_count = max(class_count, int(labels.max()) + 1)

        if reference_feature_shape is None:
            reference_feature_shape = features.shape[1:]
        elif features.shape[1:] != reference_feature_shape:
            raise ValueError(
                "All patient files must use the same channel/time dimensions. "
                f"Expected {reference_feature_shape}, got {features.shape[1:]} "
                f"for patient {patient_ids[index]}."
            )

    return max(class_count, 3)


def _patient_class_counts(patient_datasets, n_classes):
    counts = np.zeros((len(patient_datasets), n_classes), dtype=np.int64)
    for index, (_, labels) in enumerate(patient_datasets):
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        counts[index] = np.bincount(labels, minlength=n_classes)[:n_classes]
    return counts


def _distribution(counts):
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def _candidate_score(
    patient_counts,
    train_idx,
    val_idx,
    test_idx,
    target_window_ratios,
):
    """Lower is better. Penalize class drift, missing classes, and size drift."""
    global_counts = patient_counts.sum(axis=0)
    global_distribution = _distribution(global_counts)
    present_globally = global_counts > 0
    total_windows = float(global_counts.sum())

    score = 0.0
    for split_idx, target_ratio in zip(
        (train_idx, val_idx, test_idx),
        target_window_ratios,
    ):
        split_counts = patient_counts[split_idx].sum(axis=0)
        split_distribution = _distribution(split_counts)

        # Main objective: keep each split's class proportions close to the full cohort.
        score += float(np.square(split_distribution - global_distribution).sum())

        # Keep window volume approximately aligned with target train/val/test ratios.
        split_window_ratio = split_counts.sum() / total_windows
        score += 1.5 * float((split_window_ratio - target_ratio) ** 2)

        # Missing a globally present class makes macro metrics unstable.
        missing = present_globally & (split_counts == 0)
        score += 100.0 * float(missing.sum())

        # Very small class counts are fragile even when technically non-zero.
        low_counts = np.maximum(0, 10 - split_counts[present_globally])
        score += 0.02 * float(low_counts.sum())

    return score


def select_balanced_patient_indices(
    patient_datasets,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    search_trials=10000,
    patient_ids=None,
):
    """Select a deterministic patient split with reduced class-distribution drift."""
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

    n_classes = _validate_patient_datasets(patient_datasets, patient_ids)
    patient_counts = _patient_class_counts(patient_datasets, n_classes=n_classes)

    # Match the actual patient-count allocation. For example, 10 subjects
    # use 6/2/2 rather than an unstable 7/1/2 split.
    target_window_ratios = (
        n_train / n_patients,
        n_val / n_patients,
        n_test / n_patients,
    )

    rng = np.random.default_rng(seed)
    best = None

    # At least one candidate is always evaluated.
    for _ in range(max(1, int(search_trials))):
        permutation = rng.permutation(n_patients)
        train_idx = np.sort(permutation[:n_train])
        val_idx = np.sort(permutation[n_train:n_train + n_val])
        test_idx = np.sort(permutation[n_train + n_val:])

        score = _candidate_score(
            patient_counts=patient_counts,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            target_window_ratios=target_window_ratios,
        )

        candidate = PatientSplitSelection(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            score=float(score),
        )

        if best is None or candidate.score < best.score - 1e-12:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to generate a patient-level split candidate.")

    return best


def split_dataset_by_patient(
    patient_datasets,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    patient_ids=None,
    search_trials=10000,
):
    """Split at patient level using deterministic class-balanced selection."""
    n_patients = len(patient_datasets)
    if patient_ids is None:
        patient_ids = [str(index) for index in range(n_patients)]

    selection = select_balanced_patient_indices(
        patient_datasets=patient_datasets,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        search_trials=search_trials,
        patient_ids=patient_ids,
    )

    train_idx = selection.train_indices
    val_idx = selection.val_indices
    test_idx = selection.test_indices

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
    print(f"  Split search trials      : {max(1, int(search_trials))}")
    print(f"  Split balance score      : {selection.score:.6f}")
    print(f"  Train patients ({len(train_idx)}) : {_names(train_idx)}")
    print(f"  Val patients   ({len(val_idx)}) : {_names(val_idx)}")
    print(f"  Test patients  ({len(test_idx)}) : {_names(test_idx)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
