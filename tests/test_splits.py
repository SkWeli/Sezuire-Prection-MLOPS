import numpy as np

from src.evaluation.splits import (
    select_balanced_patient_indices,
    split_dataset_by_patient,
)


def _patient(n0, n1, n2, patient_marker):
    labels = np.array([0] * n0 + [1] * n1 + [2] * n2, dtype=np.int64)
    features = np.full((len(labels), 2, 8), patient_marker, dtype=np.float32)
    return features, labels


def test_balanced_patient_split_is_reproducible_and_non_overlapping():
    patients = [
        _patient(80, 10, 10, 0),
        _patient(70, 20, 10, 1),
        _patient(60, 20, 20, 2),
        _patient(90, 5, 5, 3),
        _patient(65, 15, 20, 4),
        _patient(75, 15, 10, 5),
        _patient(55, 25, 20, 6),
        _patient(85, 10, 5, 7),
        _patient(60, 30, 10, 8),
        _patient(70, 10, 20, 9),
    ]
    patient_ids = [f"p{i}" for i in range(len(patients))]

    first = select_balanced_patient_indices(
        patients, patient_ids=patient_ids, seed=42, search_trials=2000
    )
    second = select_balanced_patient_indices(
        patients, patient_ids=patient_ids, seed=42, search_trials=2000
    )

    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.val_indices, second.val_indices)
    assert np.array_equal(first.test_indices, second.test_indices)

    assigned = np.concatenate(
        [first.train_indices, first.val_indices, first.test_indices]
    )
    assert sorted(assigned.tolist()) == list(range(len(patients)))
    assert len(np.unique(assigned)) == len(patients)


def test_balanced_patient_split_preserves_all_classes_when_possible():
    patients = [
        _patient(80, 10, 10, 0),
        _patient(70, 20, 10, 1),
        _patient(60, 20, 20, 2),
        _patient(90, 5, 5, 3),
        _patient(65, 15, 20, 4),
        _patient(75, 15, 10, 5),
        _patient(55, 25, 20, 6),
        _patient(85, 10, 5, 7),
        _patient(60, 30, 10, 8),
        _patient(70, 10, 20, 9),
    ]

    (_, y_train), (_, y_val), (_, y_test) = split_dataset_by_patient(
        patients,
        patient_ids=[f"p{i}" for i in range(len(patients))],
        seed=42,
        search_trials=2000,
    )

    assert set(np.unique(y_train)) == {0, 1, 2}
    assert set(np.unique(y_val)) == {0, 1, 2}
    assert set(np.unique(y_test)) == {0, 1, 2}
