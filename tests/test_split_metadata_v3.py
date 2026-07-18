import numpy as np

from src.evaluation.splits import split_dataset_by_patient


def _patient(marker):
    labels = np.array([0] * 20 + [1] * 5 + [2] * 5, dtype=np.int64)
    features = np.full((len(labels), 2, 8), marker, dtype=np.float32)
    return features, labels


def test_patient_split_can_return_reproducibility_metadata():
    patients = [_patient(index) for index in range(10)]
    patient_ids = [f"p{index}" for index in range(10)]

    datasets, selection = split_dataset_by_patient(
        patients,
        patient_ids=patient_ids,
        seed=42,
        search_trials=100,
        return_selection=True,
    )

    assert len(datasets) == 3
    assigned = np.concatenate(
        [selection.train_indices, selection.val_indices, selection.test_indices]
    )
    assert sorted(assigned.tolist()) == list(range(10))
    assert np.isfinite(selection.score)
