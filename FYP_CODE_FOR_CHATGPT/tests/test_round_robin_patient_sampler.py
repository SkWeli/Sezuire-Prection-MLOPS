"""Tests for RoundRobinPatientBatchSampler."""

from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from src.data.lazy_eeg_dataset import LazyEEGDataset
from src.data.round_robin_patient_sampler import (
    RoundRobinPatientBatchSampler,
)


def _write_patient(
    root: Path,
    patient_id: str,
    windows: int,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)
    patient_dir = root / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    path = patient_dir / f"{patient_id}.npz"
    np.savez_compressed(
        path,
        epochs=rng.normal(
            size=(windows, 20, 512)
        ).astype(np.float32),
        labels=np.arange(windows, dtype=np.int64) % 3,
        sfreq=np.array(128.0, dtype=np.float32),
    )
    return path


@pytest.fixture()
def dataset(tmp_path: Path) -> LazyEEGDataset:
    files = [
        _write_patient(tmp_path, "patient_a", 10, 1),
        _write_patient(tmp_path, "patient_b", 9, 2),
        _write_patient(tmp_path, "patient_c", 7, 3),
    ]
    return LazyEEGDataset(
        files,
        cache_size=2,
        return_patient_id=True,
    )


def test_covers_every_window_once(dataset):
    sampler = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )
    observed = [index for batch in sampler for index in batch]
    assert sorted(observed) == list(range(len(dataset)))
    assert len(observed) == len(set(observed))


def test_each_batch_contains_one_patient(dataset):
    sampler = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )
    for batch in sampler:
        patient_ids = {dataset[index][2] for index in batch}
        assert len(patient_ids) == 1


def test_round_robin_switches_before_large_patient_finishes(dataset):
    sampler = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=2,
        batches_per_patient_block=1,
        shuffle_patients=False,
        shuffle_within_patient=False,
        seed=42,
    )

    batches = list(sampler)
    first_patient_ids = [
        dataset[batch[0]][2]
        for batch in batches[:3]
    ]

    assert first_patient_ids == [
        "patient_a",
        "patient_b",
        "patient_c",
    ]


def test_length_matches_emitted_batches(dataset):
    sampler = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )
    assert len(sampler) == len(list(sampler))


def test_deterministic_for_same_epoch(dataset):
    first = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )
    second = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )

    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)


def test_epoch_changes_order(dataset):
    sampler = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )

    sampler.set_epoch(0)
    first = list(sampler)

    sampler.set_epoch(1)
    second = list(sampler)

    assert first != second


def test_dataloader_contract(dataset):
    sampler = RoundRobinPatientBatchSampler(
        dataset,
        batch_size=3,
        batches_per_patient_block=2,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
    )

    windows, labels, patient_ids = next(iter(loader))

    assert windows.shape == (3, 1, 20, 512)
    assert labels.shape == (3,)
    assert len(set(patient_ids)) == 1


def test_invalid_block_size(dataset):
    with pytest.raises(ValueError):
        RoundRobinPatientBatchSampler(
            dataset,
            batch_size=3,
            batches_per_patient_block=0,
        )
