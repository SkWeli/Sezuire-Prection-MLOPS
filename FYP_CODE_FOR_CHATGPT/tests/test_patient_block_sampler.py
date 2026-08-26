"""
Tests for the patient-block batch sampler.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.lazy_eeg_dataset import LazyEEGDataset
from src.data.patient_block_sampler import (
    PatientBlockBatchSampler,
)


def _write_patient(
    directory: Path,
    patient_id: str,
    n_windows: int,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)

    patient_dir = directory / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    epochs = rng.normal(
        size=(n_windows, 20, 512),
    ).astype(np.float32)

    labels = (
        np.arange(n_windows, dtype=np.int64) % 3
    )

    path = patient_dir / f"{patient_id}.npz"

    np.savez_compressed(
        path,
        epochs=epochs,
        labels=labels,
        sfreq=np.array(128.0, dtype=np.float32),
    )

    return path


@pytest.fixture()
def dataset(tmp_path: Path) -> LazyEEGDataset:
    files = [
        _write_patient(
            tmp_path,
            "patient_a",
            n_windows=5,
            seed=1,
        ),
        _write_patient(
            tmp_path,
            "patient_b",
            n_windows=7,
            seed=2,
        ),
        _write_patient(
            tmp_path,
            "patient_c",
            n_windows=4,
            seed=3,
        ),
    ]

    return LazyEEGDataset(
        files,
        cache_size=1,
        return_patient_id=True,
    )


def test_sampler_covers_every_sample_once(dataset):
    sampler = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        seed=42,
    )

    observed = [
        index
        for batch in sampler
        for index in batch
    ]

    assert sorted(observed) == list(range(len(dataset)))
    assert len(observed) == len(set(observed))


def test_each_batch_contains_one_patient(dataset):
    sampler = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        seed=42,
    )

    for batch in sampler:
        patient_ids = {
            dataset[index][2]
            for index in batch
        }

        assert len(patient_ids) == 1


def test_batch_count(dataset):
    sampler = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        drop_last=False,
        seed=42,
    )

    # patient_a: ceil(5/3) = 2
    # patient_b: ceil(7/3) = 3
    # patient_c: ceil(4/3) = 2
    assert len(sampler) == 7


def test_drop_last(dataset):
    sampler = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        drop_last=True,
        seed=42,
    )

    batches = list(sampler)

    assert len(batches) == 4
    assert all(len(batch) == 3 for batch in batches)


def test_same_seed_and_epoch_are_deterministic(dataset):
    first = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        seed=42,
    )
    second = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        seed=42,
    )

    first.set_epoch(2)
    second.set_epoch(2)

    assert list(first) == list(second)


def test_epoch_changes_order(dataset):
    sampler = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
        seed=42,
    )

    sampler.set_epoch(0)
    epoch_zero = list(sampler)

    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one


def test_dataloader_output_contract(dataset):
    sampler = PatientBlockBatchSampler(
        dataset,
        batch_size=3,
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
    assert windows.dtype == torch.float32
    assert labels.dtype == torch.long
    assert len(set(patient_ids)) == 1


def test_invalid_dataset_type():
    with pytest.raises(TypeError):
        PatientBlockBatchSampler(
            dataset=[1, 2, 3],
            batch_size=2,
        )