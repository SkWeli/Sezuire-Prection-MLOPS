"""
Tests for the patient-level lazy EEG dataset.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.lazy_eeg_dataset import LazyEEGDataset


def _write_patient(
    directory: Path,
    patient_id: str,
    *,
    n_windows: int,
    seed: int,
) -> Path:
    """
    Create a small artificial patient NPZ with the real project contract.
    """

    rng = np.random.default_rng(seed)

    patient_dir = directory / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    epochs = rng.normal(
        size=(n_windows, 20, 512),
    ).astype(np.float32)

    labels = np.arange(n_windows, dtype=np.int64) % 3

    npz_path = patient_dir / f"{patient_id}.npz"

    np.savez_compressed(
        npz_path,
        epochs=epochs,
        labels=labels,
        sfreq=np.array(128.0, dtype=np.float32),
    )

    return npz_path


@pytest.fixture()
def patient_files(tmp_path: Path) -> list[Path]:
    return [
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
    ]


def test_dataset_length(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
    )

    assert len(dataset) == 12


def test_patient_ids(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
    )

    assert dataset.get_patient_ids() == [
        "patient_a",
        "patient_b",
    ]


def test_labels_loaded_without_eeg_iteration(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
    )

    labels = dataset.get_all_labels()

    assert labels.shape == (12,)
    assert set(np.unique(labels)) == {0, 1, 2}
    assert dataset.cached_patient_ids == []


def test_output_contract(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
    )

    window, label = dataset[0]

    assert isinstance(window, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert window.shape == (1, 20, 512)
    assert window.dtype == torch.float32
    assert label.dtype == torch.long
    assert int(label) in {0, 1, 2}
    assert torch.isfinite(window).all()


def test_normalization_matches_existing_pipeline(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
        normalize=True,
    )

    window, _ = dataset[0]

    channel_means = window.mean(dim=-1)
    channel_stds = window.std(dim=-1)

    assert torch.allclose(
        channel_means,
        torch.zeros_like(channel_means),
        atol=1e-5,
    )

    assert torch.allclose(
        channel_stds,
        torch.ones_like(channel_stds),
        atol=1e-4,
    )


def test_lazy_result_matches_manual_eager_normalization(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
        normalize=True,
    )

    lazy_window, lazy_label = dataset[3]

    with np.load(patient_files[0], allow_pickle=False) as data:
        eager_window = torch.tensor(
            data["epochs"][3],
            dtype=torch.float32,
        )
        eager_label = int(data["labels"][3])

    mean = eager_window.mean(dim=-1, keepdim=True)
    std = eager_window.std(dim=-1, keepdim=True)
    eager_window = (eager_window - mean) / (std + 1e-6)
    eager_window = eager_window.unsqueeze(0)

    assert torch.equal(
        lazy_label,
        torch.tensor(eager_label, dtype=torch.long),
    )

    assert torch.allclose(
        lazy_window,
        eager_window,
        atol=1e-6,
    )


def test_global_index_crosses_patient_boundary(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
        return_patient_id=True,
    )

    _, _, first_patient = dataset[4]
    _, _, second_patient = dataset[5]

    assert first_patient == "patient_a"
    assert second_patient == "patient_b"


def test_lru_cache_keeps_only_requested_number_of_patients(
    patient_files,
):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
        return_patient_id=True,
    )

    dataset[0]
    assert dataset.cached_patient_ids == ["patient_a"]

    dataset[5]
    assert dataset.cached_patient_ids == ["patient_b"]


def test_repeated_access_is_deterministic(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
    )

    first_window, first_label = dataset[2]
    second_window, second_label = dataset[2]

    assert torch.equal(first_window, second_window)
    assert torch.equal(first_label, second_label)


def test_dataloader_num_workers_zero(patient_files):
    dataset = LazyEEGDataset(
        patient_files,
        cache_size=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    windows, labels = next(iter(loader))

    assert windows.shape == (4, 1, 20, 512)
    assert labels.shape == (4,)
    assert windows.dtype == torch.float32
    assert labels.dtype == torch.long


def test_invalid_label_is_rejected(tmp_path: Path):
    patient_dir = tmp_path / "invalid_patient"
    patient_dir.mkdir()

    npz_path = patient_dir / "invalid_patient.npz"

    np.savez_compressed(
        npz_path,
        epochs=np.zeros((3, 20, 512), dtype=np.float32),
        labels=np.array([0, 1, 9], dtype=np.int64),
        sfreq=np.array(128.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="invalid labels"):
        LazyEEGDataset([npz_path])


def test_wrong_sampling_rate_is_rejected(tmp_path: Path):
    patient_dir = tmp_path / "wrong_sfreq"
    patient_dir.mkdir()

    npz_path = patient_dir / "wrong_sfreq.npz"

    np.savez_compressed(
        npz_path,
        epochs=np.zeros((3, 20, 512), dtype=np.float32),
        labels=np.array([0, 1, 2], dtype=np.int64),
        sfreq=np.array(256.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="expected sfreq"):
        LazyEEGDataset([npz_path])