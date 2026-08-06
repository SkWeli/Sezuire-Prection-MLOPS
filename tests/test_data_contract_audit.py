"""Small synthetic tests for the P20 data-contract audit helpers."""

from pathlib import Path

import numpy as np

from src.evaluation.data_contract_audit import audit_patient


def _write_valid_npz(path: Path, patient_id: str) -> None:
    rng = np.random.default_rng(42)
    epochs = rng.normal(size=(9, 20, 512)).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    channels = np.array([f"CH{i:02d}" for i in range(20)])
    np.savez(
        path,
        epochs=epochs,
        labels=labels,
        sfreq=np.array(128.0),
        window_step_s=np.array(2.0),
        patient_id=np.array(patient_id),
        channels=channels,
    )


def test_valid_patient_contract_passes(tmp_path):
    patient_id = "patient_a"
    npz_path = tmp_path / f"{patient_id}.npz"
    ttl_path = tmp_path / f"{patient_id}.ttl"
    _write_valid_npz(npz_path, patient_id)
    ttl_path.write_text("@prefix ex: <http://example.org/> .", encoding="utf-8")

    result = audit_patient(
        patient_id=patient_id,
        split_name="train",
        npz_path=npz_path,
        ttl_path=ttl_path,
        expected_sfreq=128.0,
        expected_channels=20,
        expected_timepoints=512,
        expected_step_s=2.0,
        finite_mode="full",
        sample_windows=4,
    )

    assert result.status == "PASS"
    assert result.windows == 9
    assert (result.interictal, result.preictal, result.ictal) == (4, 2, 3)
    assert result.channel_order_available is True


def test_invalid_label_and_shape_fail(tmp_path):
    patient_id = "patient_bad"
    npz_path = tmp_path / f"{patient_id}.npz"
    epochs = np.zeros((3, 19, 500), dtype=np.float32)
    labels = np.array([0, 1, 9], dtype=np.int64)
    np.savez(npz_path, epochs=epochs, labels=labels, sfreq=np.array(100.0))

    result = audit_patient(
        patient_id=patient_id,
        split_name="validation",
        npz_path=npz_path,
        ttl_path=None,
        expected_sfreq=128.0,
        expected_channels=20,
        expected_timepoints=512,
        expected_step_s=2.0,
        finite_mode="sample",
        sample_windows=3,
    )

    assert result.status == "FAIL"
    messages = " ".join(item.message for item in result.findings)
    assert "Invalid label IDs" in messages
    assert "Channel count" in messages
    assert "Timepoints" in messages
    assert "Sampling rate" in messages


def test_missing_npz_fails():
    result = audit_patient(
        patient_id="missing",
        split_name="test",
        npz_path=None,
        ttl_path=None,
        expected_sfreq=128.0,
        expected_channels=20,
        expected_timepoints=512,
        expected_step_s=2.0,
        finite_mode="sample",
        sample_windows=3,
    )

    assert result.status == "FAIL"
    assert any("NPZ file not found" in item.message for item in result.findings)
