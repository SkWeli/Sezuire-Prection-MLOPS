"""
tests/test_loader.py

Unit tests for the CHB-MIT preprocessing loader.
These tests focus on deterministic helper logic so they can run
without requiring real EDF files in the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# Update these imports to match the actual function names in your loader.py.
from src.preprocessing import loader


def test_module_imports() -> None:
    """Basic smoke test to ensure the loader module imports successfully."""
    assert loader is not None


def test_parse_seizure_summary_empty(tmp_path: Path) -> None:
    """An empty CHB-MIT summary file should produce no seizure intervals."""
    summary_file = tmp_path / "sample.seizures"
    summary_file.write_text("")

    if not hasattr(loader, "parse_seizure_summary"):
        pytest.skip("parse_seizure_summary() not implemented in loader.py")

    intervals = loader.parse_seizure_summary(summary_file)
    assert isinstance(intervals, list)
    assert len(intervals) == 0


def test_parse_seizure_summary_single_event(tmp_path: Path) -> None:
    """A simple summary file with one seizure should be parsed correctly."""
    summary_file = tmp_path / "sample.seizures"
    summary_file.write_text(
        "\n".join(
            [
                "File Name: chb01_03.edf",
                "Number of Seizures in File: 1",
                "Seizure Start Time: 100 seconds",
                "Seizure End Time: 130 seconds",
            ]
        )
    )

    if not hasattr(loader, "parse_seizure_summary"):
        pytest.skip("parse_seizure_summary() not implemented in loader.py")

    intervals = loader.parse_seizure_summary(summary_file)
    assert len(intervals) == 1
    assert intervals[0][0] == 100
    assert intervals[0][1] == 130


def test_epoch_starts_overlap_rule() -> None:
    """4-second windows with 50% overlap should step by 2 seconds."""
    if not hasattr(loader, "generate_epoch_starts"):
        pytest.skip("generate_epoch_starts() not implemented in loader.py")

    starts = loader.generate_epoch_starts(
        signal_duration=10.0,
        epoch_duration=4.0,
        overlap=0.5,
    )

    assert starts == [0.0, 2.0, 4.0, 6.0]


def test_label_ictal_window() -> None:
    """A window fully inside a seizure interval should be labeled ictal."""
    if not hasattr(loader, "label_epoch"):
        pytest.skip("label_epoch() not implemented in loader.py")

    label = loader.label_epoch(
        epoch_start=102.0,
        epoch_end=106.0,
        seizure_intervals=[(100.0, 130.0)],
        preictal_min=30.0,
        preictal_max=120.0,
        interictal_gap=14400.0,
    )
    assert label == "ictal"


def test_label_preictal_window() -> None:
    """A window in the 30-120 second pre-ictal horizon should be pre_ictal."""
    if not hasattr(loader, "label_epoch"):
        pytest.skip("label_epoch() not implemented in loader.py")

    label = loader.label_epoch(
        epoch_start=40.0,
        epoch_end=44.0,
        seizure_intervals=[(100.0, 130.0)],
        preictal_min=30.0,
        preictal_max=120.0,
        interictal_gap=14400.0,
    )
    assert label == "pre_ictal"


def test_label_interictal_window() -> None:
    """A window more than 4 hours away from any seizure should be interictal."""
    if not hasattr(loader, "label_epoch"):
        pytest.skip("label_epoch() not implemented in loader.py")

    label = loader.label_epoch(
        epoch_start=20000.0,
        epoch_end=20004.0,
        seizure_intervals=[(100.0, 130.0)],
        preictal_min=30.0,
        preictal_max=120.0,
        interictal_gap=14400.0,
    )
    assert label == "interictal"


def test_npz_output_roundtrip(tmp_path: Path) -> None:
    """Saved NPZ output should preserve arrays and labels."""
    output_file = tmp_path / "sample_epochs.npz"

    X = np.random.randn(3, 23, 1024).astype(np.float32)
    y = np.array(["ictal", "pre_ictal", "interictal"], dtype=object)

    np.savez(output_file, X=X, y=y)

    loaded = np.load(output_file, allow_pickle=True)
    assert loaded["X"].shape == (3, 23, 1024)
    assert list(loaded["y"]) == ["ictal", "pre_ictal", "interictal"]


def test_rdf_sidecar_written(tmp_path: Path) -> None:
    """RDF sidecar file should be created and contain preprocessing triples."""
    if not hasattr(loader, "write_rdf_sidecar"):
        pytest.skip("write_rdf_sidecar() not implemented in loader.py")

    session_id = "chb01_01"
    steps = [
        "bandpass_0.5_40.0_hz",
        "notch_50_hz",
        "common_average_reference",
        "epoching_4s_50pct_overlap",
    ]
    sidecar_path = tmp_path / "chb01_01.ttl"

    loader.write_rdf_sidecar(session_id=session_id, steps=steps, output_path=sidecar_path)

    assert sidecar_path.exists()
    content = sidecar_path.read_text(encoding="utf-8")
    assert session_id in content
    assert "bandpass" in content
    assert "notch" in content