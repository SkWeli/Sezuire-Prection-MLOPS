"""
Shared utilities for the offline viva demonstrations.

This module centralizes repeatable checks so the PyTorch and ONNX
demonstrations use exactly the same:

- artifact verification
- SHACL validation
- EEG loading
- normalization
- reporting logic
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from offline_demo.config import (
    EXPECTED_CHANNELS,
    EXPECTED_TIMEPOINTS,
    ONTOLOGY_PATH,
    REPORT_DIRECTORY,
    SHACL_SHAPES_PATH,
    SHACL_VALIDATOR_PATH,
)


def print_header(title: str) -> None:
    """
    Print a consistent section heading.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_section(title: str) -> None:
    """
    Print a numbered or named subsection heading.
    """
    print("\n" + title)
    print("-" * 80)


def sha256_file(path: Path) -> str:
    """
    Calculate the SHA256 digest of a file.

    The file is read in blocks so the same function remains safe for
    larger artifacts.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_artifact(
    path: Path,
    expected_sha256: str,
    description: str,
) -> str:
    """
    Verify that an artifact exists and matches the expected SHA256.

    This prevents the viva demonstration from accidentally loading:

    - an overwritten checkpoint;
    - an older ONNX export;
    - a corrupted model artifact.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    actual_sha256 = sha256_file(path)

    print(f"{description} path   : {path}")
    print(f"{description} SHA256 : {actual_sha256}")

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{description} SHA256 verification failed."
        )

    print(f"{description} verification: PASS")

    return actual_sha256


def run_shacl_validation(
    ttl_path: Path,
) -> tuple[bool, str, float]:
    """
    Run the existing SHACL validator as a separate Python process.

    Running the existing project validator preserves the same semantic
    validation behaviour used by the training pipeline.

    Returns:
        conforms:
            True when SHACL validation passes.

        combined_output:
            Complete validator terminal output.

        elapsed_ms:
            Total validation duration measured by this demonstration.
    """
    if not ttl_path.exists():
        raise FileNotFoundError(
            f"TTL metadata file was not found: {ttl_path}"
        )

    required_files = [
        SHACL_VALIDATOR_PATH,
        SHACL_SHAPES_PATH,
        ONTOLOGY_PATH,
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required semantic file was not found: {required_file}"
            )

    import time

    start = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            str(SHACL_VALIDATOR_PATH),
            str(ttl_path),
        ],
        cwd=str(SHACL_VALIDATOR_PATH.parents[2]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    combined_output = "\n".join(
        part
        for part in [
            result.stdout.strip(),
            result.stderr.strip(),
        ]
        if part
    )

    if combined_output:
        print(combined_output)

    conforms = result.returncode == 0

    print(
        "SHACL result : "
        f"{'PASS' if conforms else 'FAIL'}"
    )
    print(
        f"Validation time: {elapsed_ms:.2f} ms"
    )

    return (
        conforms,
        combined_output,
        elapsed_ms,
    )


def load_patient_npz(
    npz_path: Path,
) -> dict[str, Any]:
    """
    Load and verify the trusted processed patient NPZ file.

    allow_pickle=True is required because ch_names was stored as a
    NumPy object array. This is used only for the locally generated,
    trusted project dataset.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Patient NPZ was not found: {npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        required_keys = {
            "epochs",
            "labels",
            "ch_names",
            "sfreq",
            "patient_id",
        }

        missing_keys = sorted(
            required_keys - set(data.files)
        )

        if missing_keys:
            raise RuntimeError(
                f"Patient NPZ is missing fields: {missing_keys}"
            )

        epochs = np.asarray(
            data["epochs"],
            dtype=np.float32,
        )

        labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        )

        channel_names = [
            str(channel)
            for channel in data["ch_names"].tolist()
        ]

        sampling_rate = float(
            np.asarray(data["sfreq"]).item()
        )

        patient_id = str(
            np.asarray(data["patient_id"]).item()
        )

    if epochs.ndim != 3:
        raise RuntimeError(
            "Expected EEG epochs with shape [N, channels, samples], "
            f"received {epochs.shape}."
        )

    if tuple(epochs.shape[1:]) != (
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
    ):
        raise RuntimeError(
            "Unexpected EEG epoch shape. "
            f"Expected [N, {EXPECTED_CHANNELS}, "
            f"{EXPECTED_TIMEPOINTS}], received {epochs.shape}."
        )

    if len(labels) != len(epochs):
        raise RuntimeError(
            "The number of labels does not match the number of epochs."
        )

    if len(channel_names) != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Unexpected number of channel names."
        )

    return {
        "epochs": epochs,
        "labels": labels,
        "ch_names": channel_names,
        "sfreq": sampling_rate,
        "patient_id": patient_id,
    }


def normalize_windows(
    windows: np.ndarray,
) -> np.ndarray:
    """
    Apply per-window, per-channel z-score normalization.

    Each EEG channel is normalized independently across its 512 samples.

    Output shape:
        [batch, 1, 20, 512]
    """
    if windows.ndim != 3:
        raise ValueError(
            "Expected windows with shape [batch, channels, samples]."
        )

    means = windows.mean(
        axis=-1,
        keepdims=True,
    )

    standard_deviations = windows.std(
        axis=-1,
        keepdims=True,
    )

    standard_deviations = np.where(
        standard_deviations < 1e-8,
        1.0,
        standard_deviations,
    )

    normalized = (
        windows - means
    ) / standard_deviations

    normalized = normalized.astype(
        np.float32,
        copy=False,
    )

    normalized = normalized[:, np.newaxis, :, :]

    if not np.isfinite(normalized).all():
        raise RuntimeError(
            "Normalized EEG contains NaN or infinity."
        )

    return normalized


def softmax(
    logits: np.ndarray,
) -> np.ndarray:
    """
    Compute numerically stable softmax probabilities.
    """
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(
        shifted_logits
    )

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )


def save_json_report(
    report_name: str,
    contents: dict[str, Any],
) -> Path:
    """
    Save timestamped machine-readable demonstration evidence.
    """
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        REPORT_DIRECTORY
        / f"{report_name}_{timestamp}.json"
    )

    report_path.write_text(
        json.dumps(
            contents,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report_path