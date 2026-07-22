#!/usr/bin/env python3
"""
Valid-metadata PyTorch offline viva demonstration.

This script demonstrates the complete research-prototype execution path:

1. Validate patient RDF metadata using SHACL.
2. Stop immediately if semantic validation fails.
3. Verify the frozen P20 TCN checkpoint using SHA256.
4. Verify checkpoint metadata and model architecture.
5. Load and verify processed EEG data.
6. Run CPU inference on three illustrative windows.
7. Calculate class probabilities and alarm decisions.
8. Save a timestamped JSON evidence report.

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Make the project root importable.
#
# This file is stored at:
#     project_root/offline_demo/pytorch_demo.py
#
# parents[1] therefore points to the repository root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from offline_demo.config import (
    CLASS_NAMES,
    EXPECTED_CHANNELS,
    EXPECTED_CLASSES,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PATIENT_COUNT,
    EXPECTED_TIMEPOINTS,
    EXPECTED_WINDOW_LABELS,
    EXPECTED_CHECKPOINT_SHA256,
    PATIENT_ID,
    PATIENT_NPZ_PATH,
    PATIENT_TTL_PATH,
    PYTORCH_CHECKPOINT_PATH,
    SCIENTIFIC_STATUS,
    SELECTED_WINDOW_INDICES,
)

from offline_demo.demo_utils import (
    load_patient_npz,
    normalize_windows,
    print_header,
    print_section,
    run_shacl_validation,
    save_json_report,
    softmax,
    verify_artifact,
)

from src.models.tcn import SeizureTCN


def load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Load the frozen PyTorch checkpoint on the CPU.

    weights_only=False is required because the checkpoint contains
    metadata as well as the model state dictionary.

    The TypeError fallback supports older PyTorch versions that do not
    expose the weights_only argument.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The frozen checkpoint must be a dictionary."
        )

    return checkpoint


def verify_checkpoint_metadata(
    checkpoint: dict[str, Any],
) -> None:
    """
    Verify that the checkpoint is the intended frozen P20 baseline.

    This protects the viva demo from accidentally using:

    - the earlier 12-patient checkpoint;
    - the TCN-v2 experiment;
    - a model with a different input contract;
    - a checkpoint missing reproducibility metadata.
    """
    required_keys = {
        "model_name",
        "model_state_dict",
        "n_channels",
        "n_timepoints",
        "n_classes",
        "patient_count",
        "best_epoch",
        "best_val_multiclass_f1",
        "alarm_decision_threshold",
        "alarm_threshold_policy",
    }

    missing_keys = sorted(
        required_keys - set(checkpoint.keys())
    )

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint metadata is missing: {missing_keys}"
        )

    if checkpoint["model_name"] != "SeizureTCN":
        raise RuntimeError(
            "The frozen checkpoint is not a SeizureTCN checkpoint."
        )

    if int(checkpoint["patient_count"]) != EXPECTED_PATIENT_COUNT:
        raise RuntimeError(
            "Unexpected checkpoint patient count. "
            f"Expected {EXPECTED_PATIENT_COUNT}, "
            f"received {checkpoint['patient_count']}."
        )

    if int(checkpoint["n_channels"]) != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Unexpected checkpoint channel count."
        )

    if int(checkpoint["n_timepoints"]) != EXPECTED_TIMEPOINTS:
        raise RuntimeError(
            "Unexpected checkpoint timepoint count."
        )

    if int(checkpoint["n_classes"]) != EXPECTED_CLASSES:
        raise RuntimeError(
            "Unexpected checkpoint class count."
        )


def build_verified_model(
    checkpoint: dict[str, Any],
) -> SeizureTCN:
    """
    Reconstruct the exact TCN architecture and load the frozen weights.

    strict=True ensures every stored parameter matches the current
    architecture definition.
    """
    model = SeizureTCN(
        n_channels=int(checkpoint["n_channels"]),
        n_timepoints=int(checkpoint["n_timepoints"]),
        n_classes=int(checkpoint["n_classes"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Unexpected model parameter count. "
            f"Expected {EXPECTED_PARAMETER_COUNT}, "
            f"received {parameter_count}."
        )

    # Run one zero-input smoke test before loading real EEG data.
    dummy_input = torch.zeros(
        1,
        1,
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = model(dummy_input)

    if tuple(output.shape) != (1, EXPECTED_CLASSES):
        raise RuntimeError(
            "Unexpected model output shape. "
            f"Received {tuple(output.shape)}."
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "The model smoke-test output contains NaN or infinity."
        )

    return model


def select_demo_windows(
    epochs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the three fixed demonstration windows.

    The expected labels are checked so changes to the dataset cannot
    silently alter the meaning of the viva examples.
    """
    for index in SELECTED_WINDOW_INDICES:
        if index < 0 or index >= len(epochs):
            raise IndexError(
                f"Demonstration window index {index} is unavailable."
            )

    selected_epochs = epochs[
        SELECTED_WINDOW_INDICES
    ]

    selected_labels = labels[
        SELECTED_WINDOW_INDICES
    ]

    actual_labels = selected_labels.tolist()

    if actual_labels != EXPECTED_WINDOW_LABELS:
        raise RuntimeError(
            "The selected demonstration labels have changed. "
            f"Expected {EXPECTED_WINDOW_LABELS}, "
            f"received {actual_labels}."
        )

    return (
        selected_epochs,
        selected_labels,
    )


def run_pytorch_inference(
    model: SeizureTCN,
    normalized_windows: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Run CPU inference and return raw logits and total latency.
    """
    input_tensor = torch.from_numpy(
        normalized_windows
    )

    inference_start = time.perf_counter()

    with torch.inference_mode():
        logits = model(
            input_tensor
        )

    inference_duration_ms = (
        time.perf_counter() - inference_start
    ) * 1000.0

    logits_numpy = logits.cpu().numpy()

    if logits_numpy.shape != (
        len(normalized_windows),
        EXPECTED_CLASSES,
    ):
        raise RuntimeError(
            f"Unexpected inference output shape: {logits_numpy.shape}"
        )

    if not np.isfinite(logits_numpy).all():
        raise RuntimeError(
            "Inference logits contain NaN or infinity."
        )

    return (
        logits_numpy,
        inference_duration_ms,
    )


def build_prediction_rows(
    logits: np.ndarray,
    labels: np.ndarray,
    alarm_threshold: float,
) -> list[dict[str, Any]]:
    """
    Convert raw logits into class and alarm decisions.

    Multiclass prediction:
        argmax over Interictal, Pre-Ictal and Ictal probabilities.

    Alarm probability:
        P(Pre-Ictal) + P(Ictal)

    Alarm decision:
        alarm_probability >= validation-selected threshold
    """
    probabilities = softmax(
        logits
    )

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    )

    alarm_probabilities = (
        probabilities[:, 1]
        + probabilities[:, 2]
    )

    alarm_decisions = (
        alarm_probabilities
        >= alarm_threshold
    )

    rows: list[dict[str, Any]] = []

    for position, window_index in enumerate(
        SELECTED_WINDOW_INDICES
    ):
        true_class = int(
            labels[position]
        )

        predicted_class = int(
            predicted_classes[position]
        )

        row = {
            "window_index": int(window_index),
            "true_class": true_class,
            "true_class_name": CLASS_NAMES[true_class],
            "predicted_class": predicted_class,
            "predicted_class_name": (
                CLASS_NAMES[predicted_class]
            ),
            "prediction_match": (
                true_class == predicted_class
            ),
            "probabilities": {
                CLASS_NAMES[class_index]: float(
                    probabilities[
                        position,
                        class_index,
                    ]
                )
                for class_index in range(
                    EXPECTED_CLASSES
                )
            },
            "alarm_probability": float(
                alarm_probabilities[position]
            ),
            "alarm_threshold": float(
                alarm_threshold
            ),
            "alarm_decision": (
                "ALARM"
                if bool(alarm_decisions[position])
                else "NO ALARM"
            ),
        }

        rows.append(row)

    return rows


def print_prediction_rows(
    rows: list[dict[str, Any]],
) -> None:
    """
    Print each illustrative prediction in a viva-friendly format.
    """
    for row in rows:
        print_header(
            f"Window {row['window_index']} - "
            f"{row['true_class_name']} example"
        )

        print(
            f"True class      : "
            f"{row['true_class_name']}"
        )

        print(
            f"Predicted class : "
            f"{row['predicted_class_name']}"
        )

        print(
            "Prediction match: "
            f"{'YES' if row['prediction_match'] else 'NO'}"
        )

        print("\nClass probabilities")

        for class_name, probability in row[
            "probabilities"
        ].items():
            print(
                f"  {class_name:12s}: "
                f"{probability:.6f}"
            )

        print(
            f"\nAlarm probability: "
            f"{row['alarm_probability']:.6f}"
        )

        print(
            f"Alarm threshold  : "
            f"{row['alarm_threshold']:.6f}"
        )

        print(
            f"Alarm decision   : "
            f"{row['alarm_decision']}"
        )


def main() -> int:
    """
    Execute the complete valid PyTorch offline demonstration.
    """
    print_header(
        "ONTOLOGY-DRIVEN EEG OFFLINE RESEARCH PROTOTYPE"
    )

    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print(f"Patient          : {PATIENT_ID}")
    print("Inference engine : PyTorch")
    print("Inference device : CPU")

    # ------------------------------------------------------------------
    # Stage 1: Semantic validation
    # ------------------------------------------------------------------
    print_section(
        "[1/4] SHACL SEMANTIC QUALITY GATE"
    )

    print(f"TTL metadata : {PATIENT_TTL_PATH}")

    (
        shacl_conforms,
        shacl_output,
        shacl_duration_ms,
    ) = run_shacl_validation(
        PATIENT_TTL_PATH
    )

    if not shacl_conforms:
        print_header(
            "INFERENCE BLOCKED"
        )
        print(
            "The patient metadata failed SHACL validation."
        )
        print(
            "The frozen model and EEG data were not loaded."
        )
        return 2

    # ------------------------------------------------------------------
    # Stage 2: Frozen model verification
    # ------------------------------------------------------------------
    print_section(
        "[2/4] FROZEN TCN VERIFICATION"
    )

    checkpoint_hash = verify_artifact(
        path=PYTORCH_CHECKPOINT_PATH,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
        description="Checkpoint",
    )

    checkpoint = load_checkpoint(
        PYTORCH_CHECKPOINT_PATH
    )

    verify_checkpoint_metadata(
        checkpoint
    )

    model = build_verified_model(
        checkpoint
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    alarm_threshold = float(
        checkpoint["alarm_decision_threshold"]
    )

    print(f"Model          : SeizureTCN")
    print(
        f"Patients       : "
        f"{checkpoint['patient_count']}"
    )
    print(
        f"Best epoch     : "
        f"{checkpoint['best_epoch']}"
    )
    print(
        "Best val F1    : "
        f"{float(checkpoint['best_val_multiclass_f1']):.6f}"
    )
    print(
        f"Input          : "
        f"{EXPECTED_CHANNELS} channels x "
        f"{EXPECTED_TIMEPOINTS} samples"
    )
    print(
        f"Parameters     : "
        f"{parameter_count}"
    )
    print(
        f"Alarm policy   : "
        f"{checkpoint['alarm_threshold_policy']}"
    )
    print(
        f"Alarm threshold: "
        f"{alarm_threshold:.4f}"
    )
    print("Checkpoint verification: PASS")

    # ------------------------------------------------------------------
    # Stage 3: Processed EEG verification
    # ------------------------------------------------------------------
    print_section(
        "[3/4] PROCESSED EEG VERIFICATION"
    )

    patient_data = load_patient_npz(
        PATIENT_NPZ_PATH
    )

    if patient_data["patient_id"] != PATIENT_ID:
        raise RuntimeError(
            "Patient ID inside the NPZ does not match "
            f"the configured patient ID {PATIENT_ID}."
        )

    epochs = patient_data["epochs"]
    labels = patient_data["labels"]

    selected_epochs, selected_labels = (
        select_demo_windows(
            epochs,
            labels,
        )
    )

    normalized_windows = normalize_windows(
        selected_epochs
    )

    class_counts = np.bincount(
        labels,
        minlength=EXPECTED_CLASSES,
    )

    print(f"NPZ file      : {PATIENT_NPZ_PATH}")
    print(
        f"Patient ID    : "
        f"{patient_data['patient_id']}"
    )
    print(
        f"Epoch shape   : "
        f"{epochs.shape}"
    )
    print(
        f"Sampling rate : "
        f"{patient_data['sfreq']} Hz"
    )
    print(
        f"Channel count : "
        f"{len(patient_data['ch_names'])}"
    )
    print(
        "Class counts  : "
        f"Interictal={int(class_counts[0])}, "
        f"Pre-Ictal={int(class_counts[1])}, "
        f"Ictal={int(class_counts[2])}"
    )
    print(
        f"Demo batch    : "
        f"{normalized_windows.shape}"
    )
    print("Processed EEG verification: PASS")

    # ------------------------------------------------------------------
    # Stage 4: CPU inference
    # ------------------------------------------------------------------
    print_section(
        "[4/4] OFFLINE PYTORCH INFERENCE"
    )

    print(
        "Note: These are illustrative held-out-patient windows, "
        "not a new performance evaluation."
    )

    logits, inference_duration_ms = (
        run_pytorch_inference(
            model,
            normalized_windows,
        )
    )

    prediction_rows = build_prediction_rows(
        logits=logits,
        labels=selected_labels,
        alarm_threshold=alarm_threshold,
    )

    print_prediction_rows(
        prediction_rows
    )

    average_latency_ms = (
        inference_duration_ms
        / len(normalized_windows)
    )

    print_section(
        "Batch inference information"
    )

    print(
        f"Windows processed : "
        f"{len(normalized_windows)}"
    )
    print("Device            : CPU")
    print(
        f"Inference time    : "
        f"{inference_duration_ms:.3f} ms"
    )
    print(
        f"Average per window: "
        f"{average_latency_ms:.3f} ms"
    )

    # ------------------------------------------------------------------
    # Preserve demonstration evidence
    # ------------------------------------------------------------------
    report_contents = {
        "status": "pass",
        "scientific_status": SCIENTIFIC_STATUS,
        "demo_type": "valid_pytorch_offline_demo",
        "patient_id": PATIENT_ID,
        "ttl_path": str(
            PATIENT_TTL_PATH
        ),
        "npz_path": str(
            PATIENT_NPZ_PATH
        ),
        "checkpoint_path": str(
            PYTORCH_CHECKPOINT_PATH
        ),
        "checkpoint_sha256": checkpoint_hash,
        "shacl": {
            "conforms": True,
            "duration_ms": float(
                shacl_duration_ms
            ),
            "validator_output": (
                shacl_output
            ),
        },
        "model": {
            "name": "SeizureTCN",
            "patient_count": int(
                checkpoint["patient_count"]
            ),
            "best_epoch": int(
                checkpoint["best_epoch"]
            ),
            "best_validation_macro_f1": float(
                checkpoint[
                    "best_val_multiclass_f1"
                ]
            ),
            "parameter_count": int(
                parameter_count
            ),
            "input_contract": [
                "batch_size",
                1,
                EXPECTED_CHANNELS,
                EXPECTED_TIMEPOINTS,
            ],
            "output_classes": CLASS_NAMES,
            "alarm_threshold_policy": str(
                checkpoint[
                    "alarm_threshold_policy"
                ]
            ),
            "alarm_threshold": float(
                alarm_threshold
            ),
        },
        "dataset": {
            "epoch_shape": list(
                epochs.shape
            ),
            "sampling_rate_hz": float(
                patient_data["sfreq"]
            ),
            "channel_names": (
                patient_data["ch_names"]
            ),
            "class_counts": {
                CLASS_NAMES[class_index]: int(
                    class_counts[class_index]
                )
                for class_index in range(
                    EXPECTED_CLASSES
                )
            },
            "selected_window_indices": (
                SELECTED_WINDOW_INDICES
            ),
        },
        "inference": {
            "engine": "PyTorch",
            "device": "CPU",
            "batch_shape": list(
                normalized_windows.shape
            ),
            "total_latency_ms": float(
                inference_duration_ms
            ),
            "average_latency_ms": float(
                average_latency_ms
            ),
            "predictions": prediction_rows,
        },
        "interpretation": (
            "The selected windows are illustrative examples and "
            "do not replace held-out test-set performance metrics."
        ),
    }

    report_path = save_json_report(
        report_name=(
            f"{PATIENT_ID}_pytorch_offline_demo"
        ),
        contents=report_contents,
    )

    print_header(
        "PYTORCH OFFLINE PROTOTYPE: PASS"
    )

    print("SHACL validation : PASS")
    print("Frozen TCN       : VERIFIED")
    print("EEG input        : VERIFIED")
    print("Inference engine : PyTorch")
    print("Inference device : CPU")
    print("Inference        : COMPLETED")
    print(f"Evidence report  : {report_path}")

    print(
        "\nReminder: Prediction correctness for three illustrative "
        "windows is not a replacement for held-out test metrics."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())