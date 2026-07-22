#!/usr/bin/env python3
"""
Valid-metadata ONNX Runtime offline viva demonstration.

This script demonstrates the portable inference path:

1. Validate patient RDF metadata using SHACL.
2. Stop immediately if semantic validation fails.
3. Verify the exported FP32 ONNX artifact using SHA256.
4. Validate the ONNX graph and model contract.
5. Load and verify processed EEG data.
6. Run CPU inference using ONNX Runtime.
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
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Make the project root importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from offline_demo.config import (
    CLASS_NAMES,
    EXPECTED_CHANNELS,
    EXPECTED_CLASSES,
    EXPECTED_ONNX_OPSET,
    EXPECTED_ONNX_SHA256,
    EXPECTED_TIMEPOINTS,
    EXPECTED_WINDOW_LABELS,
    ONNX_MODEL_PATH,
    PATIENT_ID,
    PATIENT_NPZ_PATH,
    PATIENT_TTL_PATH,
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


def validate_onnx_model(
    onnx_path: Path,
) -> dict[str, Any]:
    """
    Validate the ONNX graph and confirm its deployment contract.

    The expected contract is:

        input:
            eeg_input
            float32[batch_size, 1, 20, 512]

        output:
            logits
            float32[batch_size, 3]
    """
    onnx_model = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        onnx_model,
        full_check=True,
    )

    default_opsets = [
        int(opset.version)
        for opset in onnx_model.opset_import
        if opset.domain in {"", "ai.onnx"}
    ]

    if not default_opsets:
        raise RuntimeError(
            "The ONNX model does not declare a default opset."
        )

    actual_opset = max(
        default_opsets
    )

    graph_inputs = [
        value.name
        for value in onnx_model.graph.input
    ]

    graph_outputs = [
        value.name
        for value in onnx_model.graph.output
    ]

    if actual_opset != EXPECTED_ONNX_OPSET:
        raise RuntimeError(
            "Unexpected ONNX opset. "
            f"Expected {EXPECTED_ONNX_OPSET}, "
            f"received {actual_opset}."
        )

    if graph_inputs != ["eeg_input"]:
        raise RuntimeError(
            f"Unexpected ONNX input names: {graph_inputs}"
        )

    if graph_outputs != ["logits"]:
        raise RuntimeError(
            f"Unexpected ONNX output names: {graph_outputs}"
        )

    print(f"Actual ONNX opset : {actual_opset}")
    print(f"ONNX inputs       : {graph_inputs}")
    print(f"ONNX outputs      : {graph_outputs}")
    print("ONNX graph check  : PASS")

    return {
        "actual_opset": actual_opset,
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "ir_version": int(
            onnx_model.ir_version
        ),
        "producer_name": (
            onnx_model.producer_name
        ),
        "producer_version": (
            onnx_model.producer_version
        ),
    }


def create_cpu_session(
    onnx_path: Path,
) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session using only CPUExecutionProvider.

    This explicitly prevents AzureExecutionProvider from being selected
    during the viva demo.
    """
    available_providers = (
        ort.get_available_providers()
    )

    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CPUExecutionProvider is not available."
        )

    session_options = ort.SessionOptions()

    # Keep execution predictable for the offline prototype.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    active_providers = session.get_providers()

    if active_providers != ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"Unexpected active providers: {active_providers}"
        )

    return session


def verify_session_contract(
    session: ort.InferenceSession,
) -> dict[str, Any]:
    """
    Inspect the ONNX Runtime input and output contract.

    Dynamic batch size is represented by a symbolic dimension.
    The remaining dimensions must stay fixed.
    """
    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    if len(session_inputs) != 1:
        raise RuntimeError(
            "Expected exactly one ONNX Runtime input."
        )

    if len(session_outputs) != 1:
        raise RuntimeError(
            "Expected exactly one ONNX Runtime output."
        )

    input_metadata = session_inputs[0]
    output_metadata = session_outputs[0]

    if input_metadata.name != "eeg_input":
        raise RuntimeError(
            f"Unexpected ORT input name: {input_metadata.name}"
        )

    if output_metadata.name != "logits":
        raise RuntimeError(
            f"Unexpected ORT output name: {output_metadata.name}"
        )

    input_shape = input_metadata.shape
    output_shape = output_metadata.shape

    if len(input_shape) != 4:
        raise RuntimeError(
            f"Unexpected ORT input rank: {input_shape}"
        )

    if input_shape[1:] != [
        1,
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
    ]:
        raise RuntimeError(
            f"Unexpected ORT input shape: {input_shape}"
        )

    if len(output_shape) != 2:
        raise RuntimeError(
            f"Unexpected ORT output rank: {output_shape}"
        )

    if output_shape[1] != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Unexpected ORT output shape: {output_shape}"
        )

    return {
        "input_name": input_metadata.name,
        "input_shape": input_shape,
        "input_type": input_metadata.type,
        "output_name": output_metadata.name,
        "output_shape": output_shape,
        "output_type": output_metadata.type,
    }


def select_demo_windows(
    epochs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select and verify the same three illustrative windows used by
    the PyTorch offline demonstration.
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


def run_onnx_inference(
    session: ort.InferenceSession,
    normalized_windows: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Run CPU inference through ONNX Runtime.

    Returns:
        logits:
            Raw three-class output values.

        duration_ms:
            Total batch inference time.
    """
    inference_start = time.perf_counter()

    outputs = session.run(
        ["logits"],
        {
            "eeg_input": normalized_windows,
        },
    )

    inference_duration_ms = (
        time.perf_counter() - inference_start
    ) * 1000.0

    logits = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    expected_shape = (
        len(normalized_windows),
        EXPECTED_CLASSES,
    )

    if logits.shape != expected_shape:
        raise RuntimeError(
            "Unexpected ONNX inference output shape. "
            f"Expected {expected_shape}, received {logits.shape}."
        )

    if not np.isfinite(logits).all():
        raise RuntimeError(
            "ONNX inference logits contain NaN or infinity."
        )

    return (
        logits,
        inference_duration_ms,
    )


def build_prediction_rows(
    logits: np.ndarray,
    labels: np.ndarray,
    alarm_threshold: float,
) -> list[dict[str, Any]]:
    """
    Convert ONNX logits into class probabilities and alarm decisions.

    Alarm probability:
        P(Pre-Ictal) + P(Ictal)
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

        rows.append({
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
        })

    return rows


def print_prediction_rows(
    rows: list[dict[str, Any]],
) -> None:
    """
    Print ONNX predictions in a clear viva-friendly layout.
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
    Execute the complete valid ONNX Runtime offline demonstration.
    """
    print_header(
        "ONTOLOGY-DRIVEN EEG ONNX OFFLINE RESEARCH PROTOTYPE"
    )

    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print(f"Patient          : {PATIENT_ID}")
    print("Inference engine : ONNX Runtime")
    print("Inference device : CPU")

    # ------------------------------------------------------------------
    # Stage 1: SHACL quality gate
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
            "The ONNX model and EEG data were not loaded."
        )
        return 2

    # ------------------------------------------------------------------
    # Stage 2: ONNX artifact verification
    # ------------------------------------------------------------------
    print_section(
        "[2/4] FP32 ONNX ARTIFACT VERIFICATION"
    )

    onnx_hash = verify_artifact(
        path=ONNX_MODEL_PATH,
        expected_sha256=EXPECTED_ONNX_SHA256,
        description="ONNX model",
    )

    onnx_contract = validate_onnx_model(
        ONNX_MODEL_PATH
    )

    session = create_cpu_session(
        ONNX_MODEL_PATH
    )

    session_contract = verify_session_contract(
        session
    )

    active_providers = session.get_providers()

    print(f"ORT providers     : {active_providers}")
    print(
        f"ORT input shape   : "
        f"{session_contract['input_shape']}"
    )
    print(
        f"ORT output shape  : "
        f"{session_contract['output_shape']}"
    )
    print("ONNX artifact verification: PASS")

    # The threshold was frozen with the selected baseline checkpoint.
    alarm_threshold = 0.17
    alarm_threshold_policy = "specificity_constrained"

    print(
        f"Alarm policy      : "
        f"{alarm_threshold_policy}"
    )
    print(
        f"Alarm threshold   : "
        f"{alarm_threshold:.4f}"
    )

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
    print(f"Epoch shape   : {epochs.shape}")
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
    # Stage 4: ONNX Runtime inference
    # ------------------------------------------------------------------
    print_section(
        "[4/4] OFFLINE ONNX RUNTIME INFERENCE"
    )

    print(
        "Note: These are illustrative held-out-patient windows, "
        "not a new performance evaluation."
    )

    logits, inference_duration_ms = (
        run_onnx_inference(
            session,
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
    print("Execution provider: CPUExecutionProvider")
    print(
        f"Inference time    : "
        f"{inference_duration_ms:.3f} ms"
    )
    print(
        f"Average per window: "
        f"{average_latency_ms:.3f} ms"
    )

    # ------------------------------------------------------------------
    # Preserve evidence
    # ------------------------------------------------------------------
    report_contents = {
        "status": "pass",
        "scientific_status": SCIENTIFIC_STATUS,
        "demo_type": "valid_onnx_offline_demo",
        "patient_id": PATIENT_ID,
        "ttl_path": str(
            PATIENT_TTL_PATH
        ),
        "npz_path": str(
            PATIENT_NPZ_PATH
        ),
        "onnx_path": str(
            ONNX_MODEL_PATH
        ),
        "onnx_sha256": onnx_hash,
        "shacl": {
            "conforms": True,
            "duration_ms": float(
                shacl_duration_ms
            ),
            "validator_output": (
                shacl_output
            ),
        },
        "onnx": {
            "actual_opset": int(
                onnx_contract["actual_opset"]
            ),
            "graph_inputs": (
                onnx_contract["graph_inputs"]
            ),
            "graph_outputs": (
                onnx_contract["graph_outputs"]
            ),
            "ir_version": int(
                onnx_contract["ir_version"]
            ),
            "providers": active_providers,
            "session_contract": session_contract,
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
            "engine": "ONNX Runtime",
            "provider": "CPUExecutionProvider",
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
            "alarm_threshold_policy": (
                alarm_threshold_policy
            ),
            "alarm_threshold": float(
                alarm_threshold
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
            f"{PATIENT_ID}_onnx_offline_demo"
        ),
        contents=report_contents,
    )

    print_header(
        "ONNX OFFLINE PROTOTYPE: PASS"
    )

    print("SHACL validation : PASS")
    print("ONNX artifact    : VERIFIED")
    print("EEG input        : VERIFIED")
    print("ORT provider     : CPUExecutionProvider")
    print("Inference        : COMPLETED")
    print(f"Evidence report  : {report_path}")

    print(
        "\nReminder: Prediction correctness for three illustrative "
        "windows is not a replacement for held-out test metrics."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())