
#!/usr/bin/env python3
"""
Shared ONNX Runtime implementation for the offline EEG demonstrations.

This module supports both:

- verified FP32 ONNX inference;
- verified static INT8 QDQ ONNX inference.

Each demonstration follows the same controlled sequence:

1. Validate patient RDF metadata using SHACL.
2. Stop immediately when semantic validation fails.
3. Verify the selected model artifact using SHA256.
4. Validate the ONNX graph and model contract.
5. Verify processed EEG input.
6. Run CPU-only ONNX Runtime inference.
7. Calculate probabilities and frozen-threshold alarm decisions.
8. Save timestamped machine-readable evidence.

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Make the repository root importable.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from offline_demo.config import (
    ALARM_THRESHOLD,
    ALARM_THRESHOLD_POLICY,
    CLASS_NAMES,
    EXPECTED_CHANNELS,
    EXPECTED_CLASSES,
    EXPECTED_ONNX_OPSET,
    EXPECTED_TIMEPOINTS,
    EXPECTED_WINDOW_LABELS,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
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


@dataclass(frozen=True)
class ONNXDemoProfile:
    """
    Describe one ONNX deployment artifact.

    require_quantization_nodes:
        False for FP32.
        True for the static INT8 QDQ model.
    """

    display_name: str
    short_name: str
    model_path: Path
    expected_sha256: str
    require_quantization_nodes: bool


def validate_onnx_model(
    model_path: Path,
    profile: ONNXDemoProfile,
) -> dict[str, Any]:
    """
    Validate the ONNX graph and inspect its deployment contract.
    """
    onnx_model = onnx.load(
        str(model_path)
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

    node_types = [
        node.op_type
        for node in onnx_model.graph.node
    ]

    quantize_linear_nodes = node_types.count(
        "QuantizeLinear"
    )

    dequantize_linear_nodes = node_types.count(
        "DequantizeLinear"
    )

    if actual_opset != EXPECTED_ONNX_OPSET:
        raise RuntimeError(
            "Unexpected ONNX opset. "
            f"Expected {EXPECTED_ONNX_OPSET}, "
            f"received {actual_opset}."
        )

    if graph_inputs != [ONNX_INPUT_NAME]:
        raise RuntimeError(
            f"Unexpected ONNX inputs: {graph_inputs}"
        )

    if graph_outputs != [ONNX_OUTPUT_NAME]:
        raise RuntimeError(
            f"Unexpected ONNX outputs: {graph_outputs}"
        )

    if profile.require_quantization_nodes:
        if quantize_linear_nodes == 0:
            raise RuntimeError(
                "The INT8 graph contains no QuantizeLinear nodes."
            )

        if dequantize_linear_nodes == 0:
            raise RuntimeError(
                "The INT8 graph contains no DequantizeLinear nodes."
            )

    print(f"Actual ONNX opset       : {actual_opset}")
    print(f"ONNX inputs             : {graph_inputs}")
    print(f"ONNX outputs            : {graph_outputs}")
    print(
        f"QuantizeLinear nodes    : "
        f"{quantize_linear_nodes}"
    )
    print(
        f"DequantizeLinear nodes  : "
        f"{dequantize_linear_nodes}"
    )
    print("ONNX graph check        : PASS")

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
        "total_nodes": int(
            len(node_types)
        ),
        "quantize_linear_nodes": int(
            quantize_linear_nodes
        ),
        "dequantize_linear_nodes": int(
            dequantize_linear_nodes
        ),
    }


def create_cpu_session(
    model_path: Path,
) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session using CPUExecutionProvider only.
    """
    available_providers = (
        ort.get_available_providers()
    )

    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CPUExecutionProvider is unavailable."
        )

    session_options = ort.SessionOptions()

    # Use predictable settings for the offline prototype.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    active_providers = session.get_providers()

    if active_providers != [
        "CPUExecutionProvider"
    ]:
        raise RuntimeError(
            f"Unexpected active providers: {active_providers}"
        )

    return session


def verify_session_contract(
    session: ort.InferenceSession,
) -> dict[str, Any]:
    """
    Verify the ONNX Runtime input and output metadata.
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

    if input_metadata.name != ONNX_INPUT_NAME:
        raise RuntimeError(
            f"Unexpected ORT input: {input_metadata.name}"
        )

    if output_metadata.name != ONNX_OUTPUT_NAME:
        raise RuntimeError(
            f"Unexpected ORT output: {output_metadata.name}"
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
    Select and verify the three established demonstration windows.
    """
    for index in SELECTED_WINDOW_INDICES:
        if index < 0 or index >= len(epochs):
            raise IndexError(
                f"Demo window index {index} is unavailable."
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
            "The selected demonstration labels changed. "
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
    Run one ONNX Runtime inference batch.
    """
    inference_start = time.perf_counter()

    outputs = session.run(
        [ONNX_OUTPUT_NAME],
        {
            ONNX_INPUT_NAME: normalized_windows,
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
            "Unexpected inference output shape. "
            f"Expected {expected_shape}, received {logits.shape}."
        )

    if not np.isfinite(logits).all():
        raise RuntimeError(
            "Inference logits contain NaN or infinity."
        )

    return (
        logits,
        inference_duration_ms,
    )


def build_prediction_rows(
    logits: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Convert logits into probabilities and frozen-threshold decisions.
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
        >= ALARM_THRESHOLD
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
            "prediction_match": bool(
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
                ALARM_THRESHOLD
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
    Print predictions in a viva-friendly layout.
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


def run_demo(
    profile: ONNXDemoProfile,
) -> int:
    """
    Execute the complete SHACL-gated ONNX Runtime demonstration.
    """
    print_header(
        "ONTOLOGY-DRIVEN EEG ONNX OFFLINE RESEARCH PROTOTYPE"
    )

    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print(f"Patient          : {PATIENT_ID}")
    print(f"Model profile    : {profile.display_name}")
    print("Inference engine : ONNX Runtime")
    print("Inference device : CPU")

    # ------------------------------------------------------------------
    # Stage 1: SHACL semantic quality gate
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
            "The model and EEG data were not loaded."
        )
        return 2

    # ------------------------------------------------------------------
    # Stage 2: ONNX artifact verification
    # ------------------------------------------------------------------

    print_section(
        f"[2/4] {profile.display_name.upper()} ARTIFACT VERIFICATION"
    )

    model_hash = verify_artifact(
        path=profile.model_path,
        expected_sha256=profile.expected_sha256,
        description=profile.display_name,
    )

    onnx_contract = validate_onnx_model(
        profile.model_path,
        profile,
    )

    session = create_cpu_session(
        profile.model_path
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

    print(
        f"Alarm policy      : "
        f"{ALARM_THRESHOLD_POLICY}"
    )
    print(
        f"Alarm threshold   : "
        f"{ALARM_THRESHOLD:.4f}"
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

    (
        selected_epochs,
        selected_labels,
    ) = select_demo_windows(
        epochs,
        labels,
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
        f"[4/4] {profile.display_name.upper()} CPU INFERENCE"
    )

    print(
        "Note: These are illustrative held-out-patient windows, "
        "not a new performance evaluation."
    )

    (
        logits,
        inference_duration_ms,
    ) = run_onnx_inference(
        session,
        normalized_windows,
    )

    prediction_rows = build_prediction_rows(
        logits,
        selected_labels,
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
        "demo_type": (
            f"valid_{profile.short_name}_offline_demo"
        ),
        "model_profile": profile.display_name,
        "patient_id": PATIENT_ID,
        "ttl_path": str(
            PATIENT_TTL_PATH
        ),
        "npz_path": str(
            PATIENT_NPZ_PATH
        ),
        "model_path": str(
            profile.model_path
        ),
        "model_sha256": model_hash,
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
            **onnx_contract,
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
                ALARM_THRESHOLD_POLICY
            ),
            "alarm_threshold": float(
                ALARM_THRESHOLD
            ),
            "predictions": prediction_rows,
        },
        "interpretation": (
            "These windows are illustrative examples and do not "
            "replace the complete frozen held-out test evaluation."
        ),
    }

    report_path = save_json_report(
        report_name=(
            f"{PATIENT_ID}_{profile.short_name}_offline_demo"
        ),
        contents=report_contents,
    )

    print_header(
        f"{profile.display_name.upper()} OFFLINE PROTOTYPE: PASS"
    )

    print("SHACL validation : PASS")
    print("Model artifact   : VERIFIED")
    print("EEG input        : VERIFIED")
    print("ORT provider     : CPUExecutionProvider")
    print("Inference        : COMPLETED")
    print(f"Evidence report  : {report_path}")

    print(
        "\nReminder: These illustrative predictions do not replace "
        "the complete frozen held-out test metrics."
    )

    return 0