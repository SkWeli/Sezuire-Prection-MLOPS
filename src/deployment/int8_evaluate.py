#!/usr/bin/env python3
"""
Evaluate FP32 and INT8 ONNX TCN models on the complete frozen test split.

This script evaluates every processed EEG window belonging to the three
patient IDs stored under:

    checkpoint["split_patient_ids"]["test"]

The evaluation is leakage-safe:

- training patients are not evaluated;
- validation patients are not evaluated;
- the alarm threshold is not tuned using test data;
- the frozen validation-selected threshold is reused unchanged.

Outputs include:

- multiclass accuracy
- balanced accuracy
- macro precision
- macro recall
- macro specificity
- macro F1
- macro one-vs-rest AUC
- per-class precision, recall, specificity and F1
- confusion matrices
- false alarms per hour
- FP32-versus-INT8 prediction disagreement
- FP32-versus-INT8 alarm disagreement
- model size and CPU inference latency

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Project locations
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "frozen"
    / "seizure_tcn_p20_baseline_review_0d6774d3.pt"
)

DEFAULT_FP32_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_fp32.onnx"
)

DEFAULT_INT8_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_int8_qdq.onnx"
)

DEFAULT_PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tusz"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "quantization"
    / "test_split"
)


# ---------------------------------------------------------------------------
# Frozen artifact identities
# ---------------------------------------------------------------------------

EXPECTED_CHECKPOINT_SHA256 = (
    "0d6774d36f2f040b4ce6ae5f9964fc30"
    "b004a7fe96b4a9fdfd401f733921a4e7"
)

EXPECTED_FP32_SHA256 = (
    "af31d5a99ac683786b70abc4eea774d9"
    "c3b9564af41856358280060cd2f77420"
)

EXPECTED_INT8_SHA256 = (
    "9fbac2f59b5acd0276036f8ab9ea65c6"
    "bc8555d71b5ba9c16124062889e43969"
)


# ---------------------------------------------------------------------------
# Frozen model and dataset contract
# ---------------------------------------------------------------------------

EXPECTED_CHANNELS = 20
EXPECTED_TIMEPOINTS = 512
EXPECTED_CLASSES = 3

INPUT_NAME = "eeg_input"
OUTPUT_NAME = "logits"

CLASS_NAMES = [
    "Interictal",
    "Pre-Ictal",
    "Ictal",
]

# This threshold was selected using validation data and stored in the
# frozen checkpoint. It must not be retuned using the test split.
EXPECTED_ALARM_THRESHOLD = 0.17

# Each processed window advances by two seconds.
WINDOW_STEP_SECONDS = 2.0

SCIENTIFIC_STATUS = (
    "Research prototype only; not clinically validated; "
    "not a medical device."
)


def sha256_file(path: Path) -> str:
    """Calculate the SHA256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_artifact(
    path: Path,
    expected_hash: str,
    description: str,
) -> str:
    """
    Confirm that an artifact exists and matches the frozen identity.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    actual_hash = sha256_file(path)

    print(f"{description} path   : {path}")
    print(f"{description} SHA256 : {actual_hash}")

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{description} SHA256 verification failed."
        )

    print(f"{description} verification: PASS")

    return actual_hash


def load_frozen_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Load and validate the frozen checkpoint metadata.
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
            "The checkpoint must contain a metadata dictionary."
        )

    required_keys = {
        "model_name",
        "split_patient_ids",
        "patient_count",
        "n_channels",
        "n_timepoints",
        "n_classes",
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
            "The frozen checkpoint is not a SeizureTCN model."
        )

    if int(checkpoint["patient_count"]) != 20:
        raise RuntimeError(
            "Expected the frozen 20-patient experiment."
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

    threshold = float(
        checkpoint["alarm_decision_threshold"]
    )

    if not np.isclose(
        threshold,
        EXPECTED_ALARM_THRESHOLD,
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "The checkpoint alarm threshold changed. "
            f"Expected {EXPECTED_ALARM_THRESHOLD}, "
            f"received {threshold}."
        )

    return checkpoint


def validate_patient_splits(
    checkpoint: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """
    Verify that train, validation and test patient IDs do not overlap.
    """
    split_ids = checkpoint["split_patient_ids"]

    expected_keys = {
        "train",
        "validation",
        "test",
    }

    if set(split_ids.keys()) != expected_keys:
        raise RuntimeError(
            f"Unexpected split keys: {sorted(split_ids.keys())}"
        )

    training_ids = list(split_ids["train"])
    validation_ids = list(split_ids["validation"])
    test_ids = list(split_ids["test"])

    if len(training_ids) != 14:
        raise RuntimeError(
            f"Expected 14 training patients, received {len(training_ids)}."
        )

    if len(validation_ids) != 3:
        raise RuntimeError(
            f"Expected 3 validation patients, received {len(validation_ids)}."
        )

    if len(test_ids) != 3:
        raise RuntimeError(
            f"Expected 3 test patients, received {len(test_ids)}."
        )

    train_set = set(training_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)

    if train_set & validation_set:
        raise RuntimeError(
            "Training and validation patient sets overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Training and test patient sets overlap."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Validation and test patient sets overlap."
        )

    print(f"Training patients  : {len(training_ids)}")
    print(f"Validation patients: {len(validation_ids)}")
    print(f"Test patients      : {len(test_ids)}")
    print(f"Frozen test IDs    : {test_ids}")
    print("Patient leakage check: PASS")

    return (
        training_ids,
        validation_ids,
        test_ids,
    )


def validate_onnx_contract(
    model_path: Path,
    description: str,
    require_quantization_nodes: bool,
) -> dict[str, Any]:
    """
    Validate an ONNX graph and its input/output names.
    """
    model = onnx.load(
        str(model_path)
    )

    onnx.checker.check_model(
        model,
        full_check=True,
    )

    graph_inputs = [
        value.name
        for value in model.graph.input
    ]

    graph_outputs = [
        value.name
        for value in model.graph.output
    ]

    if graph_inputs != [INPUT_NAME]:
        raise RuntimeError(
            f"Unexpected {description} inputs: {graph_inputs}"
        )

    if graph_outputs != [OUTPUT_NAME]:
        raise RuntimeError(
            f"Unexpected {description} outputs: {graph_outputs}"
        )

    node_types = [
        node.op_type
        for node in model.graph.node
    ]

    quantize_count = node_types.count(
        "QuantizeLinear"
    )

    dequantize_count = node_types.count(
        "DequantizeLinear"
    )

    if require_quantization_nodes:
        if quantize_count == 0:
            raise RuntimeError(
                "INT8 graph contains no QuantizeLinear nodes."
            )

        if dequantize_count == 0:
            raise RuntimeError(
                "INT8 graph contains no DequantizeLinear nodes."
            )

    default_opsets = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in {"", "ai.onnx"}
    ]

    if not default_opsets:
        raise RuntimeError(
            f"{description} has no default ONNX opset."
        )

    information = {
        "actual_opset": max(default_opsets),
        "total_nodes": len(node_types),
        "quantize_linear_nodes": quantize_count,
        "dequantize_linear_nodes": dequantize_count,
        "inputs": graph_inputs,
        "outputs": graph_outputs,
    }

    print(f"{description} graph check: PASS")

    return information


def create_cpu_session(
    model_path: Path,
) -> ort.InferenceSession:
    """
    Create a single-thread CPU ONNX Runtime session.

    Both models receive identical session settings so their latency
    measurements are directly comparable on this machine.
    """
    session_options = ort.SessionOptions()

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

    if session.get_providers() != [
        "CPUExecutionProvider"
    ]:
        raise RuntimeError(
            "The session did not use CPUExecutionProvider exclusively."
        )

    return session


def load_patient_npz(
    processed_data_directory: Path,
    patient_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load one frozen test patient's processed EEG windows.

    allow_pickle=True is required because the trusted local NPZ stores
    channel names as an object array.
    """
    npz_path = (
        processed_data_directory
        / patient_id
        / f"{patient_id}.npz"
    )

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Test patient NPZ was not found: {npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        required_keys = {
            "epochs",
            "labels",
        }

        missing_keys = sorted(
            required_keys - set(data.files)
        )

        if missing_keys:
            raise RuntimeError(
                f"{patient_id} NPZ is missing fields: {missing_keys}"
            )

        epochs = np.asarray(
            data["epochs"],
            dtype=np.float32,
        )

        labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        )

        sampling_rate = (
            float(np.asarray(data["sfreq"]).item())
            if "sfreq" in data.files
            else 128.0
        )

    if epochs.ndim != 3:
        raise RuntimeError(
            f"Unexpected epoch rank for {patient_id}: {epochs.shape}"
        )

    if tuple(epochs.shape[1:]) != (
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
    ):
        raise RuntimeError(
            f"Unexpected epoch shape for {patient_id}: {epochs.shape}"
        )

    if len(epochs) != len(labels):
        raise RuntimeError(
            f"Epoch/label count mismatch for {patient_id}."
        )

    invalid_labels = set(
        int(value)
        for value in np.unique(labels)
    ) - {0, 1, 2}

    if invalid_labels:
        raise RuntimeError(
            f"Unexpected labels for {patient_id}: {sorted(invalid_labels)}"
        )

    metadata = {
        "patient_id": patient_id,
        "npz_path": str(npz_path),
        "window_count": int(len(epochs)),
        "sampling_rate_hz": sampling_rate,
        "class_counts": {
            CLASS_NAMES[class_index]: int(
                np.sum(labels == class_index)
            )
            for class_index in range(EXPECTED_CLASSES)
        },
    }

    return (
        epochs,
        labels,
        metadata,
    )


def normalize_batch(
    windows: np.ndarray,
) -> np.ndarray:
    """
    Apply per-window, per-channel z-score normalization.

    Input:
        [batch, 20, 512]

    Output:
        [batch, 1, 20, 512]
    """
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
    """Calculate numerically stable softmax probabilities."""
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(
        shifted
    )

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )


def infer_patient(
    session: ort.InferenceSession,
    epochs: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    """
    Run one model across all windows for one patient.

    Returns:
        probabilities:
            Array shaped [window_count, 3].

        total_latency_ms:
            Total ONNX Runtime execution time. NPZ reading and
            normalization are excluded.

        batch_count:
            Number of inference batches executed.
    """
    probability_batches: list[np.ndarray] = []
    total_latency_ms = 0.0
    batch_count = 0

    for start_index in range(
        0,
        len(epochs),
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            len(epochs),
        )

        normalized_batch = normalize_batch(
            epochs[start_index:end_index]
        )

        start_time = time.perf_counter()

        outputs = session.run(
            [OUTPUT_NAME],
            {
                INPUT_NAME: normalized_batch,
            },
        )

        total_latency_ms += (
            time.perf_counter() - start_time
        ) * 1000.0

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        expected_shape = (
            end_index - start_index,
            EXPECTED_CLASSES,
        )

        if logits.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected inference output: {logits.shape}"
            )

        if not np.isfinite(logits).all():
            raise RuntimeError(
                "Inference produced NaN or infinity."
            )

        probability_batches.append(
            softmax(logits)
        )

        batch_count += 1

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )

    return (
        probabilities,
        total_latency_ms,
        batch_count,
    )


def calculate_class_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_index: int,
) -> float:
    """
    Calculate one-vs-rest specificity for one class.

    Specificity = TN / (TN + FP)
    """
    true_positive_class = (
        y_true == class_index
    )

    predicted_positive_class = (
        y_pred == class_index
    )

    true_negatives = int(
        np.sum(
            (~true_positive_class)
            & (~predicted_positive_class)
        )
    )

    false_positives = int(
        np.sum(
            (~true_positive_class)
            & predicted_positive_class
        )
    )

    denominator = (
        true_negatives + false_positives
    )

    if denominator == 0:
        return 1.0

    return float(
        true_negatives / denominator
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    alarm_threshold: float,
) -> dict[str, Any]:
    """
    Calculate multiclass and binary-alarm metrics.

    Three-class metrics use argmax predictions.

    Alarm metrics use:
        P(Pre-Ictal) + P(Ictal) >= frozen threshold
    """
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2],
    )

    per_class_precision = precision_score(
        y_true,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    per_class_recall = recall_score(
        y_true,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    per_class_f1 = f1_score(
        y_true,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    per_class_specificity = np.asarray(
        [
            calculate_class_specificity(
                y_true,
                predictions,
                class_index,
            )
            for class_index in range(
                EXPECTED_CLASSES
            )
        ],
        dtype=np.float64,
    )

    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=[0, 1, 2],
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = float("nan")

    alarm_probabilities = (
        probabilities[:, 1]
        + probabilities[:, 2]
    )

    alarm_decisions = (
        alarm_probabilities
        >= alarm_threshold
    )

    true_alarm_labels = (
        y_true > 0
    )

    alarm_true_positives = int(
        np.sum(
            true_alarm_labels
            & alarm_decisions
        )
    )

    alarm_true_negatives = int(
        np.sum(
            (~true_alarm_labels)
            & (~alarm_decisions)
        )
    )

    alarm_false_positives = int(
        np.sum(
            (~true_alarm_labels)
            & alarm_decisions
        )
    )

    alarm_false_negatives = int(
        np.sum(
            true_alarm_labels
            & (~alarm_decisions)
        )
    )

    evaluated_duration_hours = (
        len(y_true)
        * WINDOW_STEP_SECONDS
        / 3600.0
    )

    false_alarms_per_hour = (
        alarm_false_positives
        / evaluated_duration_hours
        if evaluated_duration_hours > 0
        else 0.0
    )

    class_metrics = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        class_metrics[class_name] = {
            "precision": float(
                per_class_precision[class_index]
            ),
            "recall": float(
                per_class_recall[class_index]
            ),
            "specificity": float(
                per_class_specificity[class_index]
            ),
            "f1": float(
                per_class_f1[class_index]
            ),
            "support": int(
                np.sum(y_true == class_index)
            ),
        }

    return {
        "window_count": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_specificity": float(
            per_class_specificity.mean()
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_auc_ovr": macro_auc,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
        "alarm": {
            "threshold": float(
                alarm_threshold
            ),
            "true_positives": alarm_true_positives,
            "true_negatives": alarm_true_negatives,
            "false_positives": alarm_false_positives,
            "false_negatives": alarm_false_negatives,
            "sensitivity": float(
                alarm_true_positives
                / (
                    alarm_true_positives
                    + alarm_false_negatives
                )
                if (
                    alarm_true_positives
                    + alarm_false_negatives
                ) > 0
                else 0.0
            ),
            "specificity": float(
                alarm_true_negatives
                / (
                    alarm_true_negatives
                    + alarm_false_positives
                )
                if (
                    alarm_true_negatives
                    + alarm_false_positives
                ) > 0
                else 0.0
            ),
            "evaluated_duration_hours": float(
                evaluated_duration_hours
            ),
            "false_alarms_per_hour": float(
                false_alarms_per_hour
            ),
        },
        "predictions": predictions,
        "alarm_probabilities": alarm_probabilities,
        "alarm_decisions": alarm_decisions,
    }


def serializable_metrics(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove internal NumPy prediction arrays before JSON serialization.
    """
    return {
        key: value
        for key, value in metrics.items()
        if key not in {
            "predictions",
            "alarm_probabilities",
            "alarm_decisions",
        }
    }


def save_confusion_matrix_csv(
    confusion: list[list[int]],
    output_path: Path,
) -> None:
    """Save a labelled three-class confusion matrix."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Actual / Predicted",
                *CLASS_NAMES,
            ]
        )

        for class_name, row in zip(
            CLASS_NAMES,
            confusion,
        ):
            writer.writerow(
                [
                    class_name,
                    *row,
                ]
            )


def parse_arguments() -> argparse.Namespace:
    """Parse full test-split evaluation settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen FP32 and INT8 ONNX TCN models "
            "across the complete held-out patient split."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--fp32-model",
        type=Path,
        default=DEFAULT_FP32_MODEL_PATH,
    )

    parser.add_argument(
        "--int8-model",
        type=Path,
        default=DEFAULT_INT8_MODEL_PATH,
    )

    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIRECTORY,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--maximum-acceptable-f1-drop",
        type=float,
        default=0.02,
        help=(
            "Decision-support threshold for absolute macro-F1 "
            "degradation. It is reported as an engineering criterion, "
            "not treated as a universal scientific standard."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Run full held-out FP32-versus-INT8 evaluation.
    """
    args = parse_arguments()

    checkpoint_path = args.checkpoint.resolve()
    fp32_path = args.fp32_model.resolve()
    int8_path = args.int8_model.resolve()
    processed_directory = (
        args.processed_data_dir.resolve()
    )
    report_directory = (
        args.report_dir.resolve()
    )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than zero."
        )

    print("=" * 80)
    print("FULL HELD-OUT TEST EVALUATION - FP32 VS INT8")
    print("=" * 80)
    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print("Execution provider: CPUExecutionProvider")
    print(
        "Threshold source: frozen validation-selected checkpoint value"
    )
    print()

    checkpoint_hash = verify_artifact(
        checkpoint_path,
        EXPECTED_CHECKPOINT_SHA256,
        "Checkpoint",
    )

    fp32_hash = verify_artifact(
        fp32_path,
        EXPECTED_FP32_SHA256,
        "FP32 ONNX",
    )

    int8_hash = verify_artifact(
        int8_path,
        EXPECTED_INT8_SHA256,
        "INT8 ONNX",
    )

    checkpoint = load_frozen_checkpoint(
        checkpoint_path
    )

    (
        training_patient_ids,
        validation_patient_ids,
        test_patient_ids,
    ) = validate_patient_splits(
        checkpoint
    )

    alarm_threshold = float(
        checkpoint["alarm_decision_threshold"]
    )

    print(
        f"Frozen alarm threshold: {alarm_threshold:.4f}"
    )
    print(
        f"Threshold policy       : "
        f"{checkpoint['alarm_threshold_policy']}"
    )

    fp32_graph = validate_onnx_contract(
        fp32_path,
        "FP32",
        require_quantization_nodes=False,
    )

    int8_graph = validate_onnx_contract(
        int8_path,
        "INT8",
        require_quantization_nodes=True,
    )

    fp32_session = create_cpu_session(
        fp32_path
    )

    int8_session = create_cpu_session(
        int8_path
    )

    all_labels: list[np.ndarray] = []
    all_fp32_probabilities: list[np.ndarray] = []
    all_int8_probabilities: list[np.ndarray] = []

    patient_records: list[dict[str, Any]] = []

    total_fp32_latency_ms = 0.0
    total_int8_latency_ms = 0.0
    total_fp32_batches = 0
    total_int8_batches = 0

    print("\nEvaluating frozen test patients")
    print("-" * 80)

    for patient_id in test_patient_ids:
        (
            epochs,
            labels,
            patient_metadata,
        ) = load_patient_npz(
            processed_directory,
            patient_id,
        )

        (
            fp32_probabilities,
            fp32_latency_ms,
            fp32_batch_count,
        ) = infer_patient(
            fp32_session,
            epochs,
            args.batch_size,
        )

        (
            int8_probabilities,
            int8_latency_ms,
            int8_batch_count,
        ) = infer_patient(
            int8_session,
            epochs,
            args.batch_size,
        )

        if fp32_probabilities.shape != (
            int8_probabilities.shape
        ):
            raise RuntimeError(
                f"Output shape mismatch for {patient_id}."
            )

        all_labels.append(labels)
        all_fp32_probabilities.append(
            fp32_probabilities
        )
        all_int8_probabilities.append(
            int8_probabilities
        )

        total_fp32_latency_ms += (
            fp32_latency_ms
        )

        total_int8_latency_ms += (
            int8_latency_ms
        )

        total_fp32_batches += (
            fp32_batch_count
        )

        total_int8_batches += (
            int8_batch_count
        )

        patient_record = {
            **patient_metadata,
            "fp32_latency_ms": float(
                fp32_latency_ms
            ),
            "int8_latency_ms": float(
                int8_latency_ms
            ),
            "fp32_batch_count": int(
                fp32_batch_count
            ),
            "int8_batch_count": int(
                int8_batch_count
            ),
        }

        patient_records.append(
            patient_record
        )

        print(
            f"{patient_id}: "
            f"{len(labels)} windows | "
            f"FP32={fp32_latency_ms:.2f} ms | "
            f"INT8={int8_latency_ms:.2f} ms"
        )

        # Release the large patient epoch array before loading the next file.
        del epochs

    y_true = np.concatenate(
        all_labels,
        axis=0,
    )

    fp32_probabilities = np.concatenate(
        all_fp32_probabilities,
        axis=0,
    )

    int8_probabilities = np.concatenate(
        all_int8_probabilities,
        axis=0,
    )

    if not (
        len(y_true)
        == len(fp32_probabilities)
        == len(int8_probabilities)
    ):
        raise RuntimeError(
            "Combined test arrays have inconsistent lengths."
        )

    fp32_metrics = calculate_metrics(
        y_true,
        fp32_probabilities,
        alarm_threshold,
    )

    int8_metrics = calculate_metrics(
        y_true,
        int8_probabilities,
        alarm_threshold,
    )

    prediction_disagreements = int(
        np.sum(
            fp32_metrics["predictions"]
            != int8_metrics["predictions"]
        )
    )

    alarm_disagreements = int(
        np.sum(
            fp32_metrics["alarm_decisions"]
            != int8_metrics["alarm_decisions"]
        )
    )

    maximum_probability_difference = float(
        np.max(
            np.abs(
                fp32_probabilities
                - int8_probabilities
            )
        )
    )

    mean_probability_difference = float(
        np.mean(
            np.abs(
                fp32_probabilities
                - int8_probabilities
            )
        )
    )

    fp32_size_bytes = fp32_path.stat().st_size
    int8_size_bytes = int8_path.stat().st_size

    size_reduction_percent = (
        1.0
        - int8_size_bytes
        / fp32_size_bytes
    ) * 100.0

    total_windows = len(y_true)

    fp32_mean_window_latency_ms = (
        total_fp32_latency_ms
        / total_windows
    )

    int8_mean_window_latency_ms = (
        total_int8_latency_ms
        / total_windows
    )

    latency_change_percent = (
        (
            int8_mean_window_latency_ms
            - fp32_mean_window_latency_ms
        )
        / fp32_mean_window_latency_ms
    ) * 100.0

    macro_f1_change = (
        int8_metrics["macro_f1"]
        - fp32_metrics["macro_f1"]
    )

    pre_ictal_recall_change = (
        int8_metrics["class_metrics"][
            "Pre-Ictal"
        ]["recall"]
        - fp32_metrics["class_metrics"][
            "Pre-Ictal"
        ]["recall"]
    )

    ictal_recall_change = (
        int8_metrics["class_metrics"][
            "Ictal"
        ]["recall"]
        - fp32_metrics["class_metrics"][
            "Ictal"
        ]["recall"]
    )

    retained_by_engineering_criterion = (
        macro_f1_change
        >= -args.maximum_acceptable_f1_drop
    )

    print("\nFP32 metrics")
    print("-" * 80)
    print(
        f"Accuracy          : {fp32_metrics['accuracy']:.6f}"
    )
    print(
        f"Balanced accuracy : {fp32_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"Macro F1          : {fp32_metrics['macro_f1']:.6f}"
    )
    print(
        f"Macro AUC         : {fp32_metrics['macro_auc_ovr']:.6f}"
    )
    print(
        "Pre-Ictal recall  : "
        f"{fp32_metrics['class_metrics']['Pre-Ictal']['recall']:.6f}"
    )
    print(
        "Ictal recall      : "
        f"{fp32_metrics['class_metrics']['Ictal']['recall']:.6f}"
    )
    print(
        "False alarms/hour : "
        f"{fp32_metrics['alarm']['false_alarms_per_hour']:.6f}"
    )

    print("\nINT8 metrics")
    print("-" * 80)
    print(
        f"Accuracy          : {int8_metrics['accuracy']:.6f}"
    )
    print(
        f"Balanced accuracy : {int8_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"Macro F1          : {int8_metrics['macro_f1']:.6f}"
    )
    print(
        f"Macro AUC         : {int8_metrics['macro_auc_ovr']:.6f}"
    )
    print(
        "Pre-Ictal recall  : "
        f"{int8_metrics['class_metrics']['Pre-Ictal']['recall']:.6f}"
    )
    print(
        "Ictal recall      : "
        f"{int8_metrics['class_metrics']['Ictal']['recall']:.6f}"
    )
    print(
        "False alarms/hour : "
        f"{int8_metrics['alarm']['false_alarms_per_hour']:.6f}"
    )

    print("\nDeployment comparison")
    print("-" * 80)
    print(
        f"Total test windows       : {total_windows}"
    )
    print(
        f"Prediction disagreements : "
        f"{prediction_disagreements}"
    )
    print(
        f"Alarm disagreements      : "
        f"{alarm_disagreements}"
    )
    print(
        f"Maximum probability diff : "
        f"{maximum_probability_difference:.8f}"
    )
    print(
        f"Macro F1 change          : "
        f"{macro_f1_change:+.6f}"
    )
    print(
        f"Pre-Ictal recall change  : "
        f"{pre_ictal_recall_change:+.6f}"
    )
    print(
        f"Ictal recall change      : "
        f"{ictal_recall_change:+.6f}"
    )
    print(
        f"Size reduction           : "
        f"{size_reduction_percent:.2f}%"
    )
    print(
        f"FP32 mean/window latency : "
        f"{fp32_mean_window_latency_ms:.6f} ms"
    )
    print(
        f"INT8 mean/window latency : "
        f"{int8_mean_window_latency_ms:.6f} ms"
    )
    print(
        f"Latency change           : "
        f"{latency_change_percent:+.2f}%"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fp32_confusion_path = (
        report_directory
        / "fp32_test_confusion_matrix.csv"
    )

    int8_confusion_path = (
        report_directory
        / "int8_test_confusion_matrix.csv"
    )

    save_confusion_matrix_csv(
        fp32_metrics["confusion_matrix"],
        fp32_confusion_path,
    )

    save_confusion_matrix_csv(
        int8_metrics["confusion_matrix"],
        int8_confusion_path,
    )

    comparison_csv_path = (
        report_directory
        / "fp32_vs_int8_test_metrics.csv"
    )

    comparison_rows = [
        {
            "model": "FP32_ONNX",
            "accuracy": fp32_metrics["accuracy"],
            "balanced_accuracy": (
                fp32_metrics["balanced_accuracy"]
            ),
            "macro_precision": (
                fp32_metrics["macro_precision"]
            ),
            "macro_recall": (
                fp32_metrics["macro_recall"]
            ),
            "macro_specificity": (
                fp32_metrics["macro_specificity"]
            ),
            "macro_f1": fp32_metrics["macro_f1"],
            "macro_auc_ovr": (
                fp32_metrics["macro_auc_ovr"]
            ),
            "pre_ictal_recall": (
                fp32_metrics["class_metrics"][
                    "Pre-Ictal"
                ]["recall"]
            ),
            "ictal_recall": (
                fp32_metrics["class_metrics"][
                    "Ictal"
                ]["recall"]
            ),
            "false_alarms_per_hour": (
                fp32_metrics["alarm"][
                    "false_alarms_per_hour"
                ]
            ),
            "model_size_bytes": fp32_size_bytes,
            "mean_window_latency_ms": (
                fp32_mean_window_latency_ms
            ),
        },
        {
            "model": "INT8_QDQ_ONNX",
            "accuracy": int8_metrics["accuracy"],
            "balanced_accuracy": (
                int8_metrics["balanced_accuracy"]
            ),
            "macro_precision": (
                int8_metrics["macro_precision"]
            ),
            "macro_recall": (
                int8_metrics["macro_recall"]
            ),
            "macro_specificity": (
                int8_metrics["macro_specificity"]
            ),
            "macro_f1": int8_metrics["macro_f1"],
            "macro_auc_ovr": (
                int8_metrics["macro_auc_ovr"]
            ),
            "pre_ictal_recall": (
                int8_metrics["class_metrics"][
                    "Pre-Ictal"
                ]["recall"]
            ),
            "ictal_recall": (
                int8_metrics["class_metrics"][
                    "Ictal"
                ]["recall"]
            ),
            "false_alarms_per_hour": (
                int8_metrics["alarm"][
                    "false_alarms_per_hour"
                ]
            ),
            "model_size_bytes": int8_size_bytes,
            "mean_window_latency_ms": (
                int8_mean_window_latency_ms
            ),
        },
    ]

    with comparison_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                comparison_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )

    report = {
        "status": "pass",
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "scientific_status": SCIENTIFIC_STATUS,
        "evaluation_scope": (
            "Complete frozen held-out patient test split"
        ),
        "threshold_selection": {
            "source": "frozen validation-selected checkpoint value",
            "threshold": alarm_threshold,
            "policy": checkpoint[
                "alarm_threshold_policy"
            ],
            "test_tuning_performed": False,
        },
        "patient_split": {
            "training": training_patient_ids,
            "validation": validation_patient_ids,
            "test": test_patient_ids,
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
            },
            "fp32": {
                "path": str(fp32_path),
                "sha256": fp32_hash,
                "size_bytes": fp32_size_bytes,
                "graph": fp32_graph,
            },
            "int8": {
                "path": str(int8_path),
                "sha256": int8_hash,
                "size_bytes": int8_size_bytes,
                "graph": int8_graph,
            },
        },
        "dataset": {
            "total_test_windows": total_windows,
            "window_step_seconds": WINDOW_STEP_SECONDS,
            "patient_records": patient_records,
        },
        "fp32_metrics": serializable_metrics(
            fp32_metrics
        ),
        "int8_metrics": serializable_metrics(
            int8_metrics
        ),
        "comparison": {
            "prediction_disagreements": (
                prediction_disagreements
            ),
            "prediction_disagreement_rate": float(
                prediction_disagreements
                / total_windows
            ),
            "alarm_disagreements": (
                alarm_disagreements
            ),
            "alarm_disagreement_rate": float(
                alarm_disagreements
                / total_windows
            ),
            "maximum_probability_difference": (
                maximum_probability_difference
            ),
            "mean_probability_difference": (
                mean_probability_difference
            ),
            "macro_f1_change": macro_f1_change,
            "pre_ictal_recall_change": (
                pre_ictal_recall_change
            ),
            "ictal_recall_change": (
                ictal_recall_change
            ),
            "size_reduction_percent": (
                size_reduction_percent
            ),
            "fp32_total_inference_ms": (
                total_fp32_latency_ms
            ),
            "int8_total_inference_ms": (
                total_int8_latency_ms
            ),
            "fp32_mean_window_latency_ms": (
                fp32_mean_window_latency_ms
            ),
            "int8_mean_window_latency_ms": (
                int8_mean_window_latency_ms
            ),
            "latency_change_percent": (
                latency_change_percent
            ),
        },
        "engineering_decision_support": {
            "maximum_acceptable_absolute_macro_f1_drop": (
                args.maximum_acceptable_f1_drop
            ),
            "macro_f1_retained_by_criterion": (
                retained_by_engineering_criterion
            ),
            "note": (
                "This threshold is an engineering decision-support "
                "criterion for this experiment, not a universal "
                "scientific standard."
            ),
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "pytorch": torch.__version__,
        },
    }

    report_path = (
        report_directory
        / "fp32_vs_int8_full_test_report.json"
    )

    summary_path = (
        report_directory
        / "fp32_vs_int8_full_test_summary.txt"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    recommendation = (
        "INT8 RETAINED BY CURRENT ENGINEERING CRITERION"
        if retained_by_engineering_criterion
        else "INT8 DEGRADED BEYOND CURRENT ENGINEERING CRITERION"
    )

    summary_lines = [
        "FP32 vs INT8 Full Held-Out Test Evaluation",
        "=" * 72,
        "",
        f"Status                         : PASS",
        f"Test patients                  : {', '.join(test_patient_ids)}",
        f"Total test windows             : {total_windows}",
        f"Alarm threshold                : {alarm_threshold:.6f}",
        f"Test threshold tuning          : NO",
        "",
        f"FP32 accuracy                  : {fp32_metrics['accuracy']:.6f}",
        f"INT8 accuracy                  : {int8_metrics['accuracy']:.6f}",
        f"FP32 macro F1                  : {fp32_metrics['macro_f1']:.6f}",
        f"INT8 macro F1                  : {int8_metrics['macro_f1']:.6f}",
        f"Macro F1 change                : {macro_f1_change:+.6f}",
        (
            "FP32 pre-ictal recall          : "
            f"{fp32_metrics['class_metrics']['Pre-Ictal']['recall']:.6f}"
        ),
        (
            "INT8 pre-ictal recall          : "
            f"{int8_metrics['class_metrics']['Pre-Ictal']['recall']:.6f}"
        ),
        (
            "FP32 ictal recall              : "
            f"{fp32_metrics['class_metrics']['Ictal']['recall']:.6f}"
        ),
        (
            "INT8 ictal recall              : "
            f"{int8_metrics['class_metrics']['Ictal']['recall']:.6f}"
        ),
        (
            "FP32 false alarms/hour         : "
            f"{fp32_metrics['alarm']['false_alarms_per_hour']:.6f}"
        ),
        (
            "INT8 false alarms/hour         : "
            f"{int8_metrics['alarm']['false_alarms_per_hour']:.6f}"
        ),
        "",
        f"Prediction disagreements       : {prediction_disagreements}",
        f"Alarm disagreements            : {alarm_disagreements}",
        (
            "Maximum probability difference : "
            f"{maximum_probability_difference:.8f}"
        ),
        f"Size reduction                 : {size_reduction_percent:.2f}%",
        (
            "FP32 mean latency/window       : "
            f"{fp32_mean_window_latency_ms:.6f} ms"
        ),
        (
            "INT8 mean latency/window       : "
            f"{int8_mean_window_latency_ms:.6f} ms"
        ),
        f"Latency change                 : {latency_change_percent:+.2f}%",
        "",
        f"Decision support               : {recommendation}",
        "",
        (
            "Scientific status              : Research prototype only; "
            "not clinically validated; not a medical device."
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("FULL FP32 VS INT8 TEST EVALUATION: PASS")
    print("=" * 80)
    print(f"Decision support : {recommendation}")
    print(f"JSON report      : {report_path}")
    print(f"Metrics CSV      : {comparison_csv_path}")
    print(f"FP32 confusion   : {fp32_confusion_path}")
    print(f"INT8 confusion   : {int8_confusion_path}")
    print(f"Summary          : {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())