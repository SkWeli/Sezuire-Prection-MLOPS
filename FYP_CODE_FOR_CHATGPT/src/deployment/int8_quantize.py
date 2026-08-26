#!/usr/bin/env python3
"""
Create a statically quantized INT8 ONNX version of the frozen TCN.

The quantizer uses representative EEG windows from training patients only.

Pipeline:

1. Verify the frozen FP32 ONNX artifact.
2. Read the frozen patient split from the PyTorch checkpoint.
3. Confirm calibration patients belong only to the training split.
4. Load deterministic representative windows.
5. Apply the same z-score normalization used during training/inference.
6. Run static calibration.
7. Create an INT8 QDQ ONNX model.
8. Validate the resulting graph.
9. Verify CPU ONNX Runtime loading.
10. Save a detailed reproducibility manifest.

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "frozen"
    / "seizure_tcn_p20_baseline_review_0d6774d3.pt"
)

DEFAULT_FP32_ONNX_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_fp32.onnx"
)

DEFAULT_INT8_ONNX_PATH = (
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
)


# ---------------------------------------------------------------------------
# Frozen artifact identity
# ---------------------------------------------------------------------------

EXPECTED_CHECKPOINT_SHA256 = (
    "0d6774d36f2f040b4ce6ae5f9964fc30"
    "b004a7fe96b4a9fdfd401f733921a4e7"
)

EXPECTED_FP32_ONNX_SHA256 = (
    "af31d5a99ac683786b70abc4eea774d9"
    "c3b9564af41856358280060cd2f77420"
)

EXPECTED_CHANNELS = 20
EXPECTED_TIMEPOINTS = 512
EXPECTED_CLASSES = 3

INPUT_NAME = "eeg_input"
OUTPUT_NAME = "logits"

SCIENTIFIC_STATUS = (
    "Research prototype only; not clinically validated; "
    "not a medical device."
)


def sha256_file(path: Path) -> str:
    """
    Calculate a file's SHA256 digest.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_hash(
    path: Path,
    expected_hash: str,
    description: str,
) -> str:
    """
    Verify that a model artifact is the intended frozen file.
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
            f"{description} SHA256 mismatch."
        )

    print(f"{description} verification: PASS")

    return actual_hash


def load_frozen_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Load the frozen PyTorch checkpoint and verify its split metadata.
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
            "Expected the frozen checkpoint to be a dictionary."
        )

    required_keys = {
        "model_name",
        "split_patient_ids",
        "patient_count",
        "n_channels",
        "n_timepoints",
        "n_classes",
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
            "The checkpoint is not a SeizureTCN checkpoint."
        )

    if int(checkpoint["patient_count"]) != 20:
        raise RuntimeError(
            "Expected the frozen 20-patient checkpoint."
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

    split_patient_ids = checkpoint["split_patient_ids"]

    required_splits = {
        "train",
        "validation",
        "test",
    }

    if set(split_patient_ids.keys()) != required_splits:
        raise RuntimeError(
            "Unexpected patient split keys. "
            f"Received: {sorted(split_patient_ids.keys())}"
        )

    return checkpoint


def normalize_windows(
    windows: np.ndarray,
) -> np.ndarray:
    """
    Apply per-window, per-channel z-score normalization.

    Input:
        [batch, 20, 512]

    Output:
        [batch, 1, 20, 512]
    """
    if windows.ndim != 3:
        raise ValueError(
            "Expected windows shaped [batch, channels, samples]."
        )

    if tuple(windows.shape[1:]) != (
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
    ):
        raise ValueError(
            f"Unexpected calibration input shape: {windows.shape}"
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
            "Calibration input contains NaN or infinity."
        )

    return normalized


def choose_deterministic_indices(
    labels: np.ndarray,
    maximum_windows: int,
    random_seed: int,
) -> np.ndarray:
    """
    Select deterministic calibration windows.

    The function attempts to include all available classes while remaining
    bounded by maximum_windows.

    Calibration is not evaluation, but including multiple signal states
    makes the representative range more useful than taking only the first
    windows from each file.
    """
    if maximum_windows <= 0:
        raise ValueError(
            "maximum_windows must be greater than zero."
        )

    rng = np.random.default_rng(
        random_seed
    )

    selected_indices: list[int] = []

    unique_classes = sorted(
        int(value)
        for value in np.unique(labels)
    )

    windows_per_class = max(
        1,
        maximum_windows // max(
            len(unique_classes),
            1,
        ),
    )

    for class_index in unique_classes:
        class_indices = np.flatnonzero(
            labels == class_index
        )

        if len(class_indices) == 0:
            continue

        take_count = min(
            windows_per_class,
            len(class_indices),
        )

        chosen = rng.choice(
            class_indices,
            size=take_count,
            replace=False,
        )

        selected_indices.extend(
            int(index)
            for index in chosen
        )

    # Fill any remaining calibration capacity from the full patient set.
    remaining_capacity = (
        maximum_windows - len(selected_indices)
    )

    if remaining_capacity > 0:
        all_indices = np.arange(
            len(labels),
            dtype=np.int64,
        )

        unselected = np.setdiff1d(
            all_indices,
            np.asarray(
                selected_indices,
                dtype=np.int64,
            ),
            assume_unique=False,
        )

        if len(unselected) > 0:
            take_count = min(
                remaining_capacity,
                len(unselected),
            )

            extra = rng.choice(
                unselected,
                size=take_count,
                replace=False,
            )

            selected_indices.extend(
                int(index)
                for index in extra
            )

    return np.asarray(
        sorted(selected_indices),
        dtype=np.int64,
    )


def load_calibration_windows(
    processed_data_directory: Path,
    training_patient_ids: list[str],
    windows_per_patient: int,
    random_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Load representative calibration data from training patients only.

    Each patient's NPZ is expected at:

        data/processed/tusz/<patient_id>/<patient_id>.npz
    """
    calibration_batches: list[np.ndarray] = []
    patient_records: list[dict[str, Any]] = []

    for patient_position, patient_id in enumerate(
        training_patient_ids
    ):
        npz_path = (
            processed_data_directory
            / patient_id
            / f"{patient_id}.npz"
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                "Calibration patient NPZ was not found: "
                f"{npz_path}"
            )

        with np.load(
            npz_path,
            allow_pickle=True,
        ) as data:
            if "epochs" not in data.files:
                raise RuntimeError(
                    f"Missing epochs in {npz_path}"
                )

            if "labels" not in data.files:
                raise RuntimeError(
                    f"Missing labels in {npz_path}"
                )

            epochs = np.asarray(
                data["epochs"],
                dtype=np.float32,
            )

            labels = np.asarray(
                data["labels"],
                dtype=np.int64,
            )

        if epochs.ndim != 3:
            raise RuntimeError(
                f"Unexpected epoch rank for {patient_id}: "
                f"{epochs.shape}"
            )

        if tuple(epochs.shape[1:]) != (
            EXPECTED_CHANNELS,
            EXPECTED_TIMEPOINTS,
        ):
            raise RuntimeError(
                f"Unexpected EEG shape for {patient_id}: "
                f"{epochs.shape}"
            )

        if len(epochs) != len(labels):
            raise RuntimeError(
                f"Epoch/label count mismatch for {patient_id}."
            )

        patient_seed = (
            random_seed + patient_position
        )

        selected_indices = choose_deterministic_indices(
            labels=labels,
            maximum_windows=windows_per_patient,
            random_seed=patient_seed,
        )

        selected_epochs = epochs[
            selected_indices
        ]

        normalized = normalize_windows(
            selected_epochs
        )

        calibration_batches.append(
            normalized
        )

        selected_labels = labels[
            selected_indices
        ]

        selected_counts = np.bincount(
            selected_labels,
            minlength=EXPECTED_CLASSES,
        )

        patient_records.append({
            "patient_id": patient_id,
            "npz_path": str(npz_path),
            "total_windows_available": int(
                len(epochs)
            ),
            "calibration_windows_selected": int(
                len(selected_indices)
            ),
            "selected_class_counts": {
                "Interictal": int(
                    selected_counts[0]
                ),
                "Pre-Ictal": int(
                    selected_counts[1]
                ),
                "Ictal": int(
                    selected_counts[2]
                ),
            },
            "selection_seed": int(
                patient_seed
            ),
            "selected_indices": (
                selected_indices.tolist()
            ),
        })

        print(
            f"{patient_id}: selected "
            f"{len(selected_indices)} / {len(epochs)} windows"
        )

    if not calibration_batches:
        raise RuntimeError(
            "No calibration windows were loaded."
        )

    combined_calibration_data = np.concatenate(
        calibration_batches,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        combined_calibration_data,
        patient_records,
    )


class EEGCalibrationDataReader(
    CalibrationDataReader
):
    """
    Provide representative EEG batches to ONNX Runtime calibration.

    ONNX Runtime repeatedly calls get_next() until it returns None.
    """

    def __init__(
        self,
        calibration_data: np.ndarray,
        batch_size: int,
    ) -> None:
        if calibration_data.ndim != 4:
            raise ValueError(
                "Calibration data must have shape "
                "[N, 1, 20, 512]."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self._iterator: Iterator[
            dict[str, np.ndarray]
        ] | None = None

        self.rewind()

    def _batch_generator(
        self,
    ) -> Iterator[dict[str, np.ndarray]]:
        for start_index in range(
            0,
            len(self.calibration_data),
            self.batch_size,
        ):
            batch = self.calibration_data[
                start_index:
                start_index + self.batch_size
            ]

            yield {
                INPUT_NAME: batch
            }

    def get_next(
        self,
    ) -> dict[str, np.ndarray] | None:
        """
        Return the next calibration batch.
        """
        if self._iterator is None:
            self.rewind()

        try:
            return next(self._iterator)
        except StopIteration:
            return None

    def rewind(self) -> None:
        """
        Reset the reader so calibration can begin again.
        """
        self._iterator = self._batch_generator()


def validate_fp32_onnx_contract(
    model_path: Path,
) -> dict[str, Any]:
    """
    Validate the FP32 model before quantization.
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
            f"Unexpected FP32 ONNX inputs: {graph_inputs}"
        )

    if graph_outputs != [OUTPUT_NAME]:
        raise RuntimeError(
            f"Unexpected FP32 ONNX outputs: {graph_outputs}"
        )

    default_opsets = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in {"", "ai.onnx"}
    ]

    if not default_opsets:
        raise RuntimeError(
            "FP32 ONNX model has no default opset."
        )

    return {
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "actual_opset": max(default_opsets),
        "ir_version": int(model.ir_version),
    }


def quantize_model(
    fp32_model_path: Path,
    int8_model_path: Path,
    calibration_reader: EEGCalibrationDataReader,
) -> None:
    """
    Create a static QDQ INT8 model.

    QDQ inserts QuantizeLinear and DequantizeLinear nodes around
    quantized tensors while preserving the original operator structure.
    """
    int8_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quantize_static(
        model_input=str(fp32_model_path),
        model_output=str(int8_model_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=[
            "Conv",
            "MatMul",
            "Gemm",
        ],
        per_channel=True,
        reduce_range=False,
        extra_options={
            # Keep symmetrical activation and weight ranges for the first
            # controlled experiment.
            "ActivationSymmetric": True,
            "WeightSymmetric": True,

            # Preserve QDQ pairs for predictable model inspection.
            "DedicatedQDQPair": False,
        },
    )

    if not int8_model_path.exists():
        raise RuntimeError(
            "INT8 quantization completed without creating a model."
        )

    if int8_model_path.stat().st_size == 0:
        raise RuntimeError(
            "The generated INT8 model is empty."
        )


def validate_int8_model(
    int8_model_path: Path,
) -> dict[str, Any]:
    """
    Validate the quantized ONNX graph and CPU runtime compatibility.
    """
    model = onnx.load(
        str(int8_model_path)
    )

    onnx.checker.check_model(
        model,
        full_check=True,
    )

    node_types = [
        node.op_type
        for node in model.graph.node
    ]

    quantize_node_count = node_types.count(
        "QuantizeLinear"
    )

    dequantize_node_count = node_types.count(
        "DequantizeLinear"
    )

    if quantize_node_count == 0:
        raise RuntimeError(
            "The generated model contains no QuantizeLinear nodes."
        )

    if dequantize_node_count == 0:
        raise RuntimeError(
            "The generated model contains no DequantizeLinear nodes."
        )

    session = ort.InferenceSession(
        str(int8_model_path),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    active_providers = session.get_providers()

    if active_providers != [
        "CPUExecutionProvider"
    ]:
        raise RuntimeError(
            f"Unexpected INT8 execution providers: "
            f"{active_providers}"
        )

    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    if len(session_inputs) != 1:
        raise RuntimeError(
            "Expected one INT8 model input."
        )

    if len(session_outputs) != 1:
        raise RuntimeError(
            "Expected one INT8 model output."
        )

    if session_inputs[0].name != INPUT_NAME:
        raise RuntimeError(
            "Unexpected INT8 model input name."
        )

    if session_outputs[0].name != OUTPUT_NAME:
        raise RuntimeError(
            "Unexpected INT8 model output name."
        )

    return {
        "quantize_linear_nodes": int(
            quantize_node_count
        ),
        "dequantize_linear_nodes": int(
            dequantize_node_count
        ),
        "total_nodes": int(
            len(node_types)
        ),
        "providers": active_providers,
        "input_name": session_inputs[0].name,
        "input_shape": session_inputs[0].shape,
        "output_name": session_outputs[0].name,
        "output_shape": session_outputs[0].shape,
    }


def parse_arguments() -> argparse.Namespace:
    """
    Parse quantization settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Statically quantize the verified FP32 TCN ONNX model "
            "using training-patient EEG calibration data."
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
        default=DEFAULT_FP32_ONNX_PATH,
    )

    parser.add_argument(
        "--int8-model",
        type=Path,
        default=DEFAULT_INT8_ONNX_PATH,
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
        "--windows-per-patient",
        type=int,
        default=30,
        help=(
            "Maximum calibration windows selected from each "
            "training patient."
        ),
    )

    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> int:
    """
    Execute static INT8 QDQ quantization.
    """
    args = parse_arguments()

    checkpoint_path = args.checkpoint.resolve()
    fp32_model_path = args.fp32_model.resolve()
    int8_model_path = args.int8_model.resolve()
    processed_data_directory = (
        args.processed_data_dir.resolve()
    )
    report_directory = (
        args.report_dir.resolve()
    )

    print("=" * 80)
    print("STATIC INT8 QDQ QUANTIZATION - FROZEN TCN")
    print("=" * 80)
    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print("Calibration source: training patients only")
    print("Execution provider: CPUExecutionProvider")
    print()

    checkpoint_hash = verify_hash(
        checkpoint_path,
        EXPECTED_CHECKPOINT_SHA256,
        "Checkpoint",
    )

    fp32_hash = verify_hash(
        fp32_model_path,
        EXPECTED_FP32_ONNX_SHA256,
        "FP32 ONNX",
    )

    checkpoint = load_frozen_checkpoint(
        checkpoint_path
    )

    split_patient_ids = checkpoint[
        "split_patient_ids"
    ]

    training_patient_ids = list(
        split_patient_ids["train"]
    )

    validation_patient_ids = set(
        split_patient_ids["validation"]
    )

    test_patient_ids = set(
        split_patient_ids["test"]
    )

    if len(training_patient_ids) != 14:
        raise RuntimeError(
            "Expected 14 frozen training patients."
        )

    if set(training_patient_ids) & validation_patient_ids:
        raise RuntimeError(
            "Training and validation patient sets overlap."
        )

    if set(training_patient_ids) & test_patient_ids:
        raise RuntimeError(
            "Training and test patient sets overlap."
        )

    print(
        f"Training patients : "
        f"{len(training_patient_ids)}"
    )
    print(
        f"Validation patients: "
        f"{len(validation_patient_ids)}"
    )
    print(
        f"Test patients      : "
        f"{len(test_patient_ids)}"
    )
    print("Patient leakage check: PASS")

    fp32_contract = validate_fp32_onnx_contract(
        fp32_model_path
    )

    print(
        f"FP32 ONNX opset  : "
        f"{fp32_contract['actual_opset']}"
    )
    print("FP32 graph check : PASS")

    print("\nLoading calibration windows")
    print("-" * 80)

    (
        calibration_data,
        patient_records,
    ) = load_calibration_windows(
        processed_data_directory=(
            processed_data_directory
        ),
        training_patient_ids=(
            training_patient_ids
        ),
        windows_per_patient=(
            args.windows_per_patient
        ),
        random_seed=args.seed,
    )

    print(
        f"\nCalibration tensor: "
        f"{calibration_data.shape}"
    )
    print(
        f"Calibration windows: "
        f"{len(calibration_data)}"
    )

    calibration_reader = EEGCalibrationDataReader(
        calibration_data=calibration_data,
        batch_size=args.calibration_batch_size,
    )

    print("\nQuantizing model")
    print("-" * 80)
    print("Quantization format : QDQ")
    print("Activation type     : QInt8")
    print("Weight type         : QInt8")
    print("Calibration method  : MinMax")
    print("Per-channel weights : True")

    quantize_model(
        fp32_model_path=fp32_model_path,
        int8_model_path=int8_model_path,
        calibration_reader=calibration_reader,
    )

    int8_information = validate_int8_model(
        int8_model_path
    )

    int8_hash = sha256_file(
        int8_model_path
    )

    fp32_size_bytes = (
        fp32_model_path.stat().st_size
    )

    int8_size_bytes = (
        int8_model_path.stat().st_size
    )

    size_reduction_percent = (
        1.0
        - (
            int8_size_bytes
            / fp32_size_bytes
        )
    ) * 100.0

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "status": "pass",
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "scientific_status": SCIENTIFIC_STATUS,
        "quantization": {
            "method": "static",
            "format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "calibration_method": "MinMax",
            "per_channel": True,
            "reduce_range": False,
            "operators_requested": [
                "Conv",
                "MatMul",
                "Gemm",
            ],
        },
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
        },
        "fp32_model": {
            "path": str(fp32_model_path),
            "sha256": fp32_hash,
            "size_bytes": int(
                fp32_size_bytes
            ),
        },
        "int8_model": {
            "path": str(int8_model_path),
            "sha256": int8_hash,
            "size_bytes": int(
                int8_size_bytes
            ),
            "size_reduction_percent": float(
                size_reduction_percent
            ),
            "graph": int8_information,
        },
        "calibration": {
            "source_split": "train",
            "training_patient_ids": (
                training_patient_ids
            ),
            "validation_patient_ids_excluded": sorted(
                validation_patient_ids
            ),
            "test_patient_ids_excluded": sorted(
                test_patient_ids
            ),
            "windows_per_patient_limit": int(
                args.windows_per_patient
            ),
            "batch_size": int(
                args.calibration_batch_size
            ),
            "seed": int(
                args.seed
            ),
            "total_windows": int(
                len(calibration_data)
            ),
            "input_shape": list(
                calibration_data.shape
            ),
            "patient_records": patient_records,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "pytorch": torch.__version__,
        },
    }

    manifest_path = (
        report_directory
        / "tcn_int8_qdq_quantization_manifest.json"
    )

    summary_path = (
        report_directory
        / "tcn_int8_qdq_quantization_summary.txt"
    )

    hash_path = (
        report_directory
        / "tcn_int8_qdq_sha256.txt"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "Static INT8 QDQ Quantization Summary",
        "=" * 72,
        "",
        "Status                    : PASS",
        "Model                     : SeizureTCN",
        "Calibration split         : Training patients only",
        (
            "Training patients        : "
            f"{len(training_patient_ids)}"
        ),
        (
            "Calibration windows       : "
            f"{len(calibration_data)}"
        ),
        (
            "Quantization format       : QDQ"
        ),
        (
            "Activation type           : QInt8"
        ),
        (
            "Weight type               : QInt8"
        ),
        (
            "Calibration method        : MinMax"
        ),
        (
            "QuantizeLinear nodes      : "
            f"{int8_information['quantize_linear_nodes']}"
        ),
        (
            "DequantizeLinear nodes    : "
            f"{int8_information['dequantize_linear_nodes']}"
        ),
        (
            "FP32 size KiB             : "
            f"{fp32_size_bytes / 1024.0:.2f}"
        ),
        (
            "INT8 size KiB             : "
            f"{int8_size_bytes / 1024.0:.2f}"
        ),
        (
            "Size reduction percent    : "
            f"{size_reduction_percent:.2f}"
        ),
        (
            "FP32 SHA256               : "
            f"{fp32_hash}"
        ),
        (
            "INT8 SHA256               : "
            f"{int8_hash}"
        ),
        "",
        (
            "Important: Successful quantization and graph validation "
            "do not prove retained predictive quality."
        ),
        (
            "FP32 versus INT8 parity and held-out test evaluation "
            "must be completed next."
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    hash_path.write_text(
        f"{int8_hash}  {int8_model_path.as_posix()}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("STATIC INT8 QDQ QUANTIZATION: PASS")
    print("=" * 80)
    print(f"INT8 model      : {int8_model_path}")
    print(f"INT8 SHA256     : {int8_hash}")
    print(
        f"FP32 size       : "
        f"{fp32_size_bytes / 1024.0:.2f} KiB"
    )
    print(
        f"INT8 size       : "
        f"{int8_size_bytes / 1024.0:.2f} KiB"
    )
    print(
        f"Size reduction  : "
        f"{size_reduction_percent:.2f}%"
    )
    print(
        "Quantize nodes  : "
        f"{int8_information['quantize_linear_nodes']}"
    )
    print(
        "Dequantize nodes: "
        f"{int8_information['dequantize_linear_nodes']}"
    )
    print(f"Manifest        : {manifest_path}")
    print(f"Summary         : {summary_path}")
    print(f"Hash evidence   : {hash_path}")
    print()
    print(
        "Next required step: compare FP32 and INT8 outputs, "
        "latency and decision behaviour."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())