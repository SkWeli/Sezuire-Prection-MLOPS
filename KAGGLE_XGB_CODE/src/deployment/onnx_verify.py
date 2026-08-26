#!/usr/bin/env python3
"""
Verify numerical parity between the frozen PyTorch SeizureTCN model
and the exported FP32 ONNX model.

This script uses real processed TUSZ EEG windows and confirms that:

1. PyTorch and ONNX produce numerically close logits.
2. Softmax probabilities are numerically close.
3. Predicted classes match.
4. Alarm probabilities match.
5. Alarm decisions match.
6. The ONNX model supports dynamic batch sizes.
7. CPU latency and artifact-size evidence are recorded.

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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn import SeizureTCN


EXPECTED_CHECKPOINT_SHA256 = (
    "0d6774d36f2f040b4ce6ae5f9964fc30"
    "b004a7fe96b4a9fdfd401f733921a4e7"
)

EXPECTED_ONNX_SHA256 = (
    "af31d5a99ac683786b70abc4eea774d9"
    "c3b9564af41856358280060cd2f77420"
)

EXPECTED_PARAMETER_COUNT = 76_643
EXPECTED_CHANNELS = 20
EXPECTED_TIMEPOINTS = 512
EXPECTED_CLASSES = 3

CLASS_NAMES = [
    "Interictal",
    "Pre-Ictal",
    "Ictal",
]

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "frozen"
    / "seizure_tcn_p20_baseline_review_0d6774d3.pt"
)

DEFAULT_ONNX_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_fp32.onnx"
)

DEFAULT_PATIENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tusz"
    / "aaaaaayf"
    / "aaaaaayf.npz"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "onnx"
)


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_file_hash(
    path: Path,
    expected_hash: str,
    description: str,
) -> str:
    """
    Verify that a model artifact is exactly the intended frozen file.
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

    return actual_hash


def load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Load the frozen PyTorch checkpoint on CPU.
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

    required_keys = {
        "model_state_dict",
        "n_channels",
        "n_timepoints",
        "n_classes",
        "alarm_decision_threshold",
        "alarm_threshold_policy",
        "patient_count",
        "best_epoch",
        "best_val_multiclass_f1",
    }

    missing_keys = sorted(
        required_keys - set(checkpoint.keys())
    )

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint metadata missing: {missing_keys}"
        )

    return checkpoint


def build_pytorch_model(
    checkpoint: dict[str, Any],
) -> SeizureTCN:
    """
    Reconstruct the exact frozen TCN architecture and load its weights.
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
            "Unexpected parameter count. "
            f"Expected {EXPECTED_PARAMETER_COUNT}, "
            f"received {parameter_count}."
        )

    return model


def validate_onnx_contract(
    onnx_path: Path,
) -> dict[str, Any]:
    """
    Validate the saved ONNX graph and inspect its declared contract.
    """
    onnx_model = onnx.load(str(onnx_path))

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
            "No default ONNX opset declaration was found."
        )

    actual_opset = max(default_opsets)

    graph_inputs = [
        value.name
        for value in onnx_model.graph.input
    ]

    graph_outputs = [
        value.name
        for value in onnx_model.graph.output
    ]

    if graph_inputs != ["eeg_input"]:
        raise RuntimeError(
            f"Unexpected ONNX inputs: {graph_inputs}"
        )

    if graph_outputs != ["logits"]:
        raise RuntimeError(
            f"Unexpected ONNX outputs: {graph_outputs}"
        )

    print(f"Actual ONNX opset : {actual_opset}")
    print(f"ONNX inputs       : {graph_inputs}")
    print(f"ONNX outputs      : {graph_outputs}")
    print("ONNX graph check  : PASS")

    return {
        "actual_opset": actual_opset,
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "ir_version": int(onnx_model.ir_version),
    }


def load_real_eeg_windows(
    patient_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Load three real EEG windows from the held-out patient.

    The selected examples correspond to:

    - index 0: Interictal
    - index 2556: Pre-Ictal
    - index 2542: Ictal

    The NPZ contains an object-array field for channel names, therefore
    allow_pickle=True is required for this trusted locally generated file.
    """
    if not patient_path.exists():
        raise FileNotFoundError(
            f"Patient NPZ was not found: {patient_path}"
        )

    selected_indices = [0, 2556, 2542]

    with np.load(
        patient_path,
        allow_pickle=True,
    ) as data:
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
            f"Expected epochs shape [N, 20, 512], "
            f"received {epochs.shape}."
        )

    if tuple(epochs.shape[1:]) != (
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
    ):
        raise RuntimeError(
            f"Unexpected EEG input shape: {epochs.shape}"
        )

    for index in selected_indices:
        if index >= len(epochs):
            raise IndexError(
                f"Selected window index {index} is unavailable."
            )

    selected_epochs = epochs[selected_indices]
    selected_labels = labels[selected_indices]

    return (
        selected_epochs,
        selected_labels,
        selected_indices,
    )


def normalize_windows(
    windows: np.ndarray,
) -> np.ndarray:
    """
    Apply per-window, per-channel z-score normalization.

    For each channel independently:

        normalized = (signal - mean) / standard deviation

    This matches the preprocessing used by the current prototype.
    """
    means = windows.mean(
        axis=-1,
        keepdims=True,
    )

    standard_deviations = windows.std(
        axis=-1,
        keepdims=True,
    )

    # Prevent division by zero for any constant channel.
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

    # Add the single feature-map dimension required by the model:
    # [batch, channels, samples] -> [batch, 1, channels, samples]
    normalized = normalized[:, np.newaxis, :, :]

    if not np.isfinite(normalized).all():
        raise RuntimeError(
            "Normalized EEG input contains NaN or infinity."
        )

    return normalized


def softmax_numpy(
    logits: np.ndarray,
) -> np.ndarray:
    """
    Compute numerically stable softmax probabilities.
    """
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(shifted)

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )


def create_onnx_session(
    onnx_path: Path,
) -> ort.InferenceSession:
    """
    Create a CPU-only ONNX Runtime session.

    CPUExecutionProvider is selected explicitly so AzureExecutionProvider
    does not influence the local reproducibility benchmark.
    """
    available_providers = (
        ort.get_available_providers()
    )

    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CPUExecutionProvider is unavailable."
        )

    session_options = ort.SessionOptions()

    # Keep the benchmark deterministic and simple.
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
            f"Unexpected ONNX Runtime providers: {active_providers}"
        )

    print(f"ORT providers     : {active_providers}")

    return session


def run_pytorch(
    model: SeizureTCN,
    normalized_batch: np.ndarray,
) -> np.ndarray:
    """
    Run PyTorch inference and return logits as a NumPy array.
    """
    tensor = torch.from_numpy(
        normalized_batch
    )

    with torch.inference_mode():
        logits = model(tensor)

    return logits.cpu().numpy()


def run_onnx(
    session: ort.InferenceSession,
    normalized_batch: np.ndarray,
) -> np.ndarray:
    """
    Run ONNX Runtime inference and return logits.
    """
    outputs = session.run(
        ["logits"],
        {
            "eeg_input": normalized_batch,
        },
    )

    return np.asarray(
        outputs[0],
        dtype=np.float32,
    )


def compare_outputs(
    pytorch_logits: np.ndarray,
    onnx_logits: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    alarm_threshold: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """
    Compare numerical outputs and decision-level behaviour.
    """
    if pytorch_logits.shape != onnx_logits.shape:
        raise RuntimeError(
            "PyTorch and ONNX output shapes differ."
        )

    pytorch_probabilities = softmax_numpy(
        pytorch_logits
    )

    onnx_probabilities = softmax_numpy(
        onnx_logits
    )

    logit_absolute_difference = np.abs(
        pytorch_logits - onnx_logits
    )

    probability_absolute_difference = np.abs(
        pytorch_probabilities
        - onnx_probabilities
    )

    logits_close = np.allclose(
        pytorch_logits,
        onnx_logits,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    probabilities_close = np.allclose(
        pytorch_probabilities,
        onnx_probabilities,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    pytorch_predictions = np.argmax(
        pytorch_probabilities,
        axis=1,
    )

    onnx_predictions = np.argmax(
        onnx_probabilities,
        axis=1,
    )

    prediction_match = np.array_equal(
        pytorch_predictions,
        onnx_predictions,
    )

    # Alarm probability is defined as:
    #
    #     P(Pre-Ictal) + P(Ictal)
    #
    # This is separate from the three-class argmax prediction.
    pytorch_alarm_probabilities = (
        pytorch_probabilities[:, 1]
        + pytorch_probabilities[:, 2]
    )

    onnx_alarm_probabilities = (
        onnx_probabilities[:, 1]
        + onnx_probabilities[:, 2]
    )

    pytorch_alarm_decisions = (
        pytorch_alarm_probabilities
        >= alarm_threshold
    )

    onnx_alarm_decisions = (
        onnx_alarm_probabilities
        >= alarm_threshold
    )

    alarm_decision_match = np.array_equal(
        pytorch_alarm_decisions,
        onnx_alarm_decisions,
    )

    rows = []

    print("\nPer-window parity")
    print("-" * 80)

    for position, window_index in enumerate(indices):
        true_class = int(labels[position])
        pytorch_class = int(
            pytorch_predictions[position]
        )
        onnx_class = int(
            onnx_predictions[position]
        )

        row = {
            "window_index": int(window_index),
            "true_class": true_class,
            "true_class_name": CLASS_NAMES[true_class],
            "pytorch_prediction": pytorch_class,
            "pytorch_prediction_name": (
                CLASS_NAMES[pytorch_class]
            ),
            "onnx_prediction": onnx_class,
            "onnx_prediction_name": (
                CLASS_NAMES[onnx_class]
            ),
            "prediction_match": (
                pytorch_class == onnx_class
            ),
            "pytorch_probabilities": (
                pytorch_probabilities[position].tolist()
            ),
            "onnx_probabilities": (
                onnx_probabilities[position].tolist()
            ),
            "pytorch_alarm_probability": float(
                pytorch_alarm_probabilities[position]
            ),
            "onnx_alarm_probability": float(
                onnx_alarm_probabilities[position]
            ),
            "pytorch_alarm_decision": bool(
                pytorch_alarm_decisions[position]
            ),
            "onnx_alarm_decision": bool(
                onnx_alarm_decisions[position]
            ),
        }

        rows.append(row)

        print(
            f"Window {window_index}: "
            f"PyTorch={CLASS_NAMES[pytorch_class]}, "
            f"ONNX={CLASS_NAMES[onnx_class]}, "
            f"match={row['prediction_match']}"
        )

    return {
        "logits_close": bool(logits_close),
        "probabilities_close": bool(
            probabilities_close
        ),
        "predictions_match": bool(
            prediction_match
        ),
        "alarm_decisions_match": bool(
            alarm_decision_match
        ),
        "maximum_absolute_logit_difference": float(
            logit_absolute_difference.max()
        ),
        "mean_absolute_logit_difference": float(
            logit_absolute_difference.mean()
        ),
        "maximum_absolute_probability_difference": float(
            probability_absolute_difference.max()
        ),
        "mean_absolute_probability_difference": float(
            probability_absolute_difference.mean()
        ),
        "absolute_tolerance": float(
            absolute_tolerance
        ),
        "relative_tolerance": float(
            relative_tolerance
        ),
        "windows": rows,
    }


def benchmark_engine(
    inference_function,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    """
    Benchmark an already prepared inference function.

    Warm-up runs are excluded because first execution may include
    initialization overhead.
    """
    for _ in range(warmup_runs):
        inference_function()

    durations_ms = []

    for _ in range(timed_runs):
        start = time.perf_counter()
        inference_function()
        duration_ms = (
            time.perf_counter() - start
        ) * 1000.0
        durations_ms.append(duration_ms)

    durations = np.asarray(
        durations_ms,
        dtype=np.float64,
    )

    return {
        "runs": int(timed_runs),
        "mean_ms": float(durations.mean()),
        "median_ms": float(np.median(durations)),
        "minimum_ms": float(durations.min()),
        "maximum_ms": float(durations.max()),
        "standard_deviation_ms": float(
            durations.std()
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify numerical parity between the frozen "
            "PyTorch TCN and FP32 ONNX model."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX_PATH,
    )

    parser.add_argument(
        "--patient",
        type=Path,
        default=DEFAULT_PATIENT_PATH,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for numerical parity.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for numerical parity.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--timed-runs",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main() -> int:
    """Run complete PyTorch versus ONNX parity verification."""
    args = parse_arguments()

    checkpoint_path = args.checkpoint.resolve()
    onnx_path = args.onnx.resolve()
    patient_path = args.patient.resolve()
    report_directory = args.report_dir.resolve()

    print("=" * 80)
    print("PYTORCH VS FP32 ONNX PARITY VERIFICATION")
    print("=" * 80)
    print(
        "Scientific status: Research prototype only; "
        "not clinically validated; not a medical device."
    )
    print("Execution device: CPU")
    print()

    checkpoint_hash = verify_file_hash(
        checkpoint_path,
        EXPECTED_CHECKPOINT_SHA256,
        "Checkpoint",
    )

    onnx_hash = verify_file_hash(
        onnx_path,
        EXPECTED_ONNX_SHA256,
        "ONNX model",
    )

    onnx_contract = validate_onnx_contract(
        onnx_path
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    pytorch_model = build_pytorch_model(
        checkpoint
    )

    session = create_onnx_session(
        onnx_path
    )

    (
        real_windows,
        labels,
        selected_indices,
    ) = load_real_eeg_windows(
        patient_path
    )

    normalized_windows = normalize_windows(
        real_windows
    )

    print(
        f"Real EEG batch    : {normalized_windows.shape}"
    )

    alarm_threshold = float(
        checkpoint["alarm_decision_threshold"]
    )

    # ------------------------------------------------------------------
    # Test dynamic batch size 1.
    # ------------------------------------------------------------------
    single_window = normalized_windows[:1]

    pytorch_single = run_pytorch(
        pytorch_model,
        single_window,
    )

    onnx_single = run_onnx(
        session,
        single_window,
    )

    single_batch_result = compare_outputs(
        pytorch_logits=pytorch_single,
        onnx_logits=onnx_single,
        labels=labels[:1],
        indices=selected_indices[:1],
        alarm_threshold=alarm_threshold,
        absolute_tolerance=args.atol,
        relative_tolerance=args.rtol,
    )

    # ------------------------------------------------------------------
    # Test dynamic batch size 3.
    # ------------------------------------------------------------------
    pytorch_batch = run_pytorch(
        pytorch_model,
        normalized_windows,
    )

    onnx_batch = run_onnx(
        session,
        normalized_windows,
    )

    batch_result = compare_outputs(
        pytorch_logits=pytorch_batch,
        onnx_logits=onnx_batch,
        labels=labels,
        indices=selected_indices,
        alarm_threshold=alarm_threshold,
        absolute_tolerance=args.atol,
        relative_tolerance=args.rtol,
    )

    required_checks = {
        "single_batch_logits_close": (
            single_batch_result["logits_close"]
        ),
        "single_batch_probabilities_close": (
            single_batch_result[
                "probabilities_close"
            ]
        ),
        "single_batch_predictions_match": (
            single_batch_result[
                "predictions_match"
            ]
        ),
        "single_batch_alarm_decisions_match": (
            single_batch_result[
                "alarm_decisions_match"
            ]
        ),
        "batch_logits_close": (
            batch_result["logits_close"]
        ),
        "batch_probabilities_close": (
            batch_result["probabilities_close"]
        ),
        "batch_predictions_match": (
            batch_result["predictions_match"]
        ),
        "batch_alarm_decisions_match": (
            batch_result["alarm_decisions_match"]
        ),
    }

    print("\nParity checks")
    print("-" * 80)

    failed_checks = []

    for check_name, passed in required_checks.items():
        result = "PASS" if passed else "FAIL"
        print(f"{check_name:42s}: {result}")

        if not passed:
            failed_checks.append(check_name)

    # Benchmark one window to represent edge-style inference.
    benchmark_window = normalized_windows[:1]

    pytorch_benchmark = benchmark_engine(
        inference_function=lambda: run_pytorch(
            pytorch_model,
            benchmark_window,
        ),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    onnx_benchmark = benchmark_engine(
        inference_function=lambda: run_onnx(
            session,
            benchmark_window,
        ),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    print("\nCPU latency benchmark")
    print("-" * 80)
    print(
        "PyTorch mean       : "
        f"{pytorch_benchmark['mean_ms']:.3f} ms"
    )
    print(
        "ONNX Runtime mean  : "
        f"{onnx_benchmark['mean_ms']:.3f} ms"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "status": (
            "pass"
            if not failed_checks
            else "fail"
        ),
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "scientific_status": (
            "Research prototype only; not clinically validated; "
            "not a medical device."
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "checkpoint_sha256": checkpoint_hash,
        "onnx_path": str(
            onnx_path
        ),
        "onnx_sha256": onnx_hash,
        "onnx_size_bytes": int(
            onnx_path.stat().st_size
        ),
        "patient_path": str(
            patient_path
        ),
        "patient_id": "aaaaaayf",
        "selected_window_indices": (
            selected_indices
        ),
        "labels": labels.tolist(),
        "input_shape": list(
            normalized_windows.shape
        ),
        "alarm_threshold": alarm_threshold,
        "alarm_threshold_policy": str(
            checkpoint["alarm_threshold_policy"]
        ),
        "onnx_contract": onnx_contract,
        "single_batch_result": (
            single_batch_result
        ),
        "batch_result": batch_result,
        "required_checks": required_checks,
        "failed_checks": failed_checks,
        "latency": {
            "device": "CPU",
            "batch_size": 1,
            "pytorch": pytorch_benchmark,
            "onnx_runtime": onnx_benchmark,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }

    report_path = (
        report_directory
        / "tcn_fp32_onnx_parity_report.json"
    )

    summary_path = (
        report_directory
        / "tcn_fp32_onnx_parity_summary.txt"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "PyTorch vs FP32 ONNX Parity Summary",
        "=" * 72,
        "",
        (
            "Status                    : "
            f"{'PASS' if not failed_checks else 'FAIL'}"
        ),
        f"Checkpoint SHA256         : {checkpoint_hash}",
        f"ONNX SHA256               : {onnx_hash}",
        (
            "Actual ONNX opset         : "
            f"{onnx_contract['actual_opset']}"
        ),
        (
            "Maximum logit difference  : "
            f"{batch_result['maximum_absolute_logit_difference']:.10f}"
        ),
        (
            "Maximum probability diff  : "
            f"{batch_result['maximum_absolute_probability_difference']:.10f}"
        ),
        (
            "Predictions match         : "
            f"{batch_result['predictions_match']}"
        ),
        (
            "Alarm decisions match     : "
            f"{batch_result['alarm_decisions_match']}"
        ),
        (
            "PyTorch mean latency ms   : "
            f"{pytorch_benchmark['mean_ms']:.6f}"
        ),
        (
            "ONNX mean latency ms      : "
            f"{onnx_benchmark['mean_ms']:.6f}"
        ),
        "",
        (
            "Scientific status         : Research prototype only; "
            "not clinically validated; not a medical device."
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    if failed_checks:
        print("\n" + "=" * 80)
        print("PYTORCH VS ONNX PARITY: FAIL")
        print("=" * 80)
        print(f"Failed checks: {failed_checks}")
        print(f"Evidence report: {report_path}")
        return 1

    print("\n" + "=" * 80)
    print("PYTORCH VS ONNX PARITY: PASS")
    print("=" * 80)
    print(
        "Batch size 1 and batch size 3 are numerically consistent."
    )
    print(
        f"Maximum absolute logit difference: "
        f"{batch_result['maximum_absolute_logit_difference']:.10f}"
    )
    print(
        f"Maximum probability difference   : "
        f"{batch_result['maximum_absolute_probability_difference']:.10f}"
    )
    print(f"Evidence report: {report_path}")
    print(f"Summary report : {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())