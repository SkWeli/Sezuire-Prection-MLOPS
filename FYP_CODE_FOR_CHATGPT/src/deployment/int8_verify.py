#!/usr/bin/env python3
"""
Compare the verified FP32 ONNX TCN with its statically quantized INT8 model.

This script evaluates deployment behaviour using real held-out EEG windows.

It compares:

- model hashes and graph contracts
- logits and probabilities
- predicted classes
- alarm probabilities and alarm decisions
- CPU inference latency
- artifact size

This is an artifact-level and decision-level comparison. It is not yet the
complete held-out test-set evaluation.

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FP32_SHA256 = (
    "af31d5a99ac683786b70abc4eea774d9"
    "c3b9564af41856358280060cd2f77420"
)

EXPECTED_INT8_SHA256 = (
    "9fbac2f59b5acd0276036f8ab9ea65c6"
    "bc8555d71b5ba9c16124062889e43969"
)

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

SELECTED_WINDOW_INDICES = [
    0,
    2556,
    2542,
]

EXPECTED_WINDOW_LABELS = [
    0,
    1,
    2,
]

ALARM_THRESHOLD = 0.17

SCIENTIFIC_STATUS = (
    "Research prototype only; not clinically validated; "
    "not a medical device."
)

DEFAULT_FP32_MODEL = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_fp32.onnx"
)

DEFAULT_INT8_MODEL = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_int8_qdq.onnx"
)

DEFAULT_PATIENT_NPZ = (
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
    / "quantization"
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
    """Verify that a model exists and matches its frozen hash."""
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


def validate_onnx_model(
    model_path: Path,
    description: str,
) -> dict[str, Any]:
    """Validate an ONNX graph and inspect its input/output contract."""
    model = onnx.load(str(model_path))

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

    default_opsets = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in {"", "ai.onnx"}
    ]

    node_types = [
        node.op_type
        for node in model.graph.node
    ]

    information = {
        "actual_opset": max(default_opsets),
        "input_names": graph_inputs,
        "output_names": graph_outputs,
        "total_nodes": len(node_types),
        "quantize_linear_nodes": node_types.count(
            "QuantizeLinear"
        ),
        "dequantize_linear_nodes": node_types.count(
            "DequantizeLinear"
        ),
    }

    print(f"{description} graph check: PASS")

    return information


def create_cpu_session(
    model_path: Path,
) -> ort.InferenceSession:
    """
    Create a controlled CPU-only ONNX Runtime session.

    Both FP32 and INT8 models use identical session options so the latency
    comparison is as fair as possible.
    """
    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    if session.get_providers() != [
        "CPUExecutionProvider"
    ]:
        raise RuntimeError(
            "Unexpected ONNX Runtime execution provider."
        )

    return session


def load_held_out_windows(
    patient_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the same three held-out patient windows used by the offline demo.
    """
    if not patient_path.exists():
        raise FileNotFoundError(
            f"Held-out patient file was not found: {patient_path}"
        )

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

    selected_windows = epochs[
        SELECTED_WINDOW_INDICES
    ]

    selected_labels = labels[
        SELECTED_WINDOW_INDICES
    ]

    if selected_labels.tolist() != EXPECTED_WINDOW_LABELS:
        raise RuntimeError(
            "The selected held-out labels no longer match "
            f"{EXPECTED_WINDOW_LABELS}."
        )

    return (
        selected_windows,
        selected_labels,
    )


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
    if tuple(windows.shape[1:]) != (
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
    ):
        raise RuntimeError(
            f"Unexpected EEG shape: {windows.shape}"
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


def run_session(
    session: ort.InferenceSession,
    input_batch: np.ndarray,
) -> np.ndarray:
    """Run an ONNX Runtime session and return logits."""
    output = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: input_batch,
        },
    )[0]

    logits = np.asarray(
        output,
        dtype=np.float32,
    )

    expected_shape = (
        len(input_batch),
        EXPECTED_CLASSES,
    )

    if logits.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {logits.shape}"
        )

    if not np.isfinite(logits).all():
        raise RuntimeError(
            "Model output contains NaN or infinity."
        )

    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""
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


def build_decisions(
    logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert logits into class and alarm decisions."""
    probabilities = softmax(logits)

    predictions = np.argmax(
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

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "alarm_probabilities": alarm_probabilities,
        "alarm_decisions": alarm_decisions,
    }


def benchmark(
    inference_function: Callable[[], np.ndarray],
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    """
    Measure repeated inference latency.

    Warm-up calls are excluded from the reported timing.
    """
    for _ in range(warmup_runs):
        inference_function()

    timings_ms: list[float] = []

    for _ in range(timed_runs):
        start = time.perf_counter()

        inference_function()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        timings_ms.append(elapsed_ms)

    return {
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "mean_ms": float(
            statistics.fmean(timings_ms)
        ),
        "median_ms": float(
            statistics.median(timings_ms)
        ),
        "minimum_ms": float(
            min(timings_ms)
        ),
        "maximum_ms": float(
            max(timings_ms)
        ),
        "standard_deviation_ms": float(
            statistics.pstdev(timings_ms)
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse comparison settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare FP32 and static INT8 QDQ ONNX TCN models."
        )
    )

    parser.add_argument(
        "--fp32-model",
        type=Path,
        default=DEFAULT_FP32_MODEL,
    )

    parser.add_argument(
        "--int8-model",
        type=Path,
        default=DEFAULT_INT8_MODEL,
    )

    parser.add_argument(
        "--patient",
        type=Path,
        default=DEFAULT_PATIENT_NPZ,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--timed-runs",
        type=int,
        default=200,
    )

    return parser.parse_args()


def main() -> int:
    """Run FP32 versus INT8 comparison."""
    args = parse_arguments()

    fp32_path = args.fp32_model.resolve()
    int8_path = args.int8_model.resolve()
    patient_path = args.patient.resolve()
    report_directory = args.report_dir.resolve()

    print("=" * 80)
    print("FP32 VS INT8 ONNX DEPLOYMENT VERIFICATION")
    print("=" * 80)
    print(f"Scientific status: {SCIENTIFIC_STATUS}")
    print("Execution provider: CPUExecutionProvider")
    print()

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

    fp32_graph = validate_onnx_model(
        fp32_path,
        "FP32",
    )

    int8_graph = validate_onnx_model(
        int8_path,
        "INT8",
    )

    if int8_graph["quantize_linear_nodes"] == 0:
        raise RuntimeError(
            "INT8 model contains no QuantizeLinear nodes."
        )

    if int8_graph["dequantize_linear_nodes"] == 0:
        raise RuntimeError(
            "INT8 model contains no DequantizeLinear nodes."
        )

    fp32_session = create_cpu_session(
        fp32_path
    )

    int8_session = create_cpu_session(
        int8_path
    )

    windows, labels = load_held_out_windows(
        patient_path
    )

    normalized = normalize_windows(
        windows
    )

    fp32_logits = run_session(
        fp32_session,
        normalized,
    )

    int8_logits = run_session(
        int8_session,
        normalized,
    )

    fp32_decisions = build_decisions(
        fp32_logits
    )

    int8_decisions = build_decisions(
        int8_logits
    )

    logit_difference = np.abs(
        fp32_logits - int8_logits
    )

    probability_difference = np.abs(
        fp32_decisions["probabilities"]
        - int8_decisions["probabilities"]
    )

    prediction_matches = (
        fp32_decisions["predictions"]
        == int8_decisions["predictions"]
    )

    alarm_matches = (
        fp32_decisions["alarm_decisions"]
        == int8_decisions["alarm_decisions"]
    )

    rows: list[dict[str, Any]] = []

    print("\nDecision comparison")
    print("-" * 80)

    for position, window_index in enumerate(
        SELECTED_WINDOW_INDICES
    ):
        true_class = int(labels[position])

        fp32_class = int(
            fp32_decisions["predictions"][position]
        )

        int8_class = int(
            int8_decisions["predictions"][position]
        )

        row = {
            "window_index": int(window_index),
            "true_class": true_class,
            "true_class_name": CLASS_NAMES[true_class],
            "fp32_prediction": fp32_class,
            "fp32_prediction_name": CLASS_NAMES[
                fp32_class
            ],
            "int8_prediction": int8_class,
            "int8_prediction_name": CLASS_NAMES[
                int8_class
            ],
            "prediction_match": bool(
                prediction_matches[position]
            ),
            "fp32_probabilities": (
                fp32_decisions[
                    "probabilities"
                ][position].tolist()
            ),
            "int8_probabilities": (
                int8_decisions[
                    "probabilities"
                ][position].tolist()
            ),
            "fp32_alarm_probability": float(
                fp32_decisions[
                    "alarm_probabilities"
                ][position]
            ),
            "int8_alarm_probability": float(
                int8_decisions[
                    "alarm_probabilities"
                ][position]
            ),
            "fp32_alarm_decision": bool(
                fp32_decisions[
                    "alarm_decisions"
                ][position]
            ),
            "int8_alarm_decision": bool(
                int8_decisions[
                    "alarm_decisions"
                ][position]
            ),
            "alarm_decision_match": bool(
                alarm_matches[position]
            ),
        }

        rows.append(row)

        print(
            f"Window {window_index}: "
            f"FP32={row['fp32_prediction_name']}, "
            f"INT8={row['int8_prediction_name']}, "
            f"prediction_match={row['prediction_match']}, "
            f"alarm_match={row['alarm_decision_match']}"
        )

    benchmark_input = normalized[:1]

    fp32_latency = benchmark(
        inference_function=lambda: run_session(
            fp32_session,
            benchmark_input,
        ),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    int8_latency = benchmark(
        inference_function=lambda: run_session(
            int8_session,
            benchmark_input,
        ),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size

    size_reduction_percent = (
        1.0 - int8_size / fp32_size
    ) * 100.0

    latency_change_percent = (
        (
            int8_latency["mean_ms"]
            - fp32_latency["mean_ms"]
        )
        / fp32_latency["mean_ms"]
    ) * 100.0

    print("\nNumerical comparison")
    print("-" * 80)
    print(
        "Maximum absolute logit difference      : "
        f"{float(logit_difference.max()):.8f}"
    )
    print(
        "Maximum absolute probability difference: "
        f"{float(probability_difference.max()):.8f}"
    )
    print(
        "Predicted classes matching             : "
        f"{int(prediction_matches.sum())}/{len(prediction_matches)}"
    )
    print(
        "Alarm decisions matching               : "
        f"{int(alarm_matches.sum())}/{len(alarm_matches)}"
    )

    print("\nDeployment comparison")
    print("-" * 80)
    print(
        f"FP32 size       : {fp32_size / 1024.0:.2f} KiB"
    )
    print(
        f"INT8 size       : {int8_size / 1024.0:.2f} KiB"
    )
    print(
        f"Size reduction  : {size_reduction_percent:.2f}%"
    )
    print(
        "FP32 mean latency: "
        f"{fp32_latency['mean_ms']:.4f} ms"
    )
    print(
        "INT8 mean latency: "
        f"{int8_latency['mean_ms']:.4f} ms"
    )
    print(
        "Latency change   : "
        f"{latency_change_percent:+.2f}%"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "status": "pass",
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "scientific_status": SCIENTIFIC_STATUS,
        "patient_id": "aaaaaayf",
        "selected_window_indices": (
            SELECTED_WINDOW_INDICES
        ),
        "alarm_threshold": ALARM_THRESHOLD,
        "artifacts": {
            "fp32": {
                "path": str(fp32_path),
                "sha256": fp32_hash,
                "size_bytes": int(fp32_size),
                "graph": fp32_graph,
            },
            "int8": {
                "path": str(int8_path),
                "sha256": int8_hash,
                "size_bytes": int(int8_size),
                "graph": int8_graph,
            },
        },
        "comparison": {
            "maximum_absolute_logit_difference": float(
                logit_difference.max()
            ),
            "mean_absolute_logit_difference": float(
                logit_difference.mean()
            ),
            "maximum_absolute_probability_difference": float(
                probability_difference.max()
            ),
            "mean_absolute_probability_difference": float(
                probability_difference.mean()
            ),
            "predictions_matching": int(
                prediction_matches.sum()
            ),
            "predictions_total": int(
                len(prediction_matches)
            ),
            "alarm_decisions_matching": int(
                alarm_matches.sum()
            ),
            "alarm_decisions_total": int(
                len(alarm_matches)
            ),
            "windows": rows,
        },
        "deployment": {
            "size_reduction_percent": float(
                size_reduction_percent
            ),
            "fp32_latency": fp32_latency,
            "int8_latency": int8_latency,
            "latency_change_percent": float(
                latency_change_percent
            ),
        },
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "interpretation": (
            "This report compares artifact behaviour on three illustrative "
            "held-out windows. Full held-out test-set evaluation is still "
            "required before selecting the INT8 deployment candidate."
        ),
    }

    report_path = (
        report_directory
        / "tcn_fp32_vs_int8_verification.json"
    )

    summary_path = (
        report_directory
        / "tcn_fp32_vs_int8_summary.txt"
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
        "FP32 vs INT8 ONNX Verification Summary",
        "=" * 72,
        "",
        "Status                    : PASS",
        (
            "Predictions matching      : "
            f"{int(prediction_matches.sum())}/"
            f"{len(prediction_matches)}"
        ),
        (
            "Alarm decisions matching  : "
            f"{int(alarm_matches.sum())}/"
            f"{len(alarm_matches)}"
        ),
        (
            "Maximum logit difference  : "
            f"{float(logit_difference.max()):.8f}"
        ),
        (
            "Maximum probability diff  : "
            f"{float(probability_difference.max()):.8f}"
        ),
        (
            "FP32 size KiB             : "
            f"{fp32_size / 1024.0:.2f}"
        ),
        (
            "INT8 size KiB             : "
            f"{int8_size / 1024.0:.2f}"
        ),
        (
            "Size reduction percent    : "
            f"{size_reduction_percent:.2f}"
        ),
        (
            "FP32 mean latency ms      : "
            f"{fp32_latency['mean_ms']:.6f}"
        ),
        (
            "INT8 mean latency ms      : "
            f"{int8_latency['mean_ms']:.6f}"
        ),
        (
            "Latency change percent    : "
            f"{latency_change_percent:+.2f}"
        ),
        "",
        (
            "Full test-split evaluation is required before deciding "
            "whether the INT8 model should replace FP32."
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("FP32 VS INT8 ARTIFACT VERIFICATION: PASS")
    print("=" * 80)
    print(f"Evidence report: {report_path}")
    print(f"Summary report : {summary_path}")
    print()
    print(
        "Next required step: evaluate FP32 and INT8 across the complete "
        "frozen held-out test-patient split."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())