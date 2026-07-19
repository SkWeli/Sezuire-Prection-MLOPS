#!/usr/bin/env python3
"""
Offline research-prototype demonstration.

Pipeline:
    TTL metadata
        -> SHACL validation
        -> frozen TCN loading
        -> processed EEG loading
        -> three-class inference
        -> operational alarm decision

Scientific status:
    Research prototype only.
    Not clinically validated.
    Not a medical device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn import SeizureTCN


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*weight_norm.*",
)


EXPECTED_CHECKPOINT_SHA256 = (
    "0d6774d36f2f040b4ce6ae5f9964fc30"
    "b004a7fe96b4a9fdfd401f733921a4e7"
)

DEFAULT_PATIENT_ID = "aaaaaayf"

DEFAULT_INDICES = {
    "Interictal": 0,
    "Pre-Ictal": 2556,
    "Ictal": 2542,
}

EXPECTED_LABELS = {
    "Interictal": 0,
    "Pre-Ictal": 1,
    "Ictal": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def scalar_to_text(value) -> str:
    array = np.asarray(value)

    if array.size == 0:
        return ""

    item = array.reshape(-1)[0]

    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")

    return str(item)


def normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Reproduce training-time per-window, per-channel z-score normalization.

    Input:
        (channels, timepoints)

    Output:
        (1, 1, channels, timepoints)
    """
    tensor = torch.as_tensor(
        window,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)

    mean = tensor.mean(
        dim=-1,
        keepdim=True,
    )

    std = tensor.std(
        dim=-1,
        keepdim=True,
    ) + 1e-6

    return ((tensor - mean) / std).numpy()


def run_shacl_validation(ttl_path: Path) -> dict:
    validator_path = (
        PROJECT_ROOT
        / "src"
        / "validation"
        / "shacl_validator.py"
    )

    if not validator_path.exists():
        raise FileNotFoundError(
            f"SHACL validator not found: {validator_path}"
        )

    command = [
        sys.executable,
        str(validator_path),
        str(ttl_path),
    ]

    print("\n[1/4] SHACL SEMANTIC QUALITY GATE")
    print("-" * 80)
    print(f"TTL metadata : {ttl_path}")
    print(f"Validator    : {validator_path}")

    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    duration_ms = (
        time.perf_counter() - start
    ) * 1000.0

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())

    passed = result.returncode == 0

    print(
        "SHACL result : "
        + ("PASS" if passed else "FAIL")
    )

    print(
        f"Validation time: {duration_ms:.2f} ms"
    )

    return {
        "passed": bool(passed),
        "return_code": int(result.returncode),
        "duration_ms": float(duration_ms),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def load_frozen_model(checkpoint_path: Path):
    print("\n[2/4] FROZEN TCN VERIFICATION")
    print("-" * 80)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint_hash = sha256_file(checkpoint_path)

    print(f"Checkpoint : {checkpoint_path}")
    print(f"SHA256     : {checkpoint_hash}")

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint hash does not match the frozen P20 baseline."
        )

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
        "model_name",
        "model_state_dict",
        "n_channels",
        "n_timepoints",
        "n_classes",
        "patient_count",
        "best_epoch",
        "best_val_multiclass_f1",
        "alarm_decision_threshold",
    }

    missing_keys = sorted(
        required_keys - set(checkpoint)
    )

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {missing_keys}"
        )

    if checkpoint["model_name"] != "SeizureTCN":
        raise RuntimeError(
            "The supplied checkpoint is not a SeizureTCN checkpoint."
        )

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

    if parameter_count != 76643:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count}"
        )

    print(f"Model      : {checkpoint['model_name']}")
    print(f"Patients   : {checkpoint['patient_count']}")
    print(f"Best epoch : {checkpoint['best_epoch']}")
    print(
        "Best val F1: "
        f"{checkpoint['best_val_multiclass_f1']:.6f}"
    )
    print(
        "Input      : "
        f"{checkpoint['n_channels']} channels x "
        f"{checkpoint['n_timepoints']} samples"
    )
    print(f"Parameters : {parameter_count}")
    print(
        "Alarm rule : "
        f"{checkpoint.get('alarm_threshold_policy')}"
    )
    print(
        "Threshold  : "
        f"{checkpoint['alarm_decision_threshold']:.4f}"
    )
    print("Checkpoint verification: PASS")

    return model, checkpoint, checkpoint_hash


def load_patient_data(
    npz_path: Path,
    expected_patient_id: str,
):
    print("\n[3/4] PROCESSED EEG VERIFICATION")
    print("-" * 80)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"NPZ file not found: {npz_path}"
        )

    required_keys = {
        "epochs",
        "labels",
        "ch_names",
        "sfreq",
        "patient_id",
        "n_windows",
        "n_interictal",
        "n_pre_ictal",
        "n_ictal",
    }

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        missing_keys = sorted(
            required_keys - set(data.files)
        )

        if missing_keys:
            raise RuntimeError(
                f"NPZ is missing required keys: {missing_keys}"
            )

        epochs = np.asarray(data["epochs"])
        labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        ).reshape(-1)

        ch_names = [
            scalar_to_text(value)
            for value in np.asarray(
                data["ch_names"]
            ).reshape(-1)
        ]

        sfreq = float(
            np.asarray(data["sfreq"]).reshape(-1)[0]
        )

        stored_patient_id = scalar_to_text(
            data["patient_id"]
        )

        metadata = {
            "n_windows": int(
                np.asarray(
                    data["n_windows"]
                ).reshape(-1)[0]
            ),
            "n_interictal": int(
                np.asarray(
                    data["n_interictal"]
                ).reshape(-1)[0]
            ),
            "n_pre_ictal": int(
                np.asarray(
                    data["n_pre_ictal"]
                ).reshape(-1)[0]
            ),
            "n_ictal": int(
                np.asarray(
                    data["n_ictal"]
                ).reshape(-1)[0]
            ),
        }

    if epochs.ndim != 3:
        raise RuntimeError(
            f"Expected a 3D epochs array, received {epochs.shape}"
        )

    if epochs.shape[1:] != (20, 512):
        raise RuntimeError(
            "Expected EEG windows with shape "
            f"(20, 512), received {epochs.shape[1:]}"
        )

    if len(labels) != len(epochs):
        raise RuntimeError(
            "Epoch and label counts do not match."
        )

    if stored_patient_id != expected_patient_id:
        raise RuntimeError(
            "Patient ID mismatch: "
            f"expected={expected_patient_id}, "
            f"stored={stored_patient_id}"
        )

    if metadata["n_windows"] != len(epochs):
        raise RuntimeError(
            "NPZ n_windows metadata does not match the array."
        )

    actual_counts = np.bincount(
        labels,
        minlength=3,
    )

    expected_counts = np.asarray(
        [
            metadata["n_interictal"],
            metadata["n_pre_ictal"],
            metadata["n_ictal"],
        ]
    )

    if not np.array_equal(
        actual_counts,
        expected_counts,
    ):
        raise RuntimeError(
            "Stored class counts do not match the labels array."
        )

    print(f"NPZ file       : {npz_path}")
    print(f"Patient ID     : {stored_patient_id}")
    print(f"Epoch shape    : {epochs.shape}")
    print(f"Sampling rate  : {sfreq:.1f} Hz")
    print(f"Channel count  : {len(ch_names)}")
    print(
        "Class counts   : "
        f"Interictal={actual_counts[0]}, "
        f"Pre-Ictal={actual_counts[1]}, "
        f"Ictal={actual_counts[2]}"
    )
    print("Processed EEG verification: PASS")

    return (
        epochs,
        labels,
        ch_names,
        sfreq,
        metadata,
    )


def run_inference(
    model,
    checkpoint,
    epochs,
    labels,
):
    class_names = checkpoint.get(
        "class_names",
        ["Interictal", "Pre-Ictal", "Ictal"],
    )

    class_names = [
        str(name)
        for name in class_names
    ]

    if len(class_names) != 3:
        class_names = [
            "Interictal",
            "Pre-Ictal",
            "Ictal",
        ]

    alarm_threshold = float(
        checkpoint["alarm_decision_threshold"]
    )

    print("\n[4/4] OFFLINE TCN INFERENCE")
    print("-" * 80)
    print(
        "Note: These are illustrative test-patient windows, "
        "not a new performance evaluation."
    )

    prepared_windows = []
    selected_records = []

    for intended_name, index in DEFAULT_INDICES.items():
        if index < 0 or index >= len(epochs):
            raise IndexError(
                f"Window index is outside the dataset: {index}"
            )

        true_label = int(labels[index])
        expected_label = EXPECTED_LABELS[
            intended_name
        ]

        if true_label != expected_label:
            raise RuntimeError(
                f"Window {index} was expected to be "
                f"{intended_name} ({expected_label}), "
                f"but its true label is {true_label}."
            )

        normalized = normalize_window(
            epochs[index]
        )

        prepared_windows.append(
            normalized[0]
        )

        selected_records.append({
            "intended_class": intended_name,
            "window_index": int(index),
            "true_label": true_label,
        })

    batch = torch.as_tensor(
        np.stack(prepared_windows),
        dtype=torch.float32,
    )

    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    inference_ms = (
        time.perf_counter() - start
    ) * 1000.0

    probabilities_np = (
        probabilities.cpu().numpy()
    )

    predictions = np.argmax(
        probabilities_np,
        axis=1,
    )

    alarm_probabilities = (
        probabilities_np[:, 1]
        + probabilities_np[:, 2]
    )

    alarm_decisions = (
        alarm_probabilities
        >= alarm_threshold
    )

    results = []

    for row_index, record in enumerate(
        selected_records
    ):
        true_label = record["true_label"]
        predicted_label = int(
            predictions[row_index]
        )

        probability_values = {
            class_names[class_index]: float(
                probabilities_np[
                    row_index,
                    class_index,
                ]
            )
            for class_index in range(3)
        }

        result = {
            **record,
            "true_class": class_names[
                true_label
            ],
            "predicted_label": predicted_label,
            "predicted_class": class_names[
                predicted_label
            ],
            "probabilities": probability_values,
            "alarm_probability": float(
                alarm_probabilities[row_index]
            ),
            "alarm_threshold": alarm_threshold,
            "alarm_decision": bool(
                alarm_decisions[row_index]
            ),
            "classification_correct": bool(
                predicted_label == true_label
            ),
        }

        results.append(result)

        print("\n" + "=" * 80)
        print(
            f"Window {result['window_index']} "
            f"- {record['intended_class']} example"
        )
        print("=" * 80)
        print(
            f"True class      : "
            f"{result['true_class']}"
        )
        print(
            f"Predicted class : "
            f"{result['predicted_class']}"
        )
        print(
            "Prediction match: "
            + (
                "YES"
                if result["classification_correct"]
                else "NO"
            )
        )

        print("\nClass probabilities")

        for class_name, probability in (
            probability_values.items()
        ):
            print(
                f"  {class_name:12s}: "
                f"{probability:.6f}"
            )

        print(
            f"\nAlarm probability: "
            f"{result['alarm_probability']:.6f}"
        )
        print(
            f"Alarm threshold  : "
            f"{alarm_threshold:.6f}"
        )
        print(
            "Alarm decision   : "
            + (
                "ALARM"
                if result["alarm_decision"]
                else "NO ALARM"
            )
        )

    print("\nBatch inference information")
    print("-" * 80)
    print(f"Windows processed : {len(results)}")
    print(f"Device            : CPU")
    print(f"Inference time    : {inference_ms:.3f} ms")
    print(
        f"Average per window: "
        f"{inference_ms / len(results):.3f} ms"
    )

    return results, inference_ms


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline TUSZ semantic-validation "
            "and frozen-TCN inference prototype."
        )
    )

    parser.add_argument(
        "--patient-id",
        default=DEFAULT_PATIENT_ID,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models"
            / "frozen"
            / "seizure_tcn_p20_baseline_review_0d6774d3.pt"
        ),
    )

    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--ttl",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    patient_id = args.patient_id

    npz_path = args.npz or (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "tusz"
        / patient_id
        / f"{patient_id}.npz"
    )

    ttl_path = args.ttl or (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "tusz"
        / patient_id
        / f"{patient_id}.ttl"
    )

    checkpoint_path = args.checkpoint

    print("=" * 80)
    print("ONTOLOGY-DRIVEN EEG OFFLINE RESEARCH PROTOTYPE")
    print("=" * 80)
    print(
        "Scientific status: Research prototype only; "
        "not clinically validated; not a medical device."
    )
    print(f"Patient          : {patient_id}")
    print("Inference device : CPU")

    shacl_result = run_shacl_validation(
        ttl_path
    )

    if not shacl_result["passed"]:
        print("\nINFERENCE BLOCKED")
        print(
            "Semantic metadata failed SHACL validation."
        )
        return 2

    model, checkpoint, checkpoint_hash = (
        load_frozen_model(
            checkpoint_path
        )
    )

    (
        epochs,
        labels,
        ch_names,
        sfreq,
        npz_metadata,
    ) = load_patient_data(
        npz_path=npz_path,
        expected_patient_id=patient_id,
    )

    results, inference_ms = run_inference(
        model=model,
        checkpoint=checkpoint,
        epochs=epochs,
        labels=labels,
    )

    report = {
        "prototype_status": "PASS",
        "scientific_status": (
            "Research prototype only; not clinically validated; "
            "not a medical device."
        ),
        "execution_timestamp": (
            datetime.now().astimezone().isoformat()
        ),
        "patient_id": patient_id,
        "npz_path": str(npz_path),
        "ttl_path": str(ttl_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model_name": checkpoint["model_name"],
        "patient_count": int(
            checkpoint["patient_count"]
        ),
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_val_multiclass_f1": float(
            checkpoint[
                "best_val_multiclass_f1"
            ]
        ),
        "alarm_threshold_policy": str(
            checkpoint.get(
                "alarm_threshold_policy"
            )
        ),
        "alarm_decision_threshold": float(
            checkpoint[
                "alarm_decision_threshold"
            ]
        ),
        "shacl_validation": {
            "passed": bool(
                shacl_result["passed"]
            ),
            "return_code": int(
                shacl_result["return_code"]
            ),
            "duration_ms": float(
                shacl_result["duration_ms"]
            ),
        },
        "dataset": {
            "epoch_shape": [
                int(value)
                for value in epochs.shape
            ],
            "sampling_rate_hz": float(
                sfreq
            ),
            "channels": ch_names,
            **npz_metadata,
        },
        "inference_device": "cpu",
        "batch_inference_ms": float(
            inference_ms
        ),
        "results": results,
    }

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "prototype_demo"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        output_dir
        / f"{patient_id}_offline_demo_{timestamp}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("PROTOTYPE EXECUTION: PASS")
    print("=" * 80)
    print("SHACL validation : PASS")
    print("Frozen TCN       : VERIFIED")
    print("EEG input        : VERIFIED")
    print("Inference        : COMPLETED")
    print(f"Evidence report  : {report_path}")
    print(
        "\nReminder: Prediction correctness for three "
        "illustrative windows is not a replacement for "
        "held-out test metrics."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


