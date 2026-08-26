#!/usr/bin/env python3
"""
Export the frozen P20 baseline SeizureTCN model to FP32 ONNX.

This script performs the following quality checks:

1. Verifies the frozen checkpoint SHA256.
2. Confirms the checkpoint belongs to the selected 20-patient baseline TCN.
3. Reconstructs the PyTorch model architecture.
4. Loads the checkpoint weights using strict state-dictionary matching.
5. Runs a PyTorch forward-pass smoke test.
6. Exports the model to ONNX.
7. Validates the exported ONNX graph.
8. Saves export metadata, hashes, and evidence reports.

The exported model produces raw multiclass logits:

    output[0] = Interictal logit
    output[1] = Pre-Ictal logit
    output[2] = Ictal logit

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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
import torch


# ---------------------------------------------------------------------------
# Resolve the project root.
#
# This file is stored at:
#     project_root/src/deployment/onnx_export.py
#
# parents[2] therefore points to the repository root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports such as `from src.models.tcn import SeizureTCN`
# when this script is executed directly or through `python -m`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn import SeizureTCN


# Suppress only the known PyTorch warning related to the older
# weight_norm API used by the current frozen architecture.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*weight_norm.*",
)


# ---------------------------------------------------------------------------
# Frozen model identity
# ---------------------------------------------------------------------------

EXPECTED_CHECKPOINT_SHA256 = (
    "0d6774d36f2f040b4ce6ae5f9964fc30"
    "b004a7fe96b4a9fdfd401f733921a4e7"
)

EXPECTED_PARAMETER_COUNT = 76_643
EXPECTED_PATIENT_COUNT = 20
EXPECTED_CHANNELS = 20
EXPECTED_TIMEPOINTS = 512
EXPECTED_CLASSES = 3

CLASS_NAMES = [
    "Interictal",
    "Pre-Ictal",
    "Ictal",
]


# ---------------------------------------------------------------------------
# Default input and output locations
# ---------------------------------------------------------------------------

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

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "onnx"
)


def sha256_file(path: Path) -> str:
    """
    Calculate the SHA256 digest of a file.

    The file is read in blocks so this function remains safe for
    larger future model artifacts.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """
    Load and verify the frozen PyTorch checkpoint.

    This prevents accidentally exporting:

    - the overwritten 12-patient checkpoint;
    - the TCN-v2 AdamW experiment;
    - a corrupted or modified checkpoint.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Frozen checkpoint was not found: {checkpoint_path}"
        )

    checkpoint_hash = sha256_file(checkpoint_path)

    print(f"Checkpoint path   : {checkpoint_path}")
    print(f"Checkpoint SHA256 : {checkpoint_hash}")

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA256 does not match the frozen P20 baseline."
        )

    # PyTorch versions differ slightly in their torch.load signature.
    # This fallback supports the user's current environment while also
    # remaining compatible with older PyTorch versions.
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
            "Expected the checkpoint to be a dictionary."
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
        "alarm_threshold_policy",
    }

    missing_keys = sorted(
        required_keys - set(checkpoint.keys())
    )

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing required metadata: {missing_keys}"
        )

    if checkpoint["model_name"] != "SeizureTCN":
        raise RuntimeError(
            "The selected checkpoint is not a SeizureTCN checkpoint."
        )

    if int(checkpoint["patient_count"]) != EXPECTED_PATIENT_COUNT:
        raise RuntimeError(
            "The selected checkpoint was not trained using the "
            "frozen 20-patient experiment."
        )

    if int(checkpoint["n_channels"]) != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Unexpected channel count: {checkpoint['n_channels']}"
        )

    if int(checkpoint["n_timepoints"]) != EXPECTED_TIMEPOINTS:
        raise RuntimeError(
            f"Unexpected timepoint count: {checkpoint['n_timepoints']}"
        )

    if int(checkpoint["n_classes"]) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Unexpected class count: {checkpoint['n_classes']}"
        )

    return checkpoint


def build_verified_model(
    checkpoint: dict[str, Any],
) -> SeizureTCN:
    """
    Reconstruct the TCN architecture and load the frozen weights.

    strict=True ensures every checkpoint parameter matches the current
    Python model definition exactly.
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
            "The reconstructed model parameter count does not match "
            f"the frozen architecture. Expected "
            f"{EXPECTED_PARAMETER_COUNT}, received {parameter_count}."
        )

    # Run one CPU forward pass before attempting ONNX export.
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
            "Unexpected PyTorch output shape. "
            f"Expected (1, {EXPECTED_CLASSES}), "
            f"received {tuple(output.shape)}."
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "The PyTorch smoke-test output contains NaN or infinity."
        )

    print(f"Model parameters  : {parameter_count}")
    print(f"PyTorch output    : {tuple(output.shape)}")
    print("PyTorch smoke test: PASS")

    return model


def export_to_onnx(
    model: SeizureTCN,
    output_path: Path,
    opset_version: int,
) -> float:
    """
    Export the verified PyTorch TCN model to FP32 ONNX.

    The tensor shape is:

        (batch_size, 1, 20, 512)

    Only the batch dimension is dynamic. Channel count and signal
    length remain fixed because they are part of the model contract.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dummy_input = torch.zeros(
        1,
        1,
        EXPECTED_CHANNELS,
        EXPECTED_TIMEPOINTS,
        dtype=torch.float32,
    )

    print("\nExporting FP32 ONNX model...")
    print(f"ONNX output path : {output_path}")
    print(f"ONNX opset       : {opset_version}")
    print("Dynamic dimension: batch_size only")

    export_start = time.perf_counter()

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,

            # Store the trained parameters inside the ONNX file.
            export_params=True,

            # A stable opset suitable for the operators used by this TCN.
            opset_version=opset_version,

            # Safe graph optimization for inference.
            do_constant_folding=True,

            # Names make later ONNX Runtime code clearer.
            input_names=["eeg_input"],
            output_names=["logits"],

            # Allow inference with batch size 1 or larger while preserving
            # the fixed EEG shape of 20 channels x 512 samples.
            dynamic_axes={
                "eeg_input": {
                    0: "batch_size",
                },
                "logits": {
                    0: "batch_size",
                },
            },
        )

    export_duration_ms = (
        time.perf_counter() - export_start
    ) * 1000.0

    if not output_path.exists():
        raise RuntimeError(
            "PyTorch completed without creating the ONNX file."
        )

    if output_path.stat().st_size == 0:
        raise RuntimeError(
            "The exported ONNX file is empty."
        )

    print(
        f"Export duration   : {export_duration_ms:.2f} ms"
    )

    return export_duration_ms


def validate_and_annotate_onnx(
    onnx_path: Path,
    checkpoint: dict[str, Any],
    checkpoint_hash: str,
    opset_version: int,
) -> dict[str, Any]:
    """
    Validate the exported ONNX graph and embed model provenance.

    onnx.checker.check_model() checks that the serialized graph follows
    ONNX structural and operator rules.

    The metadata added here allows the exported artifact to be traced
    back to the exact frozen PyTorch checkpoint.
    """
    onnx_model = onnx.load(str(onnx_path))

    # Read the operator-set version from the exported ONNX graph itself.
    #
    # An ONNX model may import multiple domains. The default ONNX
    # operator domain is represented by an empty string or "ai.onnx".
    # Reading the value from the saved model prevents the report from
    # incorrectly assuming that the requested opset was actually used.
    default_opset_versions = [
        int(opset.version)
        for opset in onnx_model.opset_import
        if opset.domain in {"", "ai.onnx"}
    ]

    if not default_opset_versions:
        raise RuntimeError(
            "The exported model does not declare a default ONNX opset."
        )

    actual_opset_version = max(default_opset_versions)

    print(f"Actual ONNX opset : {actual_opset_version}")

    metadata = {
        "project": (
            "Ontology-Driven Verifiable MLOps for "
            "Edge-Deployable EEG Seizure Detection"
        ),
        "scientific_status": (
            "Research prototype; not clinically validated; "
            "not a medical device."
        ),
        "model_name": "SeizureTCN",
        "source_checkpoint_sha256": checkpoint_hash,
        "training_patient_count": str(
            checkpoint["patient_count"]
        ),
        "best_epoch": str(
            checkpoint["best_epoch"]
        ),
        "best_validation_macro_f1": str(
            checkpoint["best_val_multiclass_f1"]
        ),
        "input_contract": (
            "float32[batch_size,1,20,512]"
        ),
        "output_contract": (
            "float32[batch_size,3] logits"
        ),
        "class_0": CLASS_NAMES[0],
        "class_1": CLASS_NAMES[1],
        "class_2": CLASS_NAMES[2],
        "alarm_threshold_policy": str(
            checkpoint["alarm_threshold_policy"]
        ),
        "alarm_decision_threshold": str(
            checkpoint["alarm_decision_threshold"]
        ),
        "requested_onnx_opset": str(opset_version),
        "actual_onnx_opset": str(actual_opset_version),
    }

    # Add provenance metadata directly to the ONNX model.
    for key, value in metadata.items():
        property_entry = onnx_model.metadata_props.add()
        property_entry.key = str(key)
        property_entry.value = str(value)

    onnx_model.producer_name = "Final Year Project Senuda"
    onnx_model.producer_version = "1.0"

    # Validate the complete ONNX graph.
    onnx.checker.check_model(
        onnx_model,
        full_check=True,
    )

    # Save the metadata-enhanced and validated model.
    onnx.save(
        onnx_model,
        str(onnx_path),
    )

    # Reload and validate again to make sure the final saved file,
    # rather than only the in-memory object, is valid.
    onnx.checker.check_model(
        str(onnx_path),
        full_check=True,
    )

    print("ONNX graph check  : PASS")

    graph_inputs = [
        graph_input.name
        for graph_input in onnx_model.graph.input
    ]

    graph_outputs = [
        graph_output.name
        for graph_output in onnx_model.graph.output
    ]

    print(f"ONNX inputs       : {graph_inputs}")
    print(f"ONNX outputs      : {graph_outputs}")

    return {
        "metadata": metadata,
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "ir_version": int(onnx_model.ir_version),
        "producer_name": onnx_model.producer_name,
        "producer_version": onnx_model.producer_version,
        "requested_opset_version": int(opset_version),
        "actual_opset_version": int(actual_opset_version),
    }


def save_export_evidence(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    onnx_path: Path,
    report_directory: Path,
    export_duration_ms: float,
    onnx_information: dict[str, Any],
    opset_version: int,
) -> tuple[Path, Path, Path]:
    """
    Save machine-readable and human-readable export evidence.
    """
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    onnx_hash = sha256_file(
        onnx_path
    )

    onnx_size_bytes = onnx_path.stat().st_size

    manifest = {
        "status": "onnx_export_pass",
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "scientific_status": (
            "Research prototype only; not clinically validated; "
            "not a medical device."
        ),
        "source_checkpoint": str(
            checkpoint_path.resolve()
        ),
        "source_checkpoint_sha256": checkpoint_hash,
        "onnx_model": str(
            onnx_path.resolve()
        ),
        "onnx_sha256": onnx_hash,
        "onnx_size_bytes": int(
            onnx_size_bytes
        ),
        "onnx_size_kib": float(
            onnx_size_bytes / 1024.0
        ),
        "requested_onnx_opset": int(
            opset_version
        ),
        "actual_onnx_opset": int(
            onnx_information["actual_opset_version"]
        ),
        "export_duration_ms": float(
            export_duration_ms
        ),
        "model_name": str(
            checkpoint["model_name"]
        ),
        "patient_count": int(
            checkpoint["patient_count"]
        ),
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_val_multiclass_f1": float(
            checkpoint["best_val_multiclass_f1"]
        ),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "input_shape": [
            "batch_size",
            1,
            EXPECTED_CHANNELS,
            EXPECTED_TIMEPOINTS,
        ],
        "output_shape": [
            "batch_size",
            EXPECTED_CLASSES,
        ],
        "class_names": CLASS_NAMES,
        "alarm_threshold_policy": str(
            checkpoint["alarm_threshold_policy"]
        ),
        "alarm_decision_threshold": float(
            checkpoint["alarm_decision_threshold"]
        ),
        "onnx_graph": onnx_information,
        "pytorch_version": torch.__version__,
        "onnx_version": onnx.__version__,
    }

    manifest_path = (
        report_directory
        / "tcn_fp32_onnx_export_manifest.json"
    )

    hash_path = (
        report_directory
        / "tcn_fp32_onnx_sha256.txt"
    )

    summary_path = (
        report_directory
        / "tcn_fp32_onnx_export_summary.txt"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with hash_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"{onnx_hash}  {onnx_path.as_posix()}\n"
        )

    summary_lines = [
        "FP32 ONNX Export Summary",
        "=" * 72,
        "",
        f"Status                    : PASS",
        f"Model                     : SeizureTCN",
        f"Training patient count    : {checkpoint['patient_count']}",
        f"Best epoch                : {checkpoint['best_epoch']}",
        (
            "Best validation macro F1  : "
            f"{checkpoint['best_val_multiclass_f1']}"
        ),
        f"Parameter count           : {EXPECTED_PARAMETER_COUNT}",
        (
            "Input contract            : "
            "float32[batch_size, 1, 20, 512]"
        ),
        (
            "Output contract           : "
            "float32[batch_size, 3] logits"
        ),
        (
            "Requested ONNX opset      : "
            f"{opset_version}"
        ),
        (
            "Actual ONNX opset         : "
            f"{onnx_information['actual_opset_version']}"
        ),
        f"ONNX file                 : {onnx_path}",
        f"ONNX size bytes           : {onnx_size_bytes}",
        f"ONNX size KiB             : {onnx_size_bytes / 1024.0:.2f}",
        f"Export duration ms        : {export_duration_ms:.2f}",
        f"Checkpoint SHA256         : {checkpoint_hash}",
        f"ONNX SHA256               : {onnx_hash}",
        f"PyTorch version           : {torch.__version__}",
        f"ONNX version              : {onnx.__version__}",
        "",
        (
            "Scientific status         : Research prototype only; "
            "not clinically validated; not a medical device."
        ),
        "",
        (
            "Important: Structural ONNX validation does not yet prove "
            "numerical equivalence with PyTorch."
        ),
        (
            "PyTorch versus ONNX Runtime parity testing is the next stage."
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    return (
        manifest_path,
        hash_path,
        summary_path,
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options.

    The defaults point to the already verified P20 checkpoint and the
    intended FP32 ONNX artifact locations.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export the frozen P20 SeizureTCN baseline "
            "to validated FP32 ONNX."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Path to the frozen PyTorch TCN checkpoint."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=(
            "Destination path for the FP32 ONNX model."
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
        help=(
            "Directory for export manifests and hash evidence."
        ),
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help=(
            "ONNX operator-set version used during export. "
            "Opset 18 is used because the installed PyTorch exporter "
            "implements this model using ONNX opset 18."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Execute the complete verified FP32 ONNX export.
    """
    args = parse_arguments()

    checkpoint_path = args.checkpoint.resolve()
    onnx_path = args.output.resolve()
    report_directory = args.report_dir.resolve()

    print("=" * 80)
    print("FROZEN P20 TCN - FP32 ONNX EXPORT")
    print("=" * 80)
    print(
        "Scientific status: Research prototype only; "
        "not clinically validated; not a medical device."
    )
    print(f"PyTorch version : {torch.__version__}")
    print(f"ONNX version    : {onnx.__version__}")
    print("Execution device: CPU")
    print()

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    model = build_verified_model(
        checkpoint
    )

    export_duration_ms = export_to_onnx(
        model=model,
        output_path=onnx_path,
        opset_version=args.opset,
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    onnx_information = validate_and_annotate_onnx(
        onnx_path=onnx_path,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        opset_version=args.opset,
    )

    (
        manifest_path,
        hash_path,
        summary_path,
    ) = save_export_evidence(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        onnx_path=onnx_path,
        report_directory=report_directory,
        export_duration_ms=export_duration_ms,
        onnx_information=onnx_information,
        opset_version=args.opset,
    )

    print("\n" + "=" * 80)
    print("FP32 ONNX EXPORT: PASS")
    print("=" * 80)
    print(f"ONNX model     : {onnx_path}")
    print(f"ONNX SHA256    : {sha256_file(onnx_path)}")
    print(
        f"ONNX size      : "
        f"{onnx_path.stat().st_size / 1024.0:.2f} KiB"
    )
    print(f"Export manifest: {manifest_path}")
    print(f"Hash evidence  : {hash_path}")
    print(f"Summary report : {summary_path}")
    print()
    print(
        "Next required step: compare PyTorch and ONNX Runtime "
        "outputs using real TUSZ EEG windows."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())