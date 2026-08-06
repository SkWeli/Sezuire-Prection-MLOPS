"""
Shared configuration for the offline viva demonstrations.

Keeping paths and verified artifact identities in one file prevents the
PyTorch, ONNX and invalid-metadata demonstrations from drifting apart.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATIENT_ID = "aaaaaayf"

PATIENT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tusz"
    / PATIENT_ID
)

PATIENT_NPZ_PATH = (
    PATIENT_DIRECTORY
    / f"{PATIENT_ID}.npz"
)

PATIENT_TTL_PATH = (
    PATIENT_DIRECTORY
    / f"{PATIENT_ID}.ttl"
)

ONTOLOGY_PATH = (
    PROJECT_ROOT
    / "ontology"
    / "eeg_epilepsy.ttl"
)

SHACL_SHAPES_PATH = (
    PROJECT_ROOT
    / "ontology"
    / "shacl_shapes.ttl"
)

SHACL_VALIDATOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "validation"
    / "shacl_validator.py"
)

PYTORCH_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "frozen"
    / "seizure_tcn_p20_baseline_review_0d6774d3.pt"
)

# ---------------------------------------------------------------------------
# Deployment model artifacts
# ---------------------------------------------------------------------------

FP32_ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_fp32.onnx"
)

INT8_ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_int8_qdq.onnx"
)

# Backward-compatible alias used by any older FP32 demo code.
ONNX_MODEL_PATH = FP32_ONNX_MODEL_PATH

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "prototype_demo"
)


# ---------------------------------------------------------------------------
# Verified artifact identities
# ---------------------------------------------------------------------------

EXPECTED_CHECKPOINT_SHA256 = (
    "0d6774d36f2f040b4ce6ae5f9964fc30"
    "b004a7fe96b4a9fdfd401f733921a4e7"
)

EXPECTED_FP32_ONNX_SHA256 = (
    "af31d5a99ac683786b70abc4eea774d9"
    "c3b9564af41856358280060cd2f77420"
)

EXPECTED_INT8_ONNX_SHA256 = (
    "9fbac2f59b5acd0276036f8ab9ea65c6"
    "bc8555d71b5ba9c16124062889e43969"
)

# Backward-compatible alias for the original FP32 demonstration.
EXPECTED_ONNX_SHA256 = EXPECTED_FP32_ONNX_SHA256


# ---------------------------------------------------------------------------
# Model and input contract
# ---------------------------------------------------------------------------

EXPECTED_PATIENT_COUNT = 20
EXPECTED_PARAMETER_COUNT = 76_643

EXPECTED_CHANNELS = 20
EXPECTED_TIMEPOINTS = 512
EXPECTED_CLASSES = 3

EXPECTED_ONNX_OPSET = 18

ONNX_INPUT_NAME = "eeg_input"
ONNX_OUTPUT_NAME = "logits"

ALARM_THRESHOLD = 0.17
ALARM_THRESHOLD_POLICY = "specificity_constrained"

CLASS_NAMES = [
    "Interictal",
    "Pre-Ictal",
    "Ictal",
]


# ---------------------------------------------------------------------------
# Demonstration window selection
#
# These are illustrative windows from the held-out patient.
# They are not a replacement for the full test-set evaluation.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scientific disclaimer
# ---------------------------------------------------------------------------

SCIENTIFIC_STATUS = (
    "Research prototype only; not clinically validated; "
    "not a medical device."
)