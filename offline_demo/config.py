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

ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "seizure_tcn_p20_baseline_fp32.onnx"
)

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

EXPECTED_ONNX_SHA256 = (
    "af31d5a99ac683786b70abc4eea774d9"
    "c3b9564af41856358280060cd2f77420"
)


# ---------------------------------------------------------------------------
# Model and input contract
# ---------------------------------------------------------------------------

EXPECTED_PATIENT_COUNT = 20
EXPECTED_PARAMETER_COUNT = 76_643

EXPECTED_CHANNELS = 20
EXPECTED_TIMEPOINTS = 512
EXPECTED_CLASSES = 3

EXPECTED_ONNX_OPSET = 18

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