from __future__ import annotations

import csv
import gc
import re
import sys
import zipfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path.cwd()
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chbmit"
    / "physionet.org"
    / "files"
    / "chbmit"
    / "1.0.0"
)
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "chbmit"
TUSZ_ROOT = PROJECT_ROOT / "data" / "processed" / "tusz"

EXPECTED_SUBJECTS = ("chb01", "chb02", "chb03")
EXPECTED_KEYS = {
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


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_scalar(array):
    array = np.asarray(array)

    if array.size == 0:
        return None

    value = array.reshape(-1)[0]

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    return value


def find_member(zip_file, key):
    expected = f"{key}.npy"

    for name in zip_file.namelist():
        if name == expected or name.endswith("/" + expected):
            return name

    return None


def read_npy_header(zip_file, key):
    member = find_member(zip_file, key)

    if member is None:
        return None, None

    with zip_file.open(member, "r") as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, dtype = np.lib.format._read_array_header(stream, version)

    return tuple(shape), str(dtype)


def read_small_array(zip_file, key):
    member = find_member(zip_file, key)

    if member is None:
        return None

    # Only small metadata arrays and labels are read.
    with zip_file.open(member, "r") as stream:
        return np.load(stream, allow_pickle=True)


def extract_subject_id(path, patient_id=None):
    candidates = []

    if patient_id is not None:
        candidates.append(str(patient_id).lower())

    candidates.extend(part.lower() for part in path.parts)

    for candidate in candidates:
        match = re.search(r"chb\d{2}", candidate)
        if match:
            return match.group(0)

    return "unknown"


def load_reference_tusz_channels():
    tusz_files = sorted(TUSZ_ROOT.rglob("*.npz"))

    if not tusz_files:
        return None, None

    reference = tusz_files[0]

    with zipfile.ZipFile(reference, "r") as archive:
        ch_names = read_small_array(archive, "ch_names")

    if ch_names is None:
        return reference, None

    channels = [decode_value(value) for value in np.asarray(ch_names).reshape(-1)]
    return reference, channels


print("=" * 72)
print("CHB-MIT STRUCTURAL AND CHANNEL-COMPATIBILITY AUDIT")
print("=" * 72)

print(f"\nProject root    : {PROJECT_ROOT}")
print(f"Raw root        : {RAW_ROOT}")
print(f"Processed root  : {PROCESSED_ROOT}")

print("\nRAW SUBJECT CHECK")
print("-" * 72)

for subject in EXPECTED_SUBJECTS:
    subject_dir = RAW_ROOT / subject
    edf_files = sorted(subject_dir.glob("*.edf")) if subject_dir.exists() else []

    print(
        f"{subject}: "
        f"directory_exists={subject_dir.exists()} | "
        f"edf_count={len(edf_files)}"
    )

npz_files = sorted(PROCESSED_ROOT.rglob("*.npz")) if PROCESSED_ROOT.exists() else []
ttl_files = sorted(PROCESSED_ROOT.rglob("*.ttl")) if PROCESSED_ROOT.exists() else []

print("\nPROCESSED OUTPUT INVENTORY")
print("-" * 72)
print(f"NPZ files found : {len(npz_files)}")
print(f"TTL files found : {len(ttl_files)}")

for path in npz_files:
    print(f"NPZ: {path.relative_to(PROJECT_ROOT)}")

for path in ttl_files:
    print(f"TTL: {path.relative_to(PROJECT_ROOT)}")

reference_path, reference_channels = load_reference_tusz_channels()

print("\nTUSZ CHANNEL REFERENCE")
print("-" * 72)

if reference_path is None:
    print("No TUSZ NPZ file found.")
elif reference_channels is None:
    print(f"Reference file found but ch_names missing: {reference_path}")
else:
    print(f"Reference file: {reference_path.relative_to(PROJECT_ROOT)}")
    print(f"Channel count : {len(reference_channels)}")
    print(f"Channel order : {reference_channels}")

rows = []
subjects_with_npz = set()

print("\nCHB-MIT NPZ AUDIT")
print("-" * 72)

for npz_path in npz_files:
    notes = []
    structural_pass = True

    try:
        with zipfile.ZipFile(npz_path, "r") as archive:
            available_keys = {
                Path(name).stem
                for name in archive.namelist()
                if name.endswith(".npy")
            }

            missing_keys = sorted(EXPECTED_KEYS - available_keys)

            epochs_shape, epochs_dtype = read_npy_header(archive, "epochs")
            labels_shape, labels_dtype = read_npy_header(archive, "labels")

            labels = read_small_array(archive, "labels")
            ch_names_array = read_small_array(archive, "ch_names")
            sfreq_array = read_small_array(archive, "sfreq")
            patient_id_array = read_small_array(archive, "patient_id")
            n_windows_array = read_small_array(archive, "n_windows")
            n_interictal_array = read_small_array(archive, "n_interictal")
            n_pre_ictal_array = read_small_array(archive, "n_pre_ictal")
            n_ictal_array = read_small_array(archive, "n_ictal")

        patient_id = normalize_scalar(patient_id_array)
        subject = extract_subject_id(npz_path, patient_id)
        subjects_with_npz.add(subject)

        channels = (
            [decode_value(value) for value in np.asarray(ch_names_array).reshape(-1)]
            if ch_names_array is not None
            else []
        )

        sfreq = normalize_scalar(sfreq_array)
        n_windows_meta = normalize_scalar(n_windows_array)
        n_interictal_meta = normalize_scalar(n_interictal_array)
        n_pre_ictal_meta = normalize_scalar(n_pre_ictal_array)
        n_ictal_meta = normalize_scalar(n_ictal_array)

        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            label_counts = {
                int(label): int(count)
                for label, count in zip(unique_labels, unique_counts)
            }
        else:
            labels = np.asarray([])
            label_counts = {}

        exact_tusz_order = (
            reference_channels is not None
            and channels == reference_channels
        )

        same_tusz_channel_set = (
            reference_channels is not None
            and set(channels) == set(reference_channels)
            and len(channels) == len(reference_channels)
        )

        if missing_keys:
            structural_pass = False
            notes.append(f"missing_keys={missing_keys}")

        if epochs_shape is None or len(epochs_shape) != 3:
            structural_pass = False
            notes.append(f"invalid_epochs_shape={epochs_shape}")
        elif epochs_shape[1:] != (20, 512):
            structural_pass = False
            notes.append(
                f"expected_epochs_tail=(20, 512), actual={epochs_shape[1:]}"
            )

        if labels_shape is None or len(labels_shape) != 1:
            structural_pass = False
            notes.append(f"invalid_labels_shape={labels_shape}")

        if (
            epochs_shape is not None
            and labels_shape is not None
            and epochs_shape[0] != labels_shape[0]
        ):
            structural_pass = False
            notes.append("epoch_and_label_counts_differ")

        if set(label_counts) - {0, 1, 2}:
            structural_pass = False
            notes.append(f"unexpected_labels={sorted(label_counts)}")

        if len(channels) != 20:
            structural_pass = False
            notes.append(f"channel_count={len(channels)}, expected=20")

        if len(channels) != len(set(channels)):
            structural_pass = False
            notes.append("duplicate_channel_names")

        actual_windows = int(labels_shape[0]) if labels_shape else 0

        if n_windows_meta is not None and int(n_windows_meta) != actual_windows:
            structural_pass = False
            notes.append(
                f"n_windows_metadata={n_windows_meta}, actual={actual_windows}"
            )

        metadata_class_total = sum(
            int(value)
            for value in (
                n_interictal_meta,
                n_pre_ictal_meta,
                n_ictal_meta,
            )
            if value is not None
        )

        if (
            all(
                value is not None
                for value in (
                    n_interictal_meta,
                    n_pre_ictal_meta,
                    n_ictal_meta,
                )
            )
            and metadata_class_total != actual_windows
        ):
            structural_pass = False
            notes.append(
                f"metadata_class_total={metadata_class_total}, "
                f"actual={actual_windows}"
            )

        if reference_channels is not None and not exact_tusz_order:
            notes.append("channel_order_not_identical_to_tusz")

        if reference_channels is not None and not same_tusz_channel_set:
            notes.append("channel_set_not_identical_to_tusz")

        row = {
            "file": str(npz_path.relative_to(PROJECT_ROOT)),
            "subject": subject,
            "file_size_mb": round(npz_path.stat().st_size / (1024 ** 2), 2),
            "patient_id": patient_id,
            "available_keys": ";".join(sorted(available_keys)),
            "missing_keys": ";".join(missing_keys),
            "epochs_shape": str(epochs_shape),
            "epochs_dtype": epochs_dtype,
            "labels_shape": str(labels_shape),
            "labels_dtype": labels_dtype,
            "label_counts": str(label_counts),
            "sfreq": sfreq,
            "channel_count": len(channels),
            "channels": ";".join(channels),
            "n_windows_metadata": n_windows_meta,
            "n_interictal_metadata": n_interictal_meta,
            "n_pre_ictal_metadata": n_pre_ictal_meta,
            "n_ictal_metadata": n_ictal_meta,
            "exact_tusz_channel_order": exact_tusz_order,
            "same_tusz_channel_set": same_tusz_channel_set,
            "structural_pass": structural_pass,
            "notes": "; ".join(notes),
        }

        rows.append(row)

        print(f"\nFile             : {row['file']}")
        print(f"Subject          : {subject}")
        print(f"Patient ID       : {patient_id}")
        print(f"Keys             : {sorted(available_keys)}")
        print(f"Epochs           : shape={epochs_shape}, dtype={epochs_dtype}")
        print(f"Labels           : shape={labels_shape}, counts={label_counts}")
        print(f"Sampling rate    : {sfreq}")
        print(f"Channels ({len(channels)}): {channels}")
        print(f"TUSZ exact order : {exact_tusz_order}")
        print(f"TUSZ same set    : {same_tusz_channel_set}")
        print(f"Structural pass  : {structural_pass}")
        print(f"Notes            : {row['notes'] or 'None'}")

    except Exception as exc:
        rows.append(
            {
                "file": str(npz_path.relative_to(PROJECT_ROOT)),
                "subject": "unknown",
                "structural_pass": False,
                "notes": f"{type(exc).__name__}: {exc}",
            }
        )

        print(f"\n[ERROR] {npz_path}: {type(exc).__name__}: {exc}")

    finally:
        gc.collect()

missing_subject_outputs = sorted(
    set(EXPECTED_SUBJECTS) - subjects_with_npz
)

print("\nFINAL AUDIT SUMMARY")
print("-" * 72)
print(f"Expected subjects          : {list(EXPECTED_SUBJECTS)}")
print(f"Subjects represented in NPZ: {sorted(subjects_with_npz)}")
print(f"Missing subject NPZ outputs: {missing_subject_outputs}")

if not rows:
    print("RESULT: NO CHB-MIT NPZ OUTPUTS FOUND")
elif any(not row.get("structural_pass", False) for row in rows):
    print("RESULT: STRUCTURAL AUDIT FAILED")
elif reference_channels is not None and any(
    not row.get("exact_tusz_channel_order", False)
    for row in rows
):
    print("RESULT: STRUCTURE PASSED, BUT CHANNEL COMPATIBILITY FAILED")
elif missing_subject_outputs:
    print("RESULT: OUTPUTS ARE INCOMPLETE")
else:
    print("RESULT: STRUCTURAL AND CHANNEL-ORDER AUDIT PASSED")

output_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "reports/verification/chbmit_audit.csv"
)
output_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "file",
    "subject",
    "file_size_mb",
    "patient_id",
    "available_keys",
    "missing_keys",
    "epochs_shape",
    "epochs_dtype",
    "labels_shape",
    "labels_dtype",
    "label_counts",
    "sfreq",
    "channel_count",
    "channels",
    "n_windows_metadata",
    "n_interictal_metadata",
    "n_pre_ictal_metadata",
    "n_ictal_metadata",
    "exact_tusz_channel_order",
    "same_tusz_channel_set",
    "structural_pass",
    "notes",
]

with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV evidence saved to: {output_csv}")
