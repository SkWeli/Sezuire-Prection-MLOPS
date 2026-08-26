"""Create patient-level EEG dataset and proposed split summaries.

Usage:
    python -m src.evaluation.dataset_audit --data data/processed/tusz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.evaluation.splits import select_balanced_patient_indices


CLASS_NAMES = ("interictal", "preictal", "ictal")


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        x_key = "epochs" if "epochs" in data.files else "X" if "X" in data.files else None
        y_key = "labels" if "labels" in data.files else "y" if "y" in data.files else None
        if x_key is None or y_key is None:
            raise KeyError(f"{path} lacks epochs/labels or X/y. Found: {data.files}")

        features = np.asarray(data[x_key])
        labels = np.asarray(data[y_key], dtype=np.int64).reshape(-1)
        sfreq = (
            float(np.asarray(data["sfreq"]).reshape(-1)[0])
            if "sfreq" in data.files
            else float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
            if "sampling_rate" in data.files
            else 128.0
        )

    if features.ndim == 4 and features.shape[1] == 1:
        features = features[:, 0]
    if features.ndim != 3:
        raise ValueError(f"Expected 3D EEG windows in {path}, got {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"Window/label mismatch in {path}")

    return features, labels, sfreq


def _counts(labels):
    return np.bincount(labels, minlength=3)[:3]


def _write_patient_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id", "npz_path", "windows", "channels", "timepoints",
        "sampling_rate_hz", "interictal", "preictal", "ictal",
        "interictal_pct", "preictal_pct", "ictal_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_split_csv(split_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "patient_count", "patient_ids", "windows",
        "interictal", "preictal", "ictal",
        "interictal_pct", "preictal_pct", "ictal_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(split_rows)


def main(data_path, patient_output, split_output, max_patients=None, seed=42):
    root = Path(data_path)
    files = sorted(root.rglob("*.npz"))
    if max_patients is not None:
        files = files[:max_patients]
    if len(files) < 3:
        raise ValueError(f"Need at least 3 NPZ files for an audit split, found {len(files)}")

    patient_datasets = []
    patient_ids = []
    rows = []

    for path in files:
        features, labels, sfreq = _read_npz(path)
        counts = _counts(labels)
        total = int(counts.sum())
        patient_id = path.stem
        patient_ids.append(patient_id)
        patient_datasets.append((features, labels))

        rows.append({
            "patient_id": patient_id,
            "npz_path": str(path),
            "windows": total,
            "channels": int(features.shape[1]),
            "timepoints": int(features.shape[2]),
            "sampling_rate_hz": sfreq,
            "interictal": int(counts[0]),
            "preictal": int(counts[1]),
            "ictal": int(counts[2]),
            "interictal_pct": counts[0] / total * 100.0,
            "preictal_pct": counts[1] / total * 100.0,
            "ictal_pct": counts[2] / total * 100.0,
        })

    _write_patient_csv(rows, Path(patient_output))

    selection = select_balanced_patient_indices(
        patient_datasets=patient_datasets,
        seed=seed,
        patient_ids=patient_ids,
    )

    split_rows = []
    for split_name, indices in (
        ("train", selection.train_indices),
        ("validation", selection.val_indices),
        ("test", selection.test_indices),
    ):
        split_counts = np.zeros(3, dtype=np.int64)
        windows = 0
        ids = []
        for index in indices:
            ids.append(patient_ids[index])
            labels = patient_datasets[index][1]
            split_counts += _counts(labels)
            windows += len(labels)

        split_rows.append({
            "split": split_name,
            "patient_count": len(indices),
            "patient_ids": ";".join(ids),
            "windows": windows,
            "interictal": int(split_counts[0]),
            "preictal": int(split_counts[1]),
            "ictal": int(split_counts[2]),
            "interictal_pct": split_counts[0] / windows * 100.0,
            "preictal_pct": split_counts[1] / windows * 100.0,
            "ictal_pct": split_counts[2] / windows * 100.0,
        })

    _write_split_csv(split_rows, Path(split_output))

    print(f"[PASS] Patient summary: {patient_output}")
    print(f"[PASS] Split summary  : {split_output}")
    print(f"[INFO] Split score    : {selection.score:.6f}")
    for row in split_rows:
        print(
            f"[INFO] {row['split']:<10} patients={row['patient_count']:>2} "
            f"windows={row['windows']:>7} "
            f"class%={row['interictal_pct']:.2f}/"
            f"{row['preictal_pct']:.2f}/{row['ictal_pct']:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit processed EEG patient distributions.")
    parser.add_argument("--data", default="data/processed/tusz")
    parser.add_argument(
        "--patient-output",
        default="reports/tables/tusz_patient_class_summary.csv",
    )
    parser.add_argument(
        "--split-output",
        default="reports/tables/tusz_balanced_split_summary.csv",
    )
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data,
        patient_output=args.patient_output,
        split_output=args.split_output,
        max_patients=args.max_patients,
        seed=args.seed,
    )
