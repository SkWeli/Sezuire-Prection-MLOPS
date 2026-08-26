"""
Extract reusable EEG feature matrices for hierarchical XGBoost.

For every processed TUSZ patient NPZ:

    epochs: (N, 20, 512)
        ↓
    220 engineered EEG features/window
        ↓
    <patient_id>_features.npz

The script is resumable:
existing patient feature files are skipped unless --overwrite is used.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.features.eeg_feature_extractor import (
    build_feature_names,
    extract_window_features,
)


DEFAULT_INPUT = Path("data/processed/tusz")
DEFAULT_OUTPUT = Path("data/features/tusz_xgb_220")


def find_patient_npz_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.npz"))

    # Avoid accidentally reading generated feature files if paths overlap.
    files = [
        path
        for path in files
        if not path.name.endswith("_features.npz")
    ]

    return files


def extract_patient(
    input_file: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    with np.load(input_file, allow_pickle=True) as data:
        epochs = data["epochs"]
        labels = data["labels"]

        patient_id = str(data["patient_id"])
        sfreq = float(data["sfreq"])
        ch_names = [str(x) for x in data["ch_names"]]

        n_windows_metadata = int(data["n_windows"])
        n_interictal_metadata = int(data["n_interictal"])
        n_preictal_metadata = int(data["n_pre_ictal"])
        n_ictal_metadata = int(data["n_ictal"])

        if epochs.ndim != 3:
            raise ValueError(
                f"{patient_id}: expected epochs with 3 dimensions, "
                f"got {epochs.shape}"
            )

        if epochs.shape[1:] != (20, 512):
            raise ValueError(
                f"{patient_id}: expected EEG shape (N,20,512), "
                f"got {epochs.shape}"
            )

        if len(labels) != len(epochs):
            raise ValueError(
                f"{patient_id}: labels/windows mismatch "
                f"{len(labels)} != {len(epochs)}"
            )

        if len(epochs) != n_windows_metadata:
            raise ValueError(
                f"{patient_id}: n_windows metadata mismatch"
            )

        if abs(sfreq - 128.0) > 1e-6:
            raise ValueError(
                f"{patient_id}: expected 128 Hz, got {sfreq}"
            )

        class_counts = np.bincount(
            labels.astype(np.int64),
            minlength=3,
        )

        expected_counts = np.array(
            [
                n_interictal_metadata,
                n_preictal_metadata,
                n_ictal_metadata,
            ]
        )

        if not np.array_equal(class_counts, expected_counts):
            raise ValueError(
                f"{patient_id}: class-count metadata mismatch. "
                f"labels={class_counts.tolist()} "
                f"metadata={expected_counts.tolist()}"
            )

        feature_names = build_feature_names(ch_names)

        if len(feature_names) != 220:
            raise ValueError(
                f"{patient_id}: expected 220 feature names, "
                f"got {len(feature_names)}"
            )

        output_file = output_dir / f"{patient_id}_features.npz"

        if output_file.exists() and not overwrite:
            print(
                f"[SKIP] {patient_id} already exists: "
                f"{output_file}"
            )

            return {
                "patient_id": patient_id,
                "n_windows": len(labels),
                "n_interictal": int(class_counts[0]),
                "n_pre_ictal": int(class_counts[1]),
                "n_ictal": int(class_counts[2]),
                "status": "SKIPPED_EXISTING",
                "seconds": 0.0,
            }

        print()
        print("=" * 70)
        print(f"[PATIENT] {patient_id}")
        print(f"[INPUT]   {input_file}")
        print(f"[WINDOWS] {len(epochs)}")
        print(
            "[CLASSES] "
            f"Interictal={class_counts[0]}, "
            f"Pre-Ictal={class_counts[1]}, "
            f"Ictal={class_counts[2]}"
        )

        X = np.empty(
            (len(epochs), 220),
            dtype=np.float32,
        )

        start = time.time()

        for i in range(len(epochs)):
            X[i] = extract_window_features(
                epochs[i],
                sfreq=sfreq,
            )

            if (i + 1) % 500 == 0:
                print(
                    f"[PROGRESS] {patient_id}: "
                    f"{i + 1}/{len(epochs)}"
                )

        elapsed = time.time() - start

        if X.shape != (len(epochs), 220):
            raise RuntimeError(
                f"{patient_id}: unexpected feature matrix "
                f"shape {X.shape}"
            )

        if not np.isfinite(X).all():
            bad_count = int(
                np.size(X) - np.isfinite(X).sum()
            )

            raise RuntimeError(
                f"{patient_id}: feature matrix contains "
                f"{bad_count} non-finite values"
            )

        window_index = np.arange(
            len(labels),
            dtype=np.int64,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            output_file,
            X=X,
            y=labels.astype(np.int64),
            patient_id=np.array(patient_id),
            window_index=window_index,
            sfreq=np.array(sfreq),
            ch_names=np.asarray(ch_names, dtype=str),
            feature_names=np.asarray(feature_names, dtype=str),
        )

        # Immediately verify saved file.
        with np.load(output_file, allow_pickle=False) as saved:
            if saved["X"].shape != X.shape:
                raise RuntimeError(
                    f"{patient_id}: saved X shape verification failed"
                )

            if not np.array_equal(
                saved["y"],
                labels,
            ):
                raise RuntimeError(
                    f"{patient_id}: saved labels verification failed"
                )

        print(
            f"[PASS] {patient_id} -> "
            f"{X.shape} in {elapsed:.2f} seconds"
        )

        return {
            "patient_id": patient_id,
            "n_windows": len(labels),
            "n_interictal": int(class_counts[0]),
            "n_pre_ictal": int(class_counts[1]),
            "n_ictal": int(class_counts[2]),
            "status": "EXTRACTED",
            "seconds": round(elapsed, 2),
        }


def write_summary(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        output_dir / "feature_extraction_summary.csv"
    )

    fieldnames = [
        "patient_id",
        "n_windows",
        "n_interictal",
        "n_pre_ictal",
        "n_ictal",
        "status",
        "seconds",
    ]

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return summary_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N patients. "
            "Useful for smoke testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    files = find_patient_npz_files(
        args.input_dir
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ files found under {args.input_dir}"
        )

    if args.limit is not None:
        files = files[: args.limit]

    print(
        f"[INFO] Found {len(files)} patient NPZ files"
    )

    rows = []

    for index, input_file in enumerate(
        files,
        start=1,
    ):
        print()
        print(
            f"[INFO] Patient {index}/{len(files)}"
        )

        result = extract_patient(
            input_file=input_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

        rows.append(result)

        # Update summary after every patient.
        write_summary(
            rows,
            args.output_dir,
        )

    summary_file = write_summary(
        rows,
        args.output_dir,
    )

    total_windows = sum(
        row["n_windows"]
        for row in rows
    )

    total_interictal = sum(
        row["n_interictal"]
        for row in rows
    )

    total_preictal = sum(
        row["n_pre_ictal"]
        for row in rows
    )

    total_ictal = sum(
        row["n_ictal"]
        for row in rows
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("Patients    :", len(rows))
    print("Windows     :", total_windows)
    print("Interictal  :", total_interictal)
    print("Pre-Ictal   :", total_preictal)
    print("Ictal       :", total_ictal)

    print()
    print("Summary CSV :", summary_file)


if __name__ == "__main__":
    main()