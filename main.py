#!/usr/bin/env python3
"""EEG Seizure MLOps pipeline command-line entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.validation.rdf_generator import generate_tusz_ttl

# Force UTF-8 for Windows PowerShell, redirected output, and child Python processes.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

CHBMIT_RAW_DIR = Path("data/raw/chbmit/physionet.org/files/chbmit/1.0.0")


def get_tusz_raw_folders(max_patients=None):
    """Discover TUSZ patient directories from dev and eval."""
    base_dir = Path("data/raw/tusz")
    folders = []

    for sub_dir in ["dev", "eval"]:
        target = base_dir / sub_dir
        if target.exists():
            folders.extend(path for path in target.iterdir() if path.is_dir())

    folders = sorted(folders, key=lambda path: path.name)
    return folders[:max_patients] if max_patients else folders


def validate_processed_metadata(dataset):
    """Validate every generated TTL file and fail immediately on any SHACL error."""
    target_dir = Path(f"data/processed/{dataset}")
    ttl_files = sorted(target_dir.rglob("*.ttl"))
    npz_files = sorted(target_dir.rglob("*.npz"))

    if not npz_files:
        print(f"[ERROR] No processed NPZ files found in {target_dir}")
        raise SystemExit(1)

    if not ttl_files:
        print(f"[ERROR] No TTL metadata files found in {target_dir}")
        print("Training is blocked until RDF metadata is generated and validated.")
        raise SystemExit(1)

    ttl_stems = {ttl_file.stem for ttl_file in ttl_files}
    missing_metadata = [npz_file for npz_file in npz_files if npz_file.stem not in ttl_stems]
    if missing_metadata:
        print("[ERROR] Processed NPZ files are missing matching TTL metadata:")
        for npz_file in missing_metadata:
            print(f"   - {npz_file}")
        print("Training is blocked until every processed subject has metadata.")
        raise SystemExit(1)

    print(
        f"[INFO] Found {len(npz_files)} processed NPZ file(s) and "
        f"{len(ttl_files)} semantic graph(s) for {dataset.upper()}."
    )

    for ttl_file in ttl_files:
        print(f"   Validating: {ttl_file}")
        subprocess.run(
            [sys.executable, "src/validation/shacl_validator.py", str(ttl_file)],
            check=True,
        )

    print("[PASS] All semantic metadata passed SHACL validation.")
    return True


def run(dataset="chbmit", stage="all", extra_args=None):
    extra_args = extra_args or []
    validation_passed = False

    print("=" * 60)
    print("  Ontology-Driven EEG Seizure Pipeline")
    print(f"  Dataset: {dataset.upper()} | Stage: {stage}")
    if extra_args:
        print(f"  Passed flags: {' '.join(extra_args)}")
    print("=" * 60)

    if stage in ("preprocess", "all"):
        print("\n[1/3] Running preprocessing...")

        if dataset == "chbmit":
            if not CHBMIT_RAW_DIR.exists():
                print(f"[ERROR] CHB-MIT raw directory not found: {CHBMIT_RAW_DIR}")
                raise SystemExit(1)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.preprocessing.pipeline",
                    "--raw-dir",
                    str(CHBMIT_RAW_DIR),
                    "--out-dir",
                    "data/processed/chbmit",
                    "--n-jobs",
                    "1",
                ]
                + extra_args,
                check=True,
            )

        elif dataset == "tusz":
            preprocess_parser = argparse.ArgumentParser(allow_abbrev=False)
            preprocess_parser.add_argument("--max-patients", type=int, default=None)
            known, _ = preprocess_parser.parse_known_args(extra_args)

            tusz_folders = get_tusz_raw_folders(known.max_patients)
            if not tusz_folders:
                print("[ERROR] No TUSZ patient folders were found under data/raw/tusz/dev or eval.")
                raise SystemExit(1)

            print(f"[INFO] Preparing to preprocess {len(tusz_folders)} TUSZ patient(s)...")
            for raw_patient_folder in tusz_folders:
                subprocess.run(
                    [
                        sys.executable,
                        "src/preprocessing/tusz_loader.py",
                        str(raw_patient_folder),
                    ],
                    check=True,
                )

            print("\n[TTL] Generating RDF metadata for processed TUSZ patients...")
            for raw_patient_folder in tusz_folders:
                patient_id = raw_patient_folder.name
                npz_file = Path("data/processed/tusz") / patient_id / f"{patient_id}.npz"

                if not npz_file.exists():
                    print(f"[ERROR] Expected processed file was not created: {npz_file}")
                    raise SystemExit(1)

                generate_tusz_ttl(npz_file)

            print("[TTL] RDF metadata generation complete.")

        print("[PASS] Preprocessing stage complete.")

    if stage in ("validate", "all"):
        print("\n[2/3] Running SHACL semantic validation...")
        validation_passed = validate_processed_metadata(dataset)

    if stage in ("train", "all"):
        print("\n[3/3] Preparing model training...")

        # A direct --stage train command must never bypass SHACL validation.
        if not validation_passed:
            print("[GATE] Validating metadata before training...")
            validation_passed = validate_processed_metadata(dataset)

        if not validation_passed:
            print("[ERROR] SHACL validation did not pass. Training is blocked.")
            raise SystemExit(1)

        data_target = Path(f"data/processed/{dataset}")
        if not data_target.exists():
            print(f"[ERROR] Processed dataset directory not found: {data_target}")
            raise SystemExit(1)

        train_start = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                "src/training/train.py",
                "--data",
                str(data_target),
            ]
            + extra_args,
            check=True,
        )
        elapsed = time.perf_counter() - train_start
        print(f"[PASS] Model trained and logged to MLflow in {elapsed:.1f} seconds.")

    print("\n[PASS] Pipeline execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Seizure MLOps Pipeline Router")
    parser.add_argument("--dataset", choices=["chbmit", "tusz"], default="tusz")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "validate", "train", "all"],
        default="preprocess",
    )
    args, extra_args = parser.parse_known_args()
    run(args.dataset, args.stage, extra_args)
