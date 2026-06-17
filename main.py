#!/usr/bin/env python3
"""
EEG Seizure MLOps Pipeline — CLI Entry Point

Usage:
    python main.py --dataset chbmit --stage preprocess
    python main.py --dataset tusz --stage preprocess
    python main.py --dataset tusz --stage all
"""

import argparse
import subprocess
import sys
import time

from pathlib import Path
from src.validation.rdf_generator import generate_tusz_ttl

# Small TUSZ subset for development.
# We use 5 patients instead of the full dataset because full TUSZ is too large
# to process quickly on a normal laptop.
TUSZ_PATIENTS = [
    "aaaaaajy",
    "aaaaaayf",
    "aaaaaazz",
    "aaaaabep",
    "aaaaabxe",
]

# CHB-MIT is downloaded inside the PhysioNet folder structure.
# The actual subject folders chb01, chb02, chb03 are inside this path.
CHBMIT_RAW_DIR = Path("data/raw/chbmit/physionet.org/files/chbmit/1.0.0")

# For Task 10, we first process only 3 CHB-MIT subjects.
CHBMIT_SUBJECTS = [
    "chb01",
    "chb02",
    "chb03",
]

def run(dataset="chbmit", stage="all"):
    print("=" * 55)
    print("  Ontology-Driven EEG Seizure Pipeline")
    print(f"  Dataset: {dataset.upper()} | Stage: {stage}")
    print("=" * 55)

    if stage in ("preprocess", "all"):
        print("\n[1/3] Running preprocessing...")

        if dataset == "chbmit":
            # CHB-MIT pipeline expects raw_dir to directly contain subject folders
            # such as chb01, chb02, chb03.
            #
            # Your downloaded data is nested inside:
            # data/raw/chbmit/physionet.org/files/chbmit/1.0.0
            #
            # So we pass that exact folder as --raw-dir.
            if not CHBMIT_RAW_DIR.exists():
                print(f"[ERROR] CHB-MIT raw directory not found: {CHBMIT_RAW_DIR}")
                sys.exit(1)

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
                ],
                check=True
            )

        elif dataset == "tusz":
            # Process each selected TUSZ patient one by one.
            # Each patient folder should exist inside data/raw/tusz/dev/
            for patient_id in TUSZ_PATIENTS:
                raw_patient_folder = Path("data/raw/tusz/dev") / patient_id

                print("\n--------------------------------------------------")
                print(f"Processing TUSZ patient from main.py: {patient_id}")
                print("--------------------------------------------------")

                # If a selected patient folder is missing, stop clearly.
                # This prevents silent mistakes in the pipeline.
                if not raw_patient_folder.exists():
                    print(f"[ERROR] Raw TUSZ patient folder not found: {raw_patient_folder}")
                    sys.exit(1)

                subprocess.run(
                    [
                        sys.executable,
                        "src/preprocessing/tusz_loader.py",
                        str(raw_patient_folder)
                    ],
                    check=True
                )

        print("Preprocessing done!")

        if dataset == "tusz":
            print("\n[TTL] Generating RDF metadata for processed TUSZ patients...")

            # Generate one TTL file for each processed TUSZ patient.
            # Example:
            # data/processed/tusz/aaaaaajy/aaaaaajy.npz
            # becomes
            # data/processed/tusz/aaaaaajy/aaaaaajy.ttl
            for patient_id in TUSZ_PATIENTS:
                npz_file = Path("data/processed/tusz") / patient_id / f"{patient_id}.npz"

                if not npz_file.exists():
                    print(f"[ERROR] Processed NPZ file not found: {npz_file}")
                    print("Preprocessing may have failed for this patient.")
                    sys.exit(1)

                generate_tusz_ttl(npz_file)

            print("[TTL] RDF metadata generation complete!")

    if stage in ("validate", "all"):
        print("\n[2/3] Running SHACL semantic validation...")

        if dataset == "tusz":
            # Validate each generated TUSZ TTL file.
            # If any one patient fails SHACL, the pipeline stops before training.
            for patient_id in TUSZ_PATIENTS:
                ttl_file = Path("data/processed/tusz") / patient_id / f"{patient_id}.ttl"

                print("\n--------------------------------------------------")
                print(f"Validating TUSZ TTL for patient: {patient_id}")
                print("--------------------------------------------------")

                if Path(ttl_file).exists():
                    subprocess.run([
                        sys.executable,
                        "src/validation/shacl_validator.py",
                        str(ttl_file)
                    ], check=True)
                else:
                    print(f"[ERROR] TTL file not found: {ttl_file}")
                    print("Run preprocessing first so RDF metadata can be generated.")
                    sys.exit(1)

        else:
            # CHB-MIT validation is kept simple for now.
            # We will improve this when we reach the CHB-MIT task.
            ttl_file = "data/processed/chbmit/chb01/chb01.ttl"

            if Path(ttl_file).exists():
                subprocess.run([
                    sys.executable,
                    "src/validation/shacl_validator.py",
                    ttl_file
                ], check=True)
            else:
                print(f"[ERROR] TTL file not found: {ttl_file}")
                print("Run preprocessing first so RDF metadata can be generated.")
                sys.exit(1)

        print("Validation stage complete!")

    if stage in ("train", "all"):
        print("\n[3/3] Training model with MLflow tracking...")
        train_start = time.perf_counter()
        
        # For now, train using the first processed TUSZ patient.
        # Later, when we reach the evaluation task, we will combine multiple patients
        # and add train/validation/test split.
        first_patient = TUSZ_PATIENTS[0]
        training_file = Path("data/processed/tusz") / first_patient / f"{first_patient}.npz"

        if not training_file.exists():
            print(f"[ERROR] Training file not found: {training_file}")
            sys.exit(1)

        subprocess.run([
            sys.executable,
            "src/training/train.py",
            "--data",
            str(training_file),
            "--epochs",
            "5"
        ], check=True)

        train_time = time.perf_counter() - train_start
        print(f"Model trained + logged to MLflow in {train_time:.1f} seconds!")

    print("\nPipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Seizure MLOps Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["chbmit", "tusz"],
        default="tusz"
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "validate", "train", "all"],
        default="preprocess"
    )
    args = parser.parse_args()
    run(args.dataset, args.stage)