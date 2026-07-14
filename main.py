#!/usr/bin/env python3
"""
EEG Seizure MLOps Pipeline — CLI Entry Point

Usage:
    python main.py --dataset chbmit --stage preprocess
    python main.py --dataset tusz --stage preprocess
    python main.py --dataset tusz --stage train --epochs 50 --max-patients 25
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from src.validation.rdf_generator import generate_tusz_ttl

# Mapped directly to the raw architecture on the external drive
CHBMIT_RAW_DIR = Path("data/raw/chbmit/physionet.org/files/chbmit/1.0.0")

def get_tusz_raw_folders(max_patients=None):
    """Dynamically pull TUSZ patient paths from both dev and eval."""
    base_dir = Path("data/raw/tusz")
    folders = []
    
    for sub_dir in ["dev", "eval"]:
        target = base_dir / sub_dir
        if target.exists():
            folders.extend([p for p in target.iterdir() if p.is_dir()])
            
    # Sort alphabetically by folder name
    folders = sorted(folders, key=lambda x: x.name)
    
    return folders[:max_patients] if max_patients else folders

def run(dataset="chbmit", stage="all", extra_args=None):
    extra_args = extra_args or []
    print("=" * 55)
    print("  Ontology-Driven EEG Seizure Pipeline")
    print(f"  Dataset: {dataset.upper()} | Stage: {stage}")
    if extra_args:
        print(f"  Passed Flags: {' '.join(extra_args)}")
    print("=" * 55)

    # ---------------------------------------------------------
    # 1. PREPROCESSING STAGE
    # ---------------------------------------------------------
    if stage in ("preprocess", "all"):
        print("\n[1/3] Running preprocessing...")

        if dataset == "chbmit":
            if not CHBMIT_RAW_DIR.exists():
                print(f"[ERROR] CHB-MIT raw directory not found: {CHBMIT_RAW_DIR}")
                sys.exit(1)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.preprocessing.pipeline",
                    "--raw-dir", str(CHBMIT_RAW_DIR),
                    "--out-dir", "data/processed/chbmit",
                    "--n-jobs", "1",
                ] + extra_args,
                check=True
            )

        elif dataset == "tusz":
            # If a user passes --max-patients to main.py during preprocessing, respect it
            parser = argparse.ArgumentParser(allow_abbrev=False)
            parser.add_argument("--max-patients", type=int, default=None)
            known, _ = parser.parse_known_args(extra_args)
            
            # Use the new folder scanning function
            tusz_folders = get_tusz_raw_folders(known.max_patients)
            print(f"🚀 Preparing to preprocess {len(tusz_folders)} TUSZ patients from dev and eval...")

            for raw_patient_folder in tusz_folders:
                patient_id = raw_patient_folder.name
                
                if not raw_patient_folder.exists():
                    print(f"[ERROR] Raw TUSZ patient folder not found: {raw_patient_folder}")
                    sys.exit(1)

                subprocess.run(
                    [sys.executable, "src/preprocessing/tusz_loader.py", str(raw_patient_folder)],
                    check=True
                )

        print("Preprocessing done!")

        if dataset == "tusz":
            print("\n[TTL] Generating RDF metadata for processed TUSZ patients...")
            tusz_folders = get_tusz_raw_folders(known.max_patients if 'known' in locals() else None)
            
            for raw_patient_folder in tusz_folders:
                patient_id = raw_patient_folder.name
                npz_file = Path("data/processed/tusz") / patient_id / f"{patient_id}.npz"
                if npz_file.exists():
                    generate_tusz_ttl(npz_file)
            print("[TTL] RDF metadata generation complete!")
            
    # ---------------------------------------------------------
    # 2. VALIDATION STAGE
    # ---------------------------------------------------------
    if stage in ("validate", "all"):
        print("\n[2/3] Running SHACL semantic validation...")
        
        target_dir = Path(f"data/processed/{dataset}")
        ttl_files = sorted(list(target_dir.rglob("*.ttl")))

        if not ttl_files:
            print(f"[ERROR] No TTL files found in {target_dir}")
            sys.exit(1)

        print(f"🔍 Found {len(ttl_files)} semantic graphs to validate for {dataset.upper()}.")
        
        for ttl_file in ttl_files:
            subprocess.run([sys.executable, "src/validation/shacl_validator.py", str(ttl_file)], check=True)

        print("Validation stage complete!")

    # ---------------------------------------------------------
    # 3. TRAINING STAGE
    # ---------------------------------------------------------
    if stage in ("train", "all"):
        print("\n[3/3] Training model with MLflow tracking...")
        train_start = time.perf_counter()
        
        # Pointing to the directory instead of a single file
        data_target = str(Path(f"data/processed/{dataset}"))

        # We pass extra_args straight down so flags like --model tcn work perfectly
        subprocess.run(
            [
                sys.executable,
                "src/training/train.py",
                "--data", data_target
            ] + extra_args,
            check=True
        )

        train_time = time.perf_counter() - train_start
        print(f"Model trained + logged to MLflow in {train_time:.1f} seconds!")

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    # parse_known_args splits the arguments main.py understands from the ones meant for sub-scripts
    parser = argparse.ArgumentParser(description="EEG Seizure MLOps Pipeline Router")
    parser.add_argument("--dataset", choices=["chbmit", "tusz"], default="tusz")
    parser.add_argument("--stage", choices=["preprocess", "validate", "train", "all"], default="preprocess")
    
    args, extra_args = parser.parse_known_args()
    run(args.dataset, args.stage, extra_args)