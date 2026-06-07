#!/usr/bin/env python3
"""
EEG Seizure MLOps Pipeline — CLI Entry Point

Usage:
    python main.py --dataset chbmit --stage preprocess
    python main.py --dataset tusz --stage preprocess
    python main.py --dataset tusz --stage all
"""
from pathlib import Path
import argparse
import subprocess
import sys
import time

def run(dataset="chbmit", stage="all"):
    print("=" * 55)
    print("  Ontology-Driven EEG Seizure Pipeline")
    print(f"  Dataset: {dataset.upper()} | Stage: {stage}")
    print("=" * 55)

    if stage in ("preprocess", "all"):
        print("\n[1/3] Running preprocessing...")

        if dataset == "chbmit":
            subprocess.run(
                ["python", "-m", "src.preprocessing.pipeline"],
                check=True
            )

        elif dataset == "tusz":
            subprocess.run(
                [
                    "python",
                    "src/preprocessing/tusz_loader.py",
                    "data/raw/tusz/dev/aaaaaajy"
                ],
                check=True
            )

        print("Preprocessing done!")

    if stage in ("validate", "all"):
        print("\n[2/3] Running SHACL semantic validation...")
        # Pass generated .ttl file (if exists) or skip gracefully
        ttl_file = "data/processed/tusz/aaaaaajy/aaaaaajy.ttl"
        if Path(ttl_file).exists():
            subprocess.run([
                "python", "src/validation/shacl_validator.py", ttl_file
            ], check=True)
        else:
            print("RDF TTL not generated yet — skipping SHACL (next feature)")
        print("Validation stage complete!")

    if stage in ("train", "all"):
        print("\n[3/3] Training model with MLflow tracking...")
        train_start = time.perf_counter()
        
        subprocess.run([
            "python", "src/training/train.py",
            "--data", "data/processed/tusz/aaaaaajy/aaaaaajy.npz",
            "--epochs", "5"
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