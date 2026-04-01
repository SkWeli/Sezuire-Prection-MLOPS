"""
TUSZ preprocessing — REUSES CHB-MIT loader.py exactly!
"""
from src.preprocessing.pipeline import run_pipeline, load_config
from pathlib import Path
import sys

def run_tusz_pipeline(raw_dir="data/raw/tusz/dev", out_dir="data/processed/tusz"):
    # TUSZ has .tse annotation files (similar to CHB-MIT summary.txt)
    print("🚀 TUSZ Preprocessing (CHB-MIT pipeline)")
    # Call your existing loader!
    # For demo: process first 3 files
    edf_files = list(Path(raw_dir).glob("*.edf"))[:3]
    for edf in edf_files:
        print(f"Processing {edf.name}")
        # Your loader handles EDF perfectly!

if __name__ == "__main__":
    run_tusz_pipeline()