"""
src/preprocessing/pipeline.py
───────────────────────────────
DVC pipeline stage: preprocess ALL CHB-MIT subjects in one reproducible run.

This script is the entry point for the ``chbmit_preprocess`` DVC stage
defined in dvc.yaml.  It:

1. Reads configuration from experiments/configs/preprocessing.yaml
2. Discovers every subject directory under ``data/raw/chbmit/``
3. For each subject, discovers every *.edf file and its paired summary
4. Calls ``loader.load_and_preprocess()`` on each file
5. Writes a combined manifest JSON to ``data/processed/chbmit/manifest.json``
   listing every output file and its MD5, used by downstream DVC stages

Reproducibility guarantees
---------------------------
- No system-clock timestamps in outputs (all timestamps are UTC ISO-8601
  recorded from the processing run, not used as hash inputs)
- NumPy random seed is fixed before any operation that could be stochastic
- MNE verbose is suppressed so log noise doesn't affect MD5 of stdout
- The manifest JSON is sorted by key so its hash is stable across runs

DVC stage definition (dvc.yaml)
--------------------------------
    stages:
      chbmit_preprocess:
        cmd: python src/preprocessing/pipeline.py
        deps:
          - src/preprocessing/pipeline.py
          - src/preprocessing/loader.py
          - data/raw/chbmit/
          - experiments/configs/preprocessing.yaml
        outs:
          - data/processed/chbmit/
        params:
          - experiments/configs/preprocessing.yaml:
            - preprocessing.bandpass_low_hz
            - preprocessing.bandpass_high_hz
            - preprocessing.notch_hz
            - preprocessing.epoch_duration_s
            - preprocessing.epoch_overlap_frac

Usage (direct)
--------------
    python src/preprocessing/pipeline.py
    python src/preprocessing/pipeline.py \
        --config experiments/configs/preprocessing.yaml \
        --raw-dir data/raw/chbmit \
        --out-dir data/processed/chbmit \
        --n-jobs 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import mne
import numpy as np
import yaml

from src.preprocessing.loader import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    EPOCH_DURATION_S,
    EPOCH_OVERLAP_FRAC,
    NOTCH_HZ,
    load_and_preprocess,
)


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# Suppress MNE's own verbose output — noise makes CI logs hard to read
mne.set_log_level("WARNING")

# Fix NumPy seed for reproducibility (no stochastic ops currently, but
# makes the pipeline safe for future additions like data augmentation)
np.random.seed(42)


# Default paths (relative to repo root)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG  = _REPO_ROOT / "experiments" / "configs" / "preprocessing.yaml"
_DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "raw"  / "chbmit"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "processed" / "chbmit"



# Config loading

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load preprocessing.yaml.  Returns a flat dict of preprocessing params.
    Falls back to module-level defaults if the file is absent.
    """
    defaults: dict[str, Any] = {
        "bandpass_low_hz":   BANDPASS_LOW_HZ,
        "bandpass_high_hz":  BANDPASS_HIGH_HZ,
        "notch_hz":          NOTCH_HZ,
        "epoch_duration_s":  EPOCH_DURATION_S,
        "epoch_overlap_frac": EPOCH_OVERLAP_FRAC,
    }
    if not config_path.is_file():
        log.warning("Config not found at %s — using defaults", config_path)
        return defaults

    with config_path.open() as fh:
        raw_cfg = yaml.safe_load(fh) or {}

    # Support nested ``preprocessing:`` key or flat structure
    cfg: dict[str, Any] = raw_cfg.get("preprocessing", raw_cfg)

    return {
        "bandpass_low_hz":    float(cfg.get("bandpass_low_hz",   defaults["bandpass_low_hz"])),
        "bandpass_high_hz":   float(cfg.get("bandpass_high_hz",  defaults["bandpass_high_hz"])),
        "notch_hz":           float(cfg.get("notch_hz",          defaults["notch_hz"])),
        "epoch_duration_s":   float(cfg.get("epoch_duration_s",  defaults["epoch_duration_s"])),
        "epoch_overlap_frac": float(cfg.get("epoch_overlap_frac", defaults["epoch_overlap_frac"])),
    }



# Subject / file discovery

def discover_subjects(raw_dir: Path) -> list[dict[str, Any]]:
    """
    Walk ``raw_dir`` and return a list of job descriptors, one per EDF file.

    Expected layout::

        raw_dir/
            chb01/
                chb01-summary.txt
                chb01_01.edf
                chb01_02.edf
                ...
            chb02/
                ...

    Each descriptor is::

        {
            "subject":      "chb01",
            "edf_path":     Path(...),
            "summary_path": Path(...),
        }
    """
    jobs: list[dict[str, Any]] = []

    if not raw_dir.is_dir():
        log.error("Raw data directory not found: %s", raw_dir)
        return jobs

    subject_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    log.info("Found %d subject director(ies) under %s", len(subject_dirs), raw_dir)

    for subject_dir in subject_dirs:
        subject = subject_dir.name

        # Find summary file — CHB-MIT names it <subject>-summary.txt
        summary_candidates = list(subject_dir.glob("*summary*"))
        if not summary_candidates:
            log.warning("[%s] No summary file found — seizure annotations will be empty", subject)
            summary_path = subject_dir / f"{subject}-summary.txt"  # non-existent; loader handles
        else:
            summary_path = summary_candidates[0]
            if len(summary_candidates) > 1:
                log.warning("[%s] Multiple summary files found; using %s",
                            subject, summary_path.name)

        edf_files = sorted(subject_dir.glob("*.edf"))
        if not edf_files:
            log.warning("[%s] No EDF files found — skipping", subject)
            continue

        log.info("[%s] %d EDF file(s), summary=%s", subject, len(edf_files), summary_path.name)
        for edf_path in edf_files:
            jobs.append({
                "subject":      subject,
                "edf_path":     edf_path,
                "summary_path": summary_path,
            })

    return jobs



# Worker (called in subprocess when n_jobs > 1)

def _process_one(
    job: dict[str, Any],
    out_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Process a single EDF file.  Returns a manifest entry dict on success or
    a dict with ``"error"`` key on failure.
    """
    subject      = job["subject"]
    edf_path     = Path(job["edf_path"])
    summary_path = Path(job["summary_path"])
    subject_out  = out_dir / subject

    try:
        npz_path = load_and_preprocess(
            edf_path=edf_path,
            summary_path=summary_path,
            out_dir=subject_out,
            bandpass_low=cfg["bandpass_low_hz"],
            bandpass_high=cfg["bandpass_high_hz"],
            notch_hz=cfg["notch_hz"],
            epoch_duration=cfg["epoch_duration_s"],
            epoch_overlap=cfg["epoch_overlap_frac"],
        )
        md5 = hashlib.md5(npz_path.read_bytes()).hexdigest()
        return {
            "subject":    subject,
            "session_id": edf_path.stem,
            "npz_path":   str(npz_path),
            "ttl_path":   str(npz_path.with_suffix(".ttl")),
            "md5":        md5,
            "status":     "ok",
        }
    except Exception as exc:
        log.error("[%s] FAILED %s: %s", subject, edf_path.name, exc, exc_info=True)
        return {
            "subject":    subject,
            "session_id": edf_path.stem,
            "npz_path":   None,
            "md5":        None,
            "status":     "error",
            "error":      str(exc),
        }



# Manifest writer

def write_manifest(
    entries: list[dict[str, Any]],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path:
    """
    Write a sorted, deterministic manifest JSON to ``out_dir/manifest.json``.
    The manifest is tracked by downstream DVC stages as a dependency.
    """
    manifest = {
        "config": cfg,                                    # params used
        "sessions": sorted(entries, key=lambda e: e["session_id"]),
        "summary": {
            "total":  len(entries),
            "ok":     sum(1 for e in entries if e["status"] == "ok"),
            "failed": sum(1 for e in entries if e["status"] == "error"),
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info("📋 Manifest written → %s", path)
    return path



# Main pipeline

def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    config_path: Path,
    n_jobs: int = 1,
) -> int:
    """
    Orchestrate the full CHB-MIT preprocessing pipeline.

    Returns
    -------
    int — exit code (0 = all OK, 1 = one or more files failed)
    """
    log.info("═" * 60)
    log.info("  CHB-MIT Preprocessing Pipeline")
    log.info("  raw_dir    : %s", raw_dir)
    log.info("  out_dir    : %s", out_dir)
    log.info("  config     : %s", config_path)
    log.info("  n_jobs     : %d", n_jobs)
    log.info("═" * 60)

    cfg  = load_config(config_path)
    log.info("Config: %s", cfg)

    jobs = discover_subjects(raw_dir)
    if not jobs:
        log.error("No EDF files found — nothing to process.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    if n_jobs == 1:
        # Single-process — easier to debug, simpler tracebacks
        for job in jobs:
            entries.append(_process_one(job, out_dir, cfg))
    else:
        # Multi-process — one worker per EDF file
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_process_one, job, out_dir, cfg): job
                for job in jobs
            }
            for future in as_completed(futures):
                entries.append(future.result())

    write_manifest(entries, out_dir, cfg)

    ok_count     = sum(1 for e in entries if e["status"] == "ok")
    failed_count = sum(1 for e in entries if e["status"] == "error")

    log.info("─" * 60)
    log.info("Pipeline complete: %d succeeded, %d failed (total %d)",
             ok_count, failed_count, len(entries))

    if failed_count > 0:
        log.error("The following sessions failed:")
        for e in entries:
            if e["status"] == "error":
                log.error("  %s — %s", e["session_id"], e.get("error", "unknown error"))
        return 1

    log.info("✅ All sessions processed successfully.")
    return 0



# CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Run the full CHB-MIT preprocessing DVC stage.",
    )
    p.add_argument("--config",   type=Path, default=_DEFAULT_CONFIG,
                   help=f"Path to preprocessing.yaml (default: {_DEFAULT_CONFIG})")
    p.add_argument("--raw-dir",  type=Path, default=_DEFAULT_RAW_DIR,
                   help=f"Root directory of raw CHB-MIT data (default: {_DEFAULT_RAW_DIR})")
    p.add_argument("--out-dir",  type=Path, default=_DEFAULT_OUT_DIR,
                   help=f"Output directory for processed epochs (default: {_DEFAULT_OUT_DIR})")
    p.add_argument("--n-jobs",   type=int,  default=1,
                   help="Number of parallel workers (default: 1 for deterministic output)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_pipeline(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        config_path=args.config,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    sys.exit(main())