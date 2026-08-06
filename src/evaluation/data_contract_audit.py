"""
CPU-friendly integrity audit for the processed TUSZ cohort.

The audit is deliberately patient-by-patient so compressed NPZ files are never
kept in memory as one giant concatenated cohort.

Outputs:
    reports/model_improvement/data_contract_audit.txt
    reports/model_improvement/patient_class_distribution.csv
    reports/model_improvement/preictal_definition_evidence.txt

Example:
    python -m src.evaluation.data_contract_audit \
        --data-dir data/processed/tusz \
        --split experiments/splits/p20_frozen_split.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = {0: "Interictal", 1: "Pre-Ictal", 2: "Ictal"}
EXPECTED_LABELS = {0, 1, 2}

CHANNEL_KEYS = ("channels", "channel_names", "ch_names", "channel_order")
PATIENT_ID_KEYS = ("patient_id", "subject_id", "patient", "subject")
STEP_KEYS = ("window_step_s", "step_s", "stride_s")
OVERLAP_KEYS = ("window_overlap_frac", "overlap_fraction", "overlap")

EXCLUDED_SCAN_DIRS = {
    ".git",
    ".venv",
    "venv_linux",
    "__pycache__",
    ".pytest_cache",
    "data",
    "mlruns",
    "mlartifacts",
    "dist",
    "artifacts",
}


@dataclass
class Finding:
    severity: str
    message: str


@dataclass
class PatientAudit:
    patient_id: str
    split: str
    npz_path: str = ""
    ttl_path: str = ""
    status: str = "PASS"
    windows: int = 0
    interictal: int = 0
    preictal: int = 0
    ictal: int = 0
    sfreq: float = float("nan")
    channels: int = 0
    timepoints: int = 0
    window_duration_s: float = float("nan")
    stored_step_s: float = float("nan")
    finite_scan: str = ""
    channel_order_available: bool = False
    channel_order: str = ""
    findings: List[Finding] = field(default_factory=list)

    def add(self, severity: str, message: str) -> None:
        severity = severity.upper()
        self.findings.append(Finding(severity, message))
        if severity == "FAIL":
            self.status = "FAIL"
        elif severity == "WARN" and self.status == "PASS":
            self.status = "WARN"


def _load_split(path: Path) -> Tuple[Dict[str, List[str]], List[Finding]]:
    findings: List[Finding] = []
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    train = list(payload.get("train", []))
    val = list(payload.get("validation", payload.get("val", [])))
    test = list(payload.get("test", []))
    split = {"train": train, "validation": val, "test": test}

    all_ids = train + val + test
    duplicates = sorted({pid for pid in all_ids if all_ids.count(pid) > 1})
    if duplicates:
        findings.append(Finding("FAIL", f"Patients appear in multiple splits: {duplicates}"))
    else:
        findings.append(Finding("PASS", "Train/validation/test patient IDs are disjoint."))

    if len(all_ids) != len(set(all_ids)):
        findings.append(Finding("FAIL", "Split contains duplicate patient IDs."))
    if len(set(all_ids)) != 20:
        findings.append(
            Finding(
                "WARN",
                f"Frozen split contains {len(set(all_ids))} unique patients; expected P20=20.",
            )
        )
    else:
        findings.append(Finding("PASS", "Frozen split contains exactly 20 unique patients."))

    if payload.get("status") != "frozen":
        findings.append(Finding("WARN", "Split JSON is not explicitly marked status='frozen'."))
    else:
        findings.append(Finding("PASS", "Split JSON is marked frozen."))

    return split, findings


def _index_files(data_dir: Path, suffix: str) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in sorted(data_dir.rglob(f"*{suffix}")):
        candidates = {path.stem, path.parent.name}
        for candidate in candidates:
            index.setdefault(candidate, []).append(path)
    return index


def _choose_patient_file(index: Dict[str, List[Path]], patient_id: str) -> Optional[Path]:
    matches = index.get(patient_id, [])
    if not matches:
        return None
    exact_stem = [path for path in matches if path.stem == patient_id]
    if exact_stem:
        return sorted(exact_stem)[0]
    return sorted(matches)[0]


def _scalar(npz: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in npz.files:
            value = np.asarray(npz[key]).squeeze()
            if value.size == 1:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _text_value(npz: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in npz.files:
            try:
                value = np.asarray(npz[key]).squeeze()
                if value.size == 1:
                    return str(value.item() if hasattr(value, "item") else value)
            except Exception:
                return None
    return None


def _channel_order(npz: np.lib.npyio.NpzFile) -> Optional[List[str]]:
    for key in CHANNEL_KEYS:
        if key in npz.files:
            try:
                values = np.asarray(npz[key]).reshape(-1)
                return [str(value) for value in values.tolist()]
            except Exception:
                return None
    return None


def _sample_indices(n_windows: int, sample_count: int) -> np.ndarray:
    if n_windows <= sample_count:
        return np.arange(n_windows, dtype=np.int64)
    return np.linspace(0, n_windows - 1, sample_count, dtype=np.int64)


def _check_normalization_contract(sample: np.ndarray) -> Tuple[bool, str]:
    """
    Apply the exact training-time per-window, per-channel z-score transform and
    verify that it produces finite, approximately zero-mean output.
    """
    sample = np.asarray(sample, dtype=np.float32)
    means = sample.mean(axis=-1, keepdims=True)
    stds = sample.std(axis=-1, keepdims=True)
    normalized = (sample - means) / (stds + 1e-6)

    if not np.isfinite(normalized).all():
        return False, "Per-window/per-channel normalization produced non-finite values."

    max_abs_mean = float(np.max(np.abs(normalized.mean(axis=-1))))
    constant_series = int(np.sum(stds.squeeze(-1) < 1e-8))
    ok = max_abs_mean < 1e-4
    detail = (
        f"simulated z-score max|mean|={max_abs_mean:.2e}; "
        f"constant channel-windows={constant_series}"
    )
    return ok, detail


def audit_patient(
    patient_id: str,
    split_name: str,
    npz_path: Optional[Path],
    ttl_path: Optional[Path],
    expected_sfreq: float,
    expected_channels: int,
    expected_timepoints: int,
    expected_step_s: float,
    finite_mode: str,
    sample_windows: int,
) -> PatientAudit:
    audit = PatientAudit(patient_id=patient_id, split=split_name)

    if npz_path is None:
        audit.add("FAIL", "NPZ file not found.")
        return audit
    audit.npz_path = str(npz_path.relative_to(PROJECT_ROOT) if npz_path.is_relative_to(PROJECT_ROOT) else npz_path)

    if ttl_path is None:
        audit.add("WARN", "Matching TTL file not found beside or below the processed-data directory.")
    else:
        audit.ttl_path = str(ttl_path.relative_to(PROJECT_ROOT) if ttl_path.is_relative_to(PROJECT_ROOT) else ttl_path)

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            missing = [key for key in ("epochs", "labels") if key not in npz.files]
            if missing:
                audit.add("FAIL", f"Missing required NPZ keys: {missing}")
                return audit

            epochs = np.asarray(npz["epochs"])
            labels = np.asarray(npz["labels"]).reshape(-1)

            if epochs.ndim == 4 and epochs.shape[1] == 1:
                audit.add("WARN", "Epochs are stored as (N,1,C,T); expected storage contract is (N,C,T).")
                epochs_view = epochs[:, 0]
            elif epochs.ndim == 3:
                epochs_view = epochs
            else:
                audit.add("FAIL", f"Unexpected epochs shape: {epochs.shape}")
                return audit

            audit.windows = int(epochs_view.shape[0])
            audit.channels = int(epochs_view.shape[1])
            audit.timepoints = int(epochs_view.shape[2])

            if labels.shape[0] != audit.windows:
                audit.add(
                    "FAIL",
                    f"Epoch/label count mismatch: {audit.windows} epochs vs {labels.shape[0]} labels.",
                )

            if audit.channels != expected_channels:
                audit.add("FAIL", f"Channel count is {audit.channels}; expected {expected_channels}.")
            if audit.timepoints != expected_timepoints:
                audit.add("FAIL", f"Timepoints are {audit.timepoints}; expected {expected_timepoints}.")

            if not np.issubdtype(labels.dtype, np.integer):
                if np.all(np.equal(labels, labels.astype(np.int64))):
                    audit.add("WARN", f"Labels are integral but stored as {labels.dtype}, not integer dtype.")
                    labels = labels.astype(np.int64)
                else:
                    audit.add("FAIL", f"Labels are not integer class IDs: dtype={labels.dtype}.")
            labels = labels.astype(np.int64, copy=False)

            unique_labels = set(np.unique(labels).tolist())
            invalid_labels = sorted(unique_labels - EXPECTED_LABELS)
            if invalid_labels:
                audit.add("FAIL", f"Invalid label IDs found: {invalid_labels}")

            counts = np.bincount(labels, minlength=3)
            audit.interictal = int(counts[0])
            audit.preictal = int(counts[1])
            audit.ictal = int(counts[2])

            audit.sfreq = _scalar(npz, ("sfreq", "sampling_rate", "fs")) or float("nan")
            if not np.isfinite(audit.sfreq):
                audit.add("WARN", "Sampling-rate key not stored; training code would fall back to 128 Hz.")
                audit.sfreq = expected_sfreq
            elif not np.isclose(audit.sfreq, expected_sfreq, atol=1e-6):
                audit.add("FAIL", f"Sampling rate is {audit.sfreq}; expected {expected_sfreq} Hz.")

            audit.window_duration_s = audit.timepoints / audit.sfreq
            expected_duration = expected_timepoints / expected_sfreq
            if not np.isclose(audit.window_duration_s, expected_duration, atol=1e-6):
                audit.add(
                    "FAIL",
                    f"Window duration is {audit.window_duration_s:.4f}s; expected {expected_duration:.4f}s.",
                )

            stored_step = _scalar(npz, STEP_KEYS)
            if stored_step is None:
                overlap = _scalar(npz, OVERLAP_KEYS)
                if overlap is not None:
                    stored_step = audit.window_duration_s * (1.0 - overlap)
            if stored_step is None:
                audit.add(
                    "WARN",
                    f"Window step/overlap is not stored in NPZ; expected {expected_step_s}s must be verified from preprocessing code/config.",
                )
            else:
                audit.stored_step_s = float(stored_step)
                if not np.isclose(audit.stored_step_s, expected_step_s, atol=1e-6):
                    audit.add(
                        "FAIL",
                        f"Stored/inferred step is {audit.stored_step_s:.4f}s; expected {expected_step_s:.4f}s.",
                    )

            stored_patient_id = _text_value(npz, PATIENT_ID_KEYS)
            if stored_patient_id is not None and stored_patient_id != patient_id:
                audit.add(
                    "FAIL",
                    f"Stored patient ID '{stored_patient_id}' does not match split/path ID '{patient_id}'.",
                )
            elif stored_patient_id is None:
                audit.add("WARN", "Patient ID is not stored as an NPZ field; path/filename is the only identity source.")

            channels = _channel_order(npz)
            if channels is None:
                audit.add("WARN", "Channel names/order are not stored in the NPZ and cannot be verified from this artifact alone.")
            else:
                audit.channel_order_available = True
                audit.channel_order = "|".join(channels)
                if len(channels) != expected_channels:
                    audit.add(
                        "FAIL",
                        f"Stored channel-name count is {len(channels)}; expected {expected_channels}.",
                    )

            indices = _sample_indices(audit.windows, sample_windows)
            sample = epochs_view[indices]
            if finite_mode == "full":
                finite_ok = bool(np.isfinite(epochs_view).all())
                audit.finite_scan = "full"
            else:
                finite_ok = bool(np.isfinite(sample).all())
                audit.finite_scan = f"sample:{len(indices)}"
            if not finite_ok:
                audit.add("FAIL", f"{audit.finite_scan} finite-value scan found NaN or infinity.")

            normalization_ok, normalization_detail = _check_normalization_contract(sample)
            if normalization_ok:
                audit.add("PASS", f"Training-time normalization contract works on sample: {normalization_detail}.")
            else:
                audit.add("FAIL", normalization_detail)

            if audit.status == "PASS":
                audit.add("PASS", "Required NPZ shape, labels, and sampling contract passed.")

    except Exception as exc:
        audit.add("FAIL", f"Could not read/audit NPZ: {type(exc).__name__}: {exc}")

    return audit


def compare_channel_orders(audits: Sequence[PatientAudit]) -> List[Finding]:
    findings: List[Finding] = []
    available = [audit for audit in audits if audit.channel_order_available]
    if not available:
        findings.append(
            Finding(
                "WARN",
                "No P20 NPZ stores channel names/order. Exact channel-order consistency remains unverified and must be traced to preprocessing code/TTL.",
            )
        )
        return findings

    reference = available[0].channel_order
    mismatches = [audit.patient_id for audit in available if audit.channel_order != reference]
    if mismatches:
        findings.append(Finding("FAIL", f"Channel-order mismatches found for: {mismatches}"))
    else:
        findings.append(
            Finding(
                "PASS",
                f"All {len(available)} NPZ files that store channel names use the same order.",
            )
        )
    if len(available) != len(audits):
        findings.append(
            Finding(
                "WARN",
                f"Only {len(available)}/{len(audits)} patient files store channel names.",
            )
        )
    return findings


def scan_preictal_evidence(root: Path, output_path: Path) -> int:
    patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"pre[-_ ]?ictal",
            r"prediction[_ -]?horizon",
            r"seizure[_ -]?prediction[_ -]?horizon",
            r"seizure[_ -]?occurrence[_ -]?period",
            r"post[-_ ]?ictal",
            r"exclusion[_ -]?period",
            r"seizure[_ -]?onset",
            r"label[_ -]?map",
            r"class[_ -]?map",
        )
    ]
    allowed_suffixes = {".py", ".yaml", ".yml", ".json", ".ttl", ".md", ".txt"}
    matches: List[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        if any(part in EXCLUDED_SCAN_DIRS for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(pattern.search(line) for pattern in patterns):
                relative = path.relative_to(root)
                matches.append(f"{relative}:{line_number}: {line.strip()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Pre-Ictal Definition Evidence Scan\n")
        handle.write("=================================\n\n")
        handle.write(
            "This is an evidence locator, not an automatic scientific conclusion. "
            "Review the matched preprocessing lines and write the final horizon, gap, "
            "post-ictal exclusion, and boundary policy into the data-contract report.\n\n"
        )
        if matches:
            handle.write("\n".join(matches))
            handle.write("\n")
        else:
            handle.write("NO MATCHES FOUND. The pre-ictal definition is not discoverable from scanned project text.\n")
    return len(matches)


def write_distribution_csv(audits: Sequence[PatientAudit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id",
        "split",
        "status",
        "windows",
        "interictal",
        "preictal",
        "ictal",
        "interictal_pct",
        "preictal_pct",
        "ictal_pct",
        "sfreq",
        "channels",
        "timepoints",
        "window_duration_s",
        "stored_step_s",
        "finite_scan",
        "channel_order_available",
        "npz_path",
        "ttl_path",
        "findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            total = max(audit.windows, 1)
            writer.writerow(
                {
                    "patient_id": audit.patient_id,
                    "split": audit.split,
                    "status": audit.status,
                    "windows": audit.windows,
                    "interictal": audit.interictal,
                    "preictal": audit.preictal,
                    "ictal": audit.ictal,
                    "interictal_pct": audit.interictal / total * 100.0,
                    "preictal_pct": audit.preictal / total * 100.0,
                    "ictal_pct": audit.ictal / total * 100.0,
                    "sfreq": audit.sfreq,
                    "channels": audit.channels,
                    "timepoints": audit.timepoints,
                    "window_duration_s": audit.window_duration_s,
                    "stored_step_s": audit.stored_step_s,
                    "finite_scan": audit.finite_scan,
                    "channel_order_available": audit.channel_order_available,
                    "npz_path": audit.npz_path,
                    "ttl_path": audit.ttl_path,
                    "findings": " | ".join(f"{item.severity}: {item.message}" for item in audit.findings),
                }
            )


def write_report(
    path: Path,
    split_path: Path,
    data_dir: Path,
    audits: Sequence[PatientAudit],
    global_findings: Sequence[Finding],
    evidence_matches: int,
    expected_step_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = np.array(
        [[a.interictal, a.preictal, a.ictal] for a in audits], dtype=np.int64
    ).sum(axis=0)
    statuses = {status: sum(a.status == status for a in audits) for status in ("PASS", "WARN", "FAIL")}
    global_fail = any(item.severity == "FAIL" for item in global_findings)
    overall = "FAIL" if statuses["FAIL"] or global_fail else ("WARN" if statuses["WARN"] or any(item.severity == "WARN" for item in global_findings) else "PASS")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("P20 EEG Data-Contract Integrity Audit\n")
        handle.write("=====================================\n\n")
        handle.write(f"Generated UTC      : {datetime.now(timezone.utc).isoformat()}\n")
        handle.write(f"Project root       : {PROJECT_ROOT}\n")
        handle.write(f"Processed data dir : {data_dir}\n")
        handle.write(f"Frozen split       : {split_path}\n")
        handle.write(f"Overall status     : {overall}\n\n")

        handle.write("Immutable class contract\n")
        handle.write("------------------------\n")
        for class_id, name in CLASS_NAMES.items():
            handle.write(f"{class_id} = {name}\n")
        handle.write("\n")

        handle.write("Global checks\n")
        handle.write("-------------\n")
        for finding in global_findings:
            handle.write(f"[{finding.severity}] {finding.message}\n")
        handle.write(f"[INFO] Pre-ictal evidence matches written: {evidence_matches}\n")
        handle.write(
            f"[INFO] Expected window contract: 20 channels × 512 samples, 128 Hz, 4 s duration, {expected_step_s:g} s step.\n\n"
        )

        handle.write("Patient summary\n")
        handle.write("---------------\n")
        handle.write(
            f"Patients audited: {len(audits)} | PASS={statuses['PASS']} WARN={statuses['WARN']} FAIL={statuses['FAIL']}\n"
        )
        handle.write(
            f"Total windows: {int(totals.sum())} | Interictal={int(totals[0])} | Pre-Ictal={int(totals[1])} | Ictal={int(totals[2])}\n\n"
        )

        for audit in audits:
            handle.write(
                f"[{audit.status}] {audit.patient_id} ({audit.split}) — "
                f"N={audit.windows}, labels={audit.interictal}/{audit.preictal}/{audit.ictal}, "
                f"shape=({audit.channels},{audit.timepoints}), sfreq={audit.sfreq:g}\n"
            )
            for finding in audit.findings:
                handle.write(f"    [{finding.severity}] {finding.message}\n")
            handle.write("\n")

        handle.write("Manual scientific fields still requiring confirmation\n")
        handle.write("------------------------------------------------------\n")
        handle.write("Pre-ictal duration/horizon : NOT AUTOMATICALLY CONCLUDED\n")
        handle.write("Prediction gap / SPH       : NOT AUTOMATICALLY CONCLUDED\n")
        handle.write("Post-ictal exclusion       : NOT AUTOMATICALLY CONCLUDED\n")
        handle.write("Clustered-seizure rule     : NOT AUTOMATICALLY CONCLUDED\n")
        handle.write("Boundary-window policy     : NOT AUTOMATICALLY CONCLUDED\n")
        handle.write(
            "Review reports/model_improvement/preictal_definition_evidence.txt and then replace these lines with evidence-backed values.\n\n"
        )

        handle.write("Interpretation\n")
        handle.write("--------------\n")
        handle.write(
            "PASS means the machine-checkable artifact contract passed. WARN means a field such as channel names or window step was not stored and therefore remains unverified. FAIL blocks new model-training claims until corrected.\n"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen P20 processed EEG contract.")
    parser.add_argument("--data-dir", default="data/processed/tusz")
    parser.add_argument("--split", default="experiments/splits/p20_frozen_split.json")
    parser.add_argument("--report-dir", default="reports/model_improvement")
    parser.add_argument("--expected-sfreq", type=float, default=128.0)
    parser.add_argument("--expected-channels", type=int, default=20)
    parser.add_argument("--expected-timepoints", type=int, default=512)
    parser.add_argument("--expected-step-s", type=float, default=2.0)
    parser.add_argument("--finite-scan", choices=("sample", "full"), default="sample")
    parser.add_argument("--sample-windows", type=int, default=32)
    args = parser.parse_args(argv)

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    split_path = (PROJECT_ROOT / args.split).resolve()
    report_dir = (PROJECT_ROOT / args.report_dir).resolve()

    if not data_dir.exists():
        print(f"[FAIL] Processed-data directory not found: {data_dir}", file=sys.stderr)
        return 2

    split, global_findings = _load_split(split_path)
    npz_index = _index_files(data_dir, ".npz")
    ttl_index = _index_files(data_dir, ".ttl")

    audits: List[PatientAudit] = []
    for split_name in ("train", "validation", "test"):
        for patient_id in split[split_name]:
            audits.append(
                audit_patient(
                    patient_id=patient_id,
                    split_name=split_name,
                    npz_path=_choose_patient_file(npz_index, patient_id),
                    ttl_path=_choose_patient_file(ttl_index, patient_id),
                    expected_sfreq=args.expected_sfreq,
                    expected_channels=args.expected_channels,
                    expected_timepoints=args.expected_timepoints,
                    expected_step_s=args.expected_step_s,
                    finite_mode=args.finite_scan,
                    sample_windows=args.sample_windows,
                )
            )

    global_findings.extend(compare_channel_orders(audits))

    evidence_path = report_dir / "preictal_definition_evidence.txt"
    evidence_matches = scan_preictal_evidence(PROJECT_ROOT, evidence_path)

    csv_path = report_dir / "patient_class_distribution.csv"
    report_path = report_dir / "data_contract_audit.txt"
    write_distribution_csv(audits, csv_path)
    write_report(
        report_path,
        split_path,
        data_dir,
        audits,
        global_findings,
        evidence_matches,
        args.expected_step_s,
    )

    fail_count = sum(audit.status == "FAIL" for audit in audits) + sum(
        finding.severity == "FAIL" for finding in global_findings
    )
    warn_count = sum(audit.status == "WARN" for audit in audits) + sum(
        finding.severity == "WARN" for finding in global_findings
    )

    print(f"Audit complete: {len(audits)} patients, {fail_count} failure(s), {warn_count} warning(s).")
    print(f"Report: {report_path}")
    print(f"CSV   : {csv_path}")
    print(f"Evidence: {evidence_path}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
