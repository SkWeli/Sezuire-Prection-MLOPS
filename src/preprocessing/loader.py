"""
src/preprocessing/loader.py
────────────────────────────
Loads CHB-MIT EDF files with MNE-Python, applies the standard preprocessing
chain, extracts labelled epochs, and saves them as compressed NumPy archives.
Every processing step is recorded as an RDF triple and written to a Turtle
sidecar file so the SHACL validator can confirm pipeline provenance.

Processing chain (in order)
----------------------------
1. Load EDF via MNE (no preload — stream-safe)
2. Pick EEG channels only
3. Bandpass filter 0.5–40 Hz  (FIR, zero-phase, hamming window)
4. Notch filter 50 Hz          (FIR, zero-phase)
5. Re-reference to common average
6. Parse seizure annotations from CHB-MIT *-summary.txt
7. Slide 4-second, 50%-overlap windows across the recording
8. Label each window: ictal / pre_ictal / interictal / unknown
9. Save epochs to data/processed/chbmit/<subject>/<session>.npz
10. Write RDF provenance sidecar to same directory

Usage (standalone)
------------------
    python -m src.preprocessing.loader \
        --edf  data/raw/chbmit/chb01/chb01_03.edf \
        --summary data/raw/chbmit/chb01/chb01-summary.txt \
        --out  data/processed/chbmit/chb01

Label definitions
-----------------
    ictal        — window overlaps a seizure annotation
    pre_ictal    — window ends within [30, 120] s before seizure onset
    interictal   — window is > 4 hours from every seizure boundary
    unknown      — everything else (peri-ictal buffer)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import mne
import numpy as np
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD


# Logging
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("loader")


# Constants — all tunable via config; defaults match project spec

BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 40.0
NOTCH_HZ: float = 50.0
EPOCH_DURATION_S: float = 4.0          # window length in seconds
EPOCH_OVERLAP_FRAC: float = 0.50       # 50 % overlap → step = 2 s
PRE_ICTAL_MIN_S: float = 30.0          # seconds before onset
PRE_ICTAL_MAX_S: float = 120.0         # seconds before onset
INTERICTAL_BUFFER_S: float = 4 * 3600  # 4 hours in seconds

# RDF namespace — must match ontology/eeg_epilepsy.ttl
EEG = Namespace("http://example.org/eeg-epilepsy#")



# Data structures

class SeizureAnnotation(NamedTuple):
    onset: float    # seconds from EDF start
    offset: float   # seconds from EDF start



# Seizure summary parser

def parse_chbmit_summary(
    summary_path: Path,
    edf_name: str,
) -> list[SeizureAnnotation]:
    """
    Parse a CHB-MIT *-summary.txt file and return the seizure annotations
    that belong to *edf_name* (e.g. ``"chb01_03.edf"``).

    The summary format is:

        File Name: chb01_03.edf
        File Start Time: 13:43:04
        File End Time: 14:43:04
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds

    Handles files with zero seizures gracefully.
    """
    if not summary_path.is_file():
        log.warning("Summary file not found: %s — treating as seizure-free", summary_path)
        return []

    text = summary_path.read_text(encoding="utf-8", errors="replace")

    # Split into per-file blocks on "File Name:" boundaries
    blocks = re.split(r"(?=File Name:)", text, flags=re.IGNORECASE)

    target = edf_name.lower().strip()
    for block in blocks:
        name_match = re.search(r"File Name:\s*(\S+)", block, re.IGNORECASE)
        if not name_match:
            continue
        if name_match.group(1).lower().strip() != target:
            continue

        # How many seizures?
        n_match = re.search(r"Number of Seizures in File:\s*(\d+)", block, re.IGNORECASE)
        n_seizures = int(n_match.group(1)) if n_match else 0
        if n_seizures == 0:
            return []

        onsets  = [float(m) for m in re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", block, re.IGNORECASE)]
        offsets = [float(m) for m in re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)",   block, re.IGNORECASE)]

        if len(onsets) != n_seizures or len(offsets) != n_seizures:
            log.warning(
                "%s: expected %d seizures but parsed %d onset / %d offset entries — "
                "using what was found",
                edf_name, n_seizures, len(onsets), len(offsets),
            )

        annotations = []
        for on, off in zip(onsets, offsets):
            if on >= off:
                log.error(
                    "%s: skipping invalid annotation onset=%.1f >= offset=%.1f",
                    edf_name, on, off,
                )
                continue
            annotations.append(SeizureAnnotation(onset=on, offset=off))

        return annotations

    log.info("No block found for '%s' in summary — treating as seizure-free", edf_name)
    return []



# Epoch labelling

def _label_window(
    t_start: float,
    t_end: float,
    seizures: list[SeizureAnnotation],
) -> str:
    """
    Assign one of four labels to a [t_start, t_end) window.

    Priority order (first match wins):
        1. ictal       — window overlaps any seizure
        2. pre_ictal   — window ends in (onset-120 s, onset-30 s]
        3. interictal  — window is > 4 h from every seizure boundary
        4. unknown     — peri-ictal buffer or not enough recording
    """
    if not seizures:
        # No seizures in file: every window is interictal by default
        # (caller's responsibility to check cross-file 4-hour constraint)
        return "interictal"

    # 1. Ictal — any overlap
    for sz in seizures:
        if t_start < sz.offset and t_end > sz.onset:
            return "ictal"

    # 2. Pre-ictal — window ENDS in the pre-ictal zone before some seizure onset
    for sz in seizures:
        gap = sz.onset - t_end          # seconds between window end and onset
        if PRE_ICTAL_MIN_S <= gap <= PRE_ICTAL_MAX_S:
            return "pre_ictal"

    # 3. Interictal — > 4 h from every seizure boundary
    for sz in seizures:
        dist = min(
            abs(t_start - sz.onset),
            abs(t_start - sz.offset),
            abs(t_end   - sz.onset),
            abs(t_end   - sz.offset),
        )
        if dist <= INTERICTAL_BUFFER_S:
            return "unknown"     # too close — peri-ictal buffer

    return "interictal"



# RDF provenance helpers

def _make_rdf_graph(session_id: str) -> Graph:
    """Initialise an rdflib Graph with the EEG namespace binding."""
    g = Graph()
    g.bind("eeg", EEG)
    g.bind("xsd", XSD)

    session_uri = URIRef(EEG[session_id])
    g.add((session_uri, RDF.type, EEG.RecordingSession))
    g.add((session_uri, EEG.sessionID, Literal(session_id, datatype=XSD.string)))
    return g


def _add_step(
    g: Graph,
    session_id: str,
    step_order: int,
    filter_type: str,
    description: str,
) -> None:
    """
    Add a PreprocessingStep individual linked to the RecordingSession.

    The step URI is deterministic so reruns produce identical RDF output
    (important for DVC reproducibility).
    """
    session_uri = URIRef(EEG[session_id])
    step_id = f"{session_id}_step{step_order:02d}"
    step_uri = URIRef(EEG[step_id])

    g.add((step_uri, RDF.type,        EEG.PreprocessingStep))
    g.add((step_uri, EEG.stepOrder,   Literal(step_order,  datatype=XSD.integer)))
    g.add((step_uri, EEG.filterType,  Literal(filter_type, datatype=XSD.string)))
    g.add((step_uri, RDFS.comment,    Literal(description,  datatype=XSD.string)))
    g.add((session_uri, EEG.hasPreprocessingStep, step_uri))


def _add_channel(g: Graph, session_id: str, label: str, ch_index: int) -> None:
    """Add an EEGChannel individual for the given channel label."""
    session_uri = URIRef(EEG[session_id])
    safe_label = re.sub(r"[^A-Za-z0-9_]", "_", label)
    ch_uri = URIRef(EEG[f"{session_id}_ch_{safe_label}_{ch_index}"])
    g.add((ch_uri, RDF.type, EEG.EEGChannel))
    g.add((ch_uri, EEG.channelLabel, Literal(label, datatype=XSD.string)))
    g.add((session_uri, EEG.hasChannel, ch_uri))


def _add_sampling_rate(g: Graph, session_id: str, sfreq: float) -> None:
    """Add a SamplingRate individual linked to the RecordingSession."""
    session_uri = URIRef(EEG[session_id])
    sr_id = f"{session_id}_sr"
    sr_uri = URIRef(EEG[sr_id])
    g.add((sr_uri, RDF.type, EEG.SamplingRate))
    g.add((sr_uri, EEG.frequencyHz, Literal(float(sfreq), datatype=XSD.float)))
    g.add((session_uri, EEG.hasSamplingRate, sr_uri))


def _add_seizure_event(
    g: Graph,
    session_id: str,
    sz: SeizureAnnotation,
    sz_index: int,
) -> None:
    """Add a SeizureEvent individual (with a PreIctalWindow) to the graph."""
    session_uri = URIRef(EEG[session_id])

    sz_id = f"{session_id}_sz{sz_index:02d}"
    sz_uri = URIRef(EEG[sz_id])
    g.add((sz_uri, RDF.type,       EEG.SeizureEvent))
    g.add((sz_uri, EEG.hasOnset,   Literal(float(sz.onset),  datatype=XSD.float)))
    g.add((sz_uri, EEG.hasOffset,  Literal(float(sz.offset), datatype=XSD.float)))
    g.add((session_uri, EEG.hasSeizureEvent, sz_uri))

    # Pre-ictal window: clamp to [30, 120] s; use 60 s as default
    piw_duration = min(PRE_ICTAL_MAX_S, max(PRE_ICTAL_MIN_S, 60.0))
    piw_start = max(0.0, sz.onset - piw_duration)
    piw_end   = sz.onset

    piw_id  = f"{session_id}_sz{sz_index:02d}_piw"
    piw_uri = URIRef(EEG[piw_id])
    g.add((piw_uri, RDF.type,             EEG.PreIctalWindow))
    g.add((piw_uri, EEG.windowDuration,   Literal(piw_duration, datatype=XSD.float)))
    g.add((piw_uri, EEG.windowStart,      Literal(piw_start,    datatype=XSD.float)))
    g.add((piw_uri, EEG.windowEnd,        Literal(piw_end,      datatype=XSD.float)))
    g.add((sz_uri, EEG.hasPreIctalWindow, piw_uri))



# Core loader

def load_and_preprocess(
    edf_path: Path,
    summary_path: Path,
    out_dir: Path,
    *,
    bandpass_low: float = BANDPASS_LOW_HZ,
    bandpass_high: float = BANDPASS_HIGH_HZ,
    notch_hz: float = NOTCH_HZ,
    epoch_duration: float = EPOCH_DURATION_S,
    epoch_overlap: float = EPOCH_OVERLAP_FRAC,
) -> Path:
    """
    Full preprocessing pipeline for a single CHB-MIT EDF file.

    Parameters
    ----------
    edf_path      : Path to the raw .edf file.
    summary_path  : Path to the subject *-summary.txt annotation file.
    out_dir       : Directory where .npz and .ttl outputs are written.
    bandpass_low  : Lower bandpass cutoff in Hz.
    bandpass_high : Upper bandpass cutoff in Hz.
    notch_hz      : Notch filter frequency in Hz.
    epoch_duration: Epoch window length in seconds.
    epoch_overlap : Fractional overlap between consecutive windows.

    Returns
    -------
    Path to the saved .npz file.
    """
    edf_path = edf_path.resolve()
    out_dir  = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = edf_path.stem          # e.g. "chb01_03"
    edf_name   = edf_path.name          # e.g. "chb01_03.edf"
    log.info("▶ Processing %s", edf_name)

    # Start RDF provenance graph 
    rdf_g = _make_rdf_graph(session_id)
    step_counter = 1

    # 1. Load EDF 
    log.info("  [1/7] Loading EDF …")
    raw: mne.io.BaseRaw = mne.io.read_raw_edf(
        str(edf_path),
        preload=False,       # stream-safe; loaded on demand during filtering
        verbose=False,
    )
    sfreq = raw.info["sfreq"]
    log.info("       sfreq=%.1f Hz, n_channels=%d, duration=%.1f s",
             sfreq, len(raw.ch_names), raw.times[-1])

    _add_step(rdf_g, session_id, step_counter,
              "load",
              f"Loaded EDF via MNE: sfreq={sfreq:.1f} Hz, "
              f"n_channels={len(raw.ch_names)}, duration={raw.times[-1]:.1f} s")
    step_counter += 1

    # 2. Pick EEG channels 
    log.info("  [2/7] Picking EEG channels …")
    raw.load_data(verbose=False)
    try:
        raw.pick_types(eeg=True, verbose=False)
    except Exception:
        # CHB-MIT files sometimes have no channel-type annotations; pick all
        log.warning("       pick_types(eeg=True) failed — keeping all channels")

    ch_names = raw.ch_names
    log.info("       %d EEG channels retained", len(ch_names))

    _add_step(rdf_g, session_id, step_counter,
              "channel_selection",
              f"Picked EEG channels: {len(ch_names)} channels retained")
    step_counter += 1

    # Add channel individuals to the RDF graph
    _add_sampling_rate(rdf_g, session_id, sfreq)
    for idx, label in enumerate(ch_names):
        _add_channel(rdf_g, session_id, label, idx)

    # 3. Bandpass filter 0.5–40 Hz 
    log.info("  [3/7] Bandpass filter %.1f–%.1f Hz (FIR, zero-phase) …",
             bandpass_low, bandpass_high)
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        method="fir",
        fir_window="hamming",
        phase="zero",
        verbose=False,
    )
    _add_step(rdf_g, session_id, step_counter,
              "bandpass",
              f"FIR zero-phase bandpass filter: {bandpass_low}–{bandpass_high} Hz, "
              "window=hamming")
    step_counter += 1

    # 4. Notch filter 50 Hz ───
    log.info("  [4/7] Notch filter %.1f Hz (FIR, zero-phase) …", notch_hz)
    raw.notch_filter(
        freqs=notch_hz,
        method="fir",
        phase="zero",
        verbose=False,
    )
    _add_step(rdf_g, session_id, step_counter,
              "notch",
              f"FIR zero-phase notch filter at {notch_hz} Hz (power-line interference)")
    step_counter += 1

    # 5. Common average reference 
    log.info("  [5/7] Re-referencing to common average …")
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    _add_step(rdf_g, session_id, step_counter,
              "rereferencing",
              "Re-referenced to common average (CAR)")
    step_counter += 1

    # 6. Parse seizure annotations 
    log.info("  [6/7] Parsing seizure annotations from %s …", summary_path.name)
    seizures = parse_chbmit_summary(summary_path, edf_name)
    log.info("       %d seizure(s) found", len(seizures))

    for i, sz in enumerate(seizures):
        log.info("       [%d] onset=%.1f s  offset=%.1f s  duration=%.1f s",
                 i + 1, sz.onset, sz.offset, sz.offset - sz.onset)
        _add_seizure_event(rdf_g, session_id, sz, i + 1)

    _add_step(rdf_g, session_id, step_counter,
              "annotation_parsing",
              f"Parsed {len(seizures)} seizure annotation(s) from CHB-MIT summary file")
    step_counter += 1

    # 7. Slide epochs and label 
    log.info("  [7/7] Extracting 4-second epochs (50%% overlap) …")
    data = raw.get_data()                          # shape: (n_channels, n_samples)
    n_samples_epoch = int(epoch_duration * sfreq)
    step_samples    = int(n_samples_epoch * (1.0 - epoch_overlap))
    n_samples_total = data.shape[1]

    epochs_data:   list[np.ndarray] = []
    epochs_labels: list[str]        = []
    epochs_t_start: list[float]     = []

    window_start = 0
    while window_start + n_samples_epoch <= n_samples_total:
        window_end = window_start + n_samples_epoch
        t_start = window_start / sfreq
        t_end   = window_end   / sfreq

        label = _label_window(t_start, t_end, seizures)
        epochs_data.append(data[:, window_start:window_end].astype(np.float32))
        epochs_labels.append(label)
        epochs_t_start.append(t_start)

        window_start += step_samples

    epochs_arr    = np.stack(epochs_data, axis=0)   # (N, C, T)
    labels_arr    = np.array(epochs_labels)
    t_starts_arr  = np.array(epochs_t_start, dtype=np.float32)

    label_counts = {lbl: int((labels_arr == lbl).sum())
                    for lbl in ("ictal", "pre_ictal", "interictal", "unknown")}
    log.info(
        "       %d epochs total — ictal=%d  pre_ictal=%d  interictal=%d  unknown=%d",
        len(epochs_labels),
        label_counts["ictal"],
        label_counts["pre_ictal"],
        label_counts["interictal"],
        label_counts["unknown"],
    )

    _add_step(rdf_g, session_id, step_counter,
              "epoch_extraction",
              f"Extracted {len(epochs_labels)} epochs: duration={epoch_duration}s, "
              f"overlap={int(epoch_overlap*100)}%; "
              f"ictal={label_counts['ictal']}, "
              f"pre_ictal={label_counts['pre_ictal']}, "
              f"interictal={label_counts['interictal']}, "
              f"unknown={label_counts['unknown']}")
    step_counter += 1

    # Save .npz 
    npz_path = out_dir / f"{session_id}.npz"
    np.savez_compressed(
        str(npz_path),
        epochs=epochs_arr,           # float32 (N, C, T)
        labels=labels_arr,           # str     (N,)
        t_starts=t_starts_arr,       # float32 (N,)
        ch_names=np.array(ch_names),
        sfreq=np.float32(sfreq),
        edf_path=str(edf_path),
        session_id=session_id,
    )
    log.info("  💾 Saved epochs → %s", npz_path)

    # Compute MD5 for reproducibility check 
    md5 = hashlib.md5(npz_path.read_bytes()).hexdigest()
    log.info("  MD5: %s", md5)

    # Save RDF sidecar .ttl 
    # Add file-level metadata
    session_uri = URIRef(EEG[session_id])
    rdf_g.add((session_uri, EEG.outputPath,
                Literal(str(npz_path), datatype=XSD.string)))
    rdf_g.add((session_uri, EEG.outputMD5,
                Literal(md5, datatype=XSD.string)))
    rdf_g.add((session_uri, EEG.processedAt,
                Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.string)))

    ttl_path = out_dir / f"{session_id}.ttl"
    rdf_g.serialize(destination=str(ttl_path), format="turtle")
    log.info("  📄 Saved RDF provenance → %s", ttl_path)

    return npz_path



# CLI entry point

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loader",
        description="Preprocess a single CHB-MIT EDF file and save epochs as .npz.",
    )
    p.add_argument("--edf",     type=Path, required=True,
                   help="Path to the raw .edf file.")
    p.add_argument("--summary", type=Path, required=True,
                   help="Path to the CHB-MIT *-summary.txt annotation file.")
    p.add_argument("--out",     type=Path, required=True,
                   help="Output directory for .npz and .ttl files.")
    p.add_argument("--bandpass-low",  type=float, default=BANDPASS_LOW_HZ)
    p.add_argument("--bandpass-high", type=float, default=BANDPASS_HIGH_HZ)
    p.add_argument("--notch-hz",      type=float, default=NOTCH_HZ)
    p.add_argument("--epoch-duration", type=float, default=EPOCH_DURATION_S)
    p.add_argument("--epoch-overlap",  type=float, default=EPOCH_OVERLAP_FRAC)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        load_and_preprocess(
            edf_path=args.edf,
            summary_path=args.summary,
            out_dir=args.out,
            bandpass_low=args.bandpass_low,
            bandpass_high=args.bandpass_high,
            notch_hz=args.notch_hz,
            epoch_duration=args.epoch_duration,
            epoch_overlap=args.epoch_overlap,
        )
    except Exception as exc:
        log.error("Processing failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())