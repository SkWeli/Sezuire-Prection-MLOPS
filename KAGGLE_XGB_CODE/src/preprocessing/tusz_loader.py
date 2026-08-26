import sys
from pathlib import Path

import mne
import numpy as np


def parse_csv_seizures(csv_path):
    """Parse TUSZ *.csv_bi or *.csv → seizure intervals"""
    intervals = []

    csv_bi = Path(str(csv_path).replace('.csv', '.csv_bi'))
    parse_path = csv_bi if csv_bi.exists() else csv_path

    if not parse_path.exists():
        print(f"    ⚠️  No CSV annotation found: {parse_path.name} — labeling all background")
        return intervals

    try:
        # TUSZ CSV has comment lines starting with '#' before the header
        with open(parse_path, 'r') as f:
            lines = [l for l in f if not l.startswith('#') and l.strip()]

        import io
        import pandas as pd
        df = pd.read_csv(io.StringIO('\n'.join(lines)))
        df.columns = df.columns.str.strip().str.lower()

        seizure_labels = {
            'seiz',                                          # csv_bi
            'gnsz', 'fnsz', 'cpsz', 'spsz',                # focal/generalized
            'tcsz', 'tnsz', 'absz', 'atsz', 'mysz'         # tonic-clonic/absence
        }

        for _, row in df.iterrows():
            label = str(row.get('label', row.get('type', ''))).strip().lower()
            prob  = float(row.get('confidence', row.get('prob', 1.0)))
            if label in seizure_labels and prob >= 0.5:
                intervals.append((float(row['start_time']), float(row['stop_time'])))

        total_sec = sum(e - s for s, e in intervals)
        print(f"     CSV: {len(intervals)} seizure interval(s) ({total_sec:.1f}s) from {parse_path.name}")

    except Exception as e:
        print(f"    ⚠️  CSV parse error: {e} — labeling all background")

    return intervals


def label_window(window_start_sample, window_size_samples, sfreq, seizure_intervals, pre_ictal_seconds=900.0):
    """
    Return temporal class label for an EEG window:
    0 = Interictal (Background)
    1 = Pre-ictal (Impending seizure horizon)
    2 = Ictal (Active seizure)
    """
    win_start_sec = window_start_sample / sfreq
    win_end_sec   = (window_start_sample + window_size_samples) / sfreq
    win_dur       = win_end_sec - win_start_sec

    label = 0  # Default to Interictal

    for (sz_start, sz_end) in seizure_intervals:
        # 1. Check for Ictal overlap (Highest Priority)
        overlap_start = max(win_start_sec, sz_start)
        overlap_end   = min(win_end_sec,   sz_end)
        overlap_sec   = max(0.0, overlap_end - overlap_start)

        if overlap_sec / win_dur >= 0.5:   # ≥50% overlap → active seizure window
            return 2  # Immediate return, Ictal overrides everything
        
        # 2. Check for Pre-Ictal overlap
        pre_ictal_start = max(0.0, sz_start - pre_ictal_seconds)
        
        # If the window falls within the predictive horizon BEFORE the seizure
        if win_end_sec > pre_ictal_start and win_start_sec <= sz_start:
            label = 1  # Assign Pre-ictal, but continue loop in case it overlaps an earlier Ictal zone
            
    return label


def load_tusz_folder(tusz_folder, output_dir="data/processed"):
    """Process entire TUSZ patient folder → standardized EEG epochs"""
    tusz_folder = Path(tusz_folder)
    output_root = Path(output_dir) / "tusz" / tusz_folder.name
    output_root.mkdir(parents=True, exist_ok=True)

    patient_id = tusz_folder.name
    print(f"Processing TUSZ patient: {patient_id}")

    edf_files = sorted(tusz_folder.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")

    if not edf_files:
        print("❌ No EDF files found — check folder structure")
        return 1

    all_epochs    = []
    all_labels    = []
    final_ch_names = None
    final_sfreq   = 128

    for edf_file in edf_files:                        # ← ALL files, not just [:3]
        print(f"  → {edf_file.relative_to(tusz_folder)}")

        # ── Load EDF ────────────────────────────────────────────────────────
        raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

        try:
            raw.pick_types(eeg=True, verbose=False)
        except Exception:
            print("    Keeping all available channels")

        # ── Signal Cleaning ─────────────────────────────────────────────────
        raw.filter(0.5, 40, method="fir", fir_window="hamming", verbose=False)
        raw.notch_filter(50, method="fir", verbose=False)
        raw.set_eeg_reference("average", verbose=False)
        raw.resample(128, verbose=False)

        data  = raw.get_data()
        sfreq = int(raw.info["sfreq"])
        final_sfreq    = sfreq
        final_ch_names = raw.ch_names

        # ── Real Seizure Labels from TSE ────────────────────────────────────
        csv_file = edf_file.with_suffix('.csv')
        seizure_intervals = parse_csv_seizures(csv_file)

        # ── Sliding Window: 4s, 50% overlap ─────────────────────────────────
        window_sec        = 4
        step_sec          = 2
        n_samples_window  = int(sfreq * window_sec)   # 512
        n_samples_step    = int(sfreq * step_sec)     # 256

        file_epochs   = 0
        file_pre_ictal = 0
        file_ictal = 0

        # Add skipping logic for aborted recordings (from our previous fix)
        if raw.n_times < n_samples_window:
            print(f"    ⚠️ Skipping {edf_file.name}: recording too short ({raw.n_times} samples)")
            continue

        for i in range(0, data.shape[1] - n_samples_window, n_samples_step):
            window = data[:, i:i + n_samples_window]
            label  = label_window(i, n_samples_window, sfreq, seizure_intervals)

            all_epochs.append(window.astype(np.float32))
            all_labels.append(label)
            file_epochs   += 1
            if label == 1:
                file_pre_ictal += 1
            elif label == 2:
                file_ictal += 1

        print(f"    Added {file_epochs} windows ({file_pre_ictal} pre-ictal, {file_ictal} ictal)")

    if not all_epochs:
        print("❌ No windows created")
        return 1

    # ── Pad / Truncate to Fixed Shape (N, 20, 512) ───────────────────────────
    max_chans    = 20
    fixed_length = 512

    padded_epochs = []
    for window in all_epochs:
        chans, time = window.shape

        if chans < max_chans:
            window = np.pad(window, ((0, max_chans - chans), (0, 0)), 'constant')
        elif chans > max_chans:
            window = window[:max_chans, :]

        if time < fixed_length:
            window = np.pad(window, ((0, 0), (0, fixed_length - time)), 'constant')
        elif time > fixed_length:
            window = window[:, :fixed_length]

        padded_epochs.append(window)

    all_epochs = np.array(padded_epochs, dtype=np.float32)
    all_labels = np.array(all_labels,    dtype=np.int64)

    # Calculate exact counts for the NPZ metadata
    n_interictal = int((all_labels == 0).sum())
    n_pre_ictal  = int((all_labels == 1).sum())
    n_ictal      = int((all_labels == 2).sum())

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    out_file = output_root / f"{patient_id}.npz"
    np.savez_compressed(
        out_file,
        epochs       = all_epochs,
        labels       = all_labels,
        ch_names     = np.array(
            final_ch_names[:20] if final_ch_names else [f'ch{i}' for i in range(20)],
            dtype=object
        ),
        sfreq        = final_sfreq,
        patient_id   = patient_id,
        n_windows    = len(all_epochs),
        n_interictal = n_interictal,
        n_pre_ictal  = n_pre_ictal,
        n_ictal      = n_ictal
    )

    print(f"\n✅ SAVED: {out_file}")
    print(f"✅ Shape: {all_epochs.shape}")
    print(f"✅ Classes: {n_interictal} Interictal | {n_pre_ictal} Pre-Ictal | {n_ictal} Ictal")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tusz_loader.py <tusz_patient_folder> [output_dir]")
        sys.exit(1)

    tusz_folder = sys.argv[1]
    output_dir  = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    sys.exit(load_tusz_folder(tusz_folder, output_dir))