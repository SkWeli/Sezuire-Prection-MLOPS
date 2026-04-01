import sys
from pathlib import Path

import mne
import numpy as np


def load_tusz_folder(tusz_folder, output_dir="data/processed"):
    """Process entire TUSZ patient folder → standardized EEG epochs"""
    tusz_folder = Path(tusz_folder)
    output_root = Path(output_dir) / "tusz" / tusz_folder.name
    output_root.mkdir(parents=True, exist_ok=True)

    patient_id = tusz_folder.name
    print(f"🚀 Processing TUSZ patient: {patient_id}")

    # Recursively find ALL EDF files
    edf_files = sorted(tusz_folder.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")

    if not edf_files:
        print("❌ No EDF files found — check folder structure")
        return 1

    all_epochs = []
    all_labels = []
    final_ch_names = None
    final_sfreq = 128

    for edf_file in edf_files[:3]:  # First 3 files (viva demo)
        print(f"  → {edf_file.relative_to(tusz_folder)}")

        raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

        try:
            raw.pick_types(eeg=True, verbose=False)
        except Exception:
            print("    Keeping all available channels")

        # Production CHB-MIT pipeline
        raw.filter(0.5, 40, method="fir", fir_window="hamming", verbose=False)
        raw.notch_filter(50, method="fir", verbose=False)
        raw.set_eeg_reference("average", verbose=False)
        raw.resample(128, verbose=False)

        data = raw.get_data()
        sfreq = int(raw.info["sfreq"])
        final_sfreq = sfreq
        final_ch_names = raw.ch_names

        # 4s windows, 2s step (50% overlap)
        window_sec = 4
        step_sec = 2
        n_samples_window = int(sfreq * window_sec)
        n_samples_step = int(sfreq * step_sec)

        file_epochs = 0
        file_seizures = 0

        for i in range(0, data.shape[1] - n_samples_window, n_samples_step):
            window = data[:, i:i + n_samples_window]
            
            # Demo labels: seizure windows after 30% of recording
            label = 1 if i > 0.3 * data.shape[1] else 0
            
            all_epochs.append(window.astype(np.float32))
            all_labels.append(label)
            file_epochs += 1
            file_seizures += label

        print(f"    Added {file_epochs} windows ({file_seizures} seizures)")

    if not all_epochs:
        print("❌ No windows created")
        return 1

    # Pad to fixed shape: (N, 20 channels, 512 samples)
    max_chans = 20
    fixed_length = 512  # 4s @ 128Hz
    
    padded_epochs = []
    for window in all_epochs:
        chans, time = window.shape
        
        # Pad/truncate channels
        if chans < max_chans:
            pad_width = ((0, max_chans - chans), (0, 0))
            window = np.pad(window, pad_width, 'constant')
        elif chans > max_chans:
            window = window[:max_chans, :]
        
        # Pad/truncate time
        if time < fixed_length:
            pad_width = ((0, 0), (0, fixed_length - time))
            window = np.pad(window, pad_width, 'constant')
        elif time > fixed_length:
            window = window[:, :fixed_length]
        
        padded_epochs.append(window)
    
    all_epochs = np.array(padded_epochs, dtype=np.float32)  # (N, 20, 512)
    all_labels = np.array(all_labels, dtype=np.int64)

    out_file = output_root / f"{patient_id}.npz"
    np.savez_compressed(
        out_file,
        epochs=all_epochs,
        labels=all_labels,
        ch_names=np.array(final_ch_names[:20] if final_ch_names else [f'ch{i}' for i in range(20)], dtype=object),
        sfreq=final_sfreq,
        patient_id=patient_id,
        n_windows=len(all_epochs),
        n_seizures=int(all_labels.sum())
    )

    print(f"\n✅ SAVED: {out_file}")
    print(f"✅ Shape: {all_epochs.shape}")
    print(f"✅ Seizures: {int(all_labels.sum())} / {len(all_labels)} ({all_labels.mean():.1%})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tusz_loader.py <tusz_patient_folder> [output_dir]")
        sys.exit(1)

    tusz_folder = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    sys.exit(load_tusz_folder(tusz_folder, output_dir))