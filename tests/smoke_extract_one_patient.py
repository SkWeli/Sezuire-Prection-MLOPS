import time
from pathlib import Path

import numpy as np

from src.features.eeg_feature_extractor import (
    extract_window_features,
    build_feature_names,
)


PATIENT_FILE = Path(
    r"E:\FYP\eeg-seizure-mlops\data\processed\tusz\aaaaaaaq\aaaaaaaq.npz"
)


def main():
    print("[INFO] Loading:", PATIENT_FILE)

    data = np.load(PATIENT_FILE, allow_pickle=True)

    epochs = data["epochs"]
    labels = data["labels"]
    sfreq = float(data["sfreq"])
    patient_id = str(data["patient_id"])
    ch_names = [str(x) for x in data["ch_names"]]

    feature_names = build_feature_names(ch_names)

    print("[INFO] Patient:", patient_id)
    print("[INFO] EEG shape:", epochs.shape)
    print("[INFO] Labels shape:", labels.shape)
    print("[INFO] Sampling rate:", sfreq)
    print("[INFO] Number of features:", len(feature_names))

    features = np.empty(
        (len(epochs), len(feature_names)),
        dtype=np.float32,
    )

    start = time.time()

    for i, window in enumerate(epochs):
        features[i] = extract_window_features(
            window,
            sfreq=sfreq,
        )

        if (i + 1) % 500 == 0:
            print(
                f"[INFO] Extracted {i + 1}/{len(epochs)} windows"
            )

    elapsed = time.time() - start

    class_counts = np.bincount(
        labels,
        minlength=3,
    )

    print("\n========== RESULT ==========")
    print("Patient:", patient_id)
    print("Feature matrix:", features.shape)
    print("Feature dtype:", features.dtype)
    print("Labels:", labels.shape)
    print("Finite:", np.isfinite(features).all())

    print("\nClass counts")
    print("Interictal:", int(class_counts[0]))
    print("Pre-Ictal :", int(class_counts[1]))
    print("Ictal     :", int(class_counts[2]))

    print("\nNPZ metadata")
    print("n_windows    :", int(data["n_windows"]))
    print("n_interictal :", int(data["n_interictal"]))
    print("n_pre_ictal  :", int(data["n_pre_ictal"]))
    print("n_ictal      :", int(data["n_ictal"]))

    print("\nFeature statistics")
    print("Minimum:", float(features.min()))
    print("Maximum:", float(features.max()))
    print("Mean   :", float(features.mean()))
    print("Std    :", float(features.std()))

    print("\nElapsed seconds:", round(elapsed, 2))

    counts_match = (
        len(labels) == int(data["n_windows"])
        and class_counts[0] == int(data["n_interictal"])
        and class_counts[1] == int(data["n_pre_ictal"])
        and class_counts[2] == int(data["n_ictal"])
    )

    passed = (
        features.shape == (len(epochs), 220)
        and np.isfinite(features).all()
        and counts_match
    )

    if passed:
        print("\n[PASS] Full patient feature extraction")
    else:
        print("\n[FAIL] Full patient feature extraction")


if __name__ == "__main__":
    main()