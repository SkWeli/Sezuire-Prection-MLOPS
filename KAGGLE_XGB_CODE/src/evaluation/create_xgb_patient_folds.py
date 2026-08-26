from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


FEATURE_DIR = Path("data/features/tusz_xgb_220")

OUTPUT_FILE = Path(
    "experiments/xgboost/patient_cv_folds.json"
)

N_SPLITS = 5
SEED = 42


def main():

    files = sorted(
        FEATURE_DIR.glob("*_features.npz")
    )

    if len(files) != 40:
        raise RuntimeError(
            f"Expected 40 patient files, found {len(files)}"
        )

    all_labels = []
    all_groups = []

    patient_stats = {}

    print("[INFO] Reading patient labels...")

    for file in files:

        with np.load(file, allow_pickle=False) as data:

            labels = data["y"].astype(np.int64)

            patient_id = str(data["patient_id"])

        counts = np.bincount(
            labels,
            minlength=3,
        )

        patient_stats[patient_id] = {
            "windows": int(len(labels)),
            "interictal": int(counts[0]),
            "pre_ictal": int(counts[1]),
            "ictal": int(counts[2]),
        }

        all_labels.append(labels)

        all_groups.append(
            np.full(
                len(labels),
                patient_id,
                dtype=object,
            )
        )

    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)

    print()
    print("[INFO] Total windows:", len(y))

    print(
        "[INFO] Global counts:",
        np.bincount(y, minlength=3).tolist(),
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    dummy_X = np.zeros(
        (len(y), 1),
        dtype=np.uint8,
    )

    fold_data = []

    validation_patient_occurrences = {}

    print()
    print("=" * 80)
    print("5-FOLD PATIENT-LEVEL CROSS-VALIDATION")
    print("=" * 80)

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(
            dummy_X,
            y,
            groups,
        ),
        start=1,
    ):

        train_patients = sorted(
            np.unique(groups[train_idx]).tolist()
        )

        val_patients = sorted(
            np.unique(groups[val_idx]).tolist()
        )

        overlap = set(train_patients) & set(val_patients)

        if overlap:
            raise RuntimeError(
                f"Patient leakage in Fold {fold_number}: {overlap}"
            )

        val_labels = y[val_idx]

        train_labels = y[train_idx]

        train_counts = np.bincount(
            train_labels,
            minlength=3,
        )

        val_counts = np.bincount(
            val_labels,
            minlength=3,
        )

        for patient_id in val_patients:
            validation_patient_occurrences[patient_id] = (
                validation_patient_occurrences.get(patient_id, 0)
                + 1
            )

        val_windows = len(val_idx)

        val_percentages = (
            val_counts / val_windows * 100
        )

        print()
        print(f"FOLD {fold_number}")
        print("-" * 80)

        print(
            f"Train patients : {len(train_patients)}"
        )

        print(
            f"Held-out patients: {len(val_patients)}"
        )

        print(
            f"Train windows  : {len(train_idx):,}"
        )

        print(
            f"Held-out windows: {len(val_idx):,}"
        )

        print()
        print("Held-out class counts")

        print(
            f"Interictal : {val_counts[0]:,} "
            f"({val_percentages[0]:.2f}%)"
        )

        print(
            f"Pre-Ictal  : {val_counts[1]:,} "
            f"({val_percentages[1]:.2f}%)"
        )

        print(
            f"Ictal       : {val_counts[2]:,} "
            f"({val_percentages[2]:.2f}%)"
        )

        print()
        print(
            "Held-out patients:",
            ", ".join(val_patients),
        )

        fold_data.append(
            {
                "fold": fold_number,

                "train_patients": train_patients,

                "held_out_patients": val_patients,

                "train_windows": int(len(train_idx)),

                "held_out_windows": int(len(val_idx)),

                "train_counts": {
                    "interictal": int(train_counts[0]),
                    "pre_ictal": int(train_counts[1]),
                    "ictal": int(train_counts[2]),
                },

                "held_out_counts": {
                    "interictal": int(val_counts[0]),
                    "pre_ictal": int(val_counts[1]),
                    "ictal": int(val_counts[2]),
                },
            }
        )

    # Every patient must be held out exactly once.
    if len(validation_patient_occurrences) != 40:
        raise RuntimeError(
            "Not all 40 patients appeared in held-out folds."
        )

    bad = {
        patient: count
        for patient, count
        in validation_patient_occurrences.items()
        if count != 1
    }

    if bad:
        raise RuntimeError(
            f"Patients not held out exactly once: {bad}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "method": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "seed": SEED,

        "grouping_rule": (
            "Patient IDs are mutually exclusive "
            "between training and held-out data."
        ),

        "patient_stats": patient_stats,

        "folds": fold_data,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("[PASS] NO PATIENT LEAKAGE")
    print("[PASS] ALL 40 PATIENTS HELD OUT EXACTLY ONCE")
    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()