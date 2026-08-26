from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SUMMARY_FILE = Path(
    "data/features/tusz_xgb_220/feature_extraction_summary.csv"
)

OUTPUT_FILE = Path(
    "experiments/xgboost/patient_cv_folds_balanced_v1.json"
)

N_FOLDS = 5
PATIENTS_PER_FOLD = 8

SEED = 42

# Random candidate allocations to examine.
N_CANDIDATES = 200_000


def load_patient_stats():

    rows = []

    with SUMMARY_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "n_windows": int(row["n_windows"]),
                    "interictal": int(row["n_interictal"]),
                    "pre_ictal": int(row["n_pre_ictal"]),
                    "ictal": int(row["n_ictal"]),
                }
            )

    if len(rows) != 40:
        raise RuntimeError(
            f"Expected 40 patients, got {len(rows)}"
        )

    return rows


def candidate_score(
    assignment,
    counts,
    windows,
):

    fold_counts = []
    fold_windows = []

    preictal_patient_counts = []
    ictal_patient_counts = []

    for fold in assignment:

        fc = counts[fold].sum(axis=0)

        fw = windows[fold].sum()

        fold_counts.append(fc)
        fold_windows.append(fw)

        preictal_patient_counts.append(
            int(np.sum(counts[fold, 1] > 0))
        )

        ictal_patient_counts.append(
            int(np.sum(counts[fold, 2] > 0))
        )

    fold_counts = np.asarray(
        fold_counts,
        dtype=np.float64,
    )

    fold_windows = np.asarray(
        fold_windows,
        dtype=np.float64,
    )

    # Every fold must contain both minority classes.
    if np.any(fold_counts[:, 1] == 0):
        return np.inf

    if np.any(fold_counts[:, 2] == 0):
        return np.inf

    global_counts = counts.sum(axis=0).astype(
        np.float64
    )

    global_prop = (
        global_counts / global_counts.sum()
    )

    fold_prop = (
        fold_counts
        / fold_counts.sum(axis=1, keepdims=True)
    )

    # Compare fold class composition with global composition.
    #
    # Relative error is used because Pre-Ictal and Ictal
    # are much less common than Interictal.
    relative_error = (
        (fold_prop - global_prop)
        / np.maximum(global_prop, 1e-12)
    )

    class_weights = np.array(
        [0.25, 1.0, 1.0],
        dtype=np.float64,
    )

    class_score = np.mean(
        np.sum(
            class_weights
            * relative_error**2,
            axis=1,
        )
    )

    # Encourage similar numbers of patients contributing
    # Pre-Ictal and Ictal examples to each fold.
    minority_presence_score = (
        np.var(preictal_patient_counts)
        + np.var(ictal_patient_counts)
    )

    # Window equality is only a weak preference.
    #
    # aaaaahie alone contains more than half of all windows,
    # so forcing equal window totals would be impossible
    # without violating the patient-independence rule.
    target_windows = windows.sum() / N_FOLDS

    log_window_error = (
        np.log1p(fold_windows)
        - np.log1p(target_windows)
    )

    window_score = np.mean(
        log_window_error**2
    )

    total_score = (
        class_score
        + 0.05 * minority_presence_score
        + 0.05 * window_score
    )

    return float(total_score)


def main():

    rows = load_patient_stats()

    patient_ids = np.array(
        [r["patient_id"] for r in rows],
        dtype=object,
    )

    windows = np.array(
        [r["n_windows"] for r in rows],
        dtype=np.int64,
    )

    counts = np.array(
        [
            [
                r["interictal"],
                r["pre_ictal"],
                r["ictal"],
            ]
            for r in rows
        ],
        dtype=np.int64,
    )

    global_counts = counts.sum(axis=0)

    print("=" * 80)
    print("BALANCED PATIENT-LEVEL 5-FOLD SEARCH")
    print("=" * 80)

    print("Patients :", len(patient_ids))
    print("Windows  :", int(windows.sum()))

    print(
        "Classes  :",
        global_counts.tolist(),
    )

    print()
    print(
        f"Searching {N_CANDIDATES:,} "
        "candidate patient allocations..."
    )

    rng = np.random.default_rng(SEED)

    best_score = np.inf
    best_assignment = None

    indices = np.arange(
        len(patient_ids),
        dtype=np.int64,
    )

    for candidate in range(N_CANDIDATES):

        permutation = rng.permutation(indices)

        assignment = permutation.reshape(
            N_FOLDS,
            PATIENTS_PER_FOLD,
        )

        score = candidate_score(
            assignment,
            counts,
            windows,
        )

        if score < best_score:

            best_score = score

            best_assignment = (
                assignment.copy()
            )

        if (
            candidate > 0
            and candidate % 50_000 == 0
        ):

            print(
                f"[INFO] Checked "
                f"{candidate:,} candidates. "
                f"Best score={best_score:.6f}"
            )

    if best_assignment is None:
        raise RuntimeError(
            "No valid fold assignment found."
        )

    print()
    print("=" * 80)
    print("BEST CANDIDATE")
    print("=" * 80)

    folds = []

    validation_occurrences = {}

    for fold_number, fold_indices in enumerate(
        best_assignment,
        start=1,
    ):

        held_out_patients = sorted(
            patient_ids[fold_indices].tolist()
        )

        train_indices = np.array(
            [
                i
                for i in indices
                if i not in set(fold_indices.tolist())
            ],
            dtype=np.int64,
        )

        train_patients = sorted(
            patient_ids[train_indices].tolist()
        )

        held_counts = counts[
            fold_indices
        ].sum(axis=0)

        train_counts = counts[
            train_indices
        ].sum(axis=0)

        held_windows = int(
            windows[fold_indices].sum()
        )

        train_windows = int(
            windows[train_indices].sum()
        )

        proportions = (
            held_counts
            / held_counts.sum()
            * 100
        )

        pre_patients = int(
            np.sum(
                counts[fold_indices, 1] > 0
            )
        )

        ictal_patients = int(
            np.sum(
                counts[fold_indices, 2] > 0
            )
        )

        for patient in held_out_patients:

            validation_occurrences[patient] = (
                validation_occurrences.get(
                    patient,
                    0,
                )
                + 1
            )

        print()
        print(f"FOLD {fold_number}")
        print("-" * 80)

        print(
            "Train patients    :",
            len(train_patients),
        )

        print(
            "Held-out patients :",
            len(held_out_patients),
        )

        print(
            f"Train windows     : "
            f"{train_windows:,}"
        )

        print(
            f"Held-out windows  : "
            f"{held_windows:,}"
        )

        print()

        print(
            f"Interictal : "
            f"{held_counts[0]:,} "
            f"({proportions[0]:.2f}%)"
        )

        print(
            f"Pre-Ictal  : "
            f"{held_counts[1]:,} "
            f"({proportions[1]:.2f}%)"
        )

        print(
            f"Ictal       : "
            f"{held_counts[2]:,} "
            f"({proportions[2]:.2f}%)"
        )

        print()

        print(
            "Patients with Pre-Ictal:",
            pre_patients,
        )

        print(
            "Patients with Ictal    :",
            ictal_patients,
        )

        print()

        print(
            "Held-out patients:",
            ", ".join(held_out_patients),
        )

        folds.append(
            {
                "fold": fold_number,

                "train_patients": train_patients,

                "held_out_patients": (
                    held_out_patients
                ),

                "train_windows": train_windows,

                "held_out_windows": held_windows,

                "train_counts": {
                    "interictal": int(
                        train_counts[0]
                    ),
                    "pre_ictal": int(
                        train_counts[1]
                    ),
                    "ictal": int(
                        train_counts[2]
                    ),
                },

                "held_out_counts": {
                    "interictal": int(
                        held_counts[0]
                    ),
                    "pre_ictal": int(
                        held_counts[1]
                    ),
                    "ictal": int(
                        held_counts[2]
                    ),
                },

                "held_out_minority_patient_counts": {
                    "with_pre_ictal": pre_patients,
                    "with_ictal": ictal_patients,
                },
            }
        )

    if len(validation_occurrences) != 40:
        raise RuntimeError(
            "Not all patients appear in held-out folds."
        )

    bad = {
        patient: count
        for patient, count
        in validation_occurrences.items()
        if count != 1
    }

    if bad:
        raise RuntimeError(
            f"Invalid held-out occurrences: {bad}"
        )

    output = {
        "method": (
            "Balanced patient-level five-fold "
            "random-search assignment"
        ),

        "seed": SEED,

        "candidate_searches": (
            N_CANDIDATES
        ),

        "n_folds": N_FOLDS,

        "patients_per_fold": (
            PATIENTS_PER_FOLD
        ),

        "best_score": best_score,

        "rules": [
            "Each patient appears in exactly one held-out fold.",
            "Each fold contains exactly 8 held-out patients.",
            "No patient is split between train and held-out data.",
            "Both Pre-Ictal and Ictal must be represented in every held-out fold.",
            "Class-composition similarity is prioritized over equal window totals.",
        ],

        "folds": folds,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print(
        "[PASS] 5 folds x 8 held-out patients"
    )

    print(
        "[PASS] Every patient held out exactly once"
    )

    print(
        "[PASS] No patient leakage"
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()