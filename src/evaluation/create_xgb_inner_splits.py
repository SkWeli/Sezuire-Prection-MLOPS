from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SUMMARY_FILE = Path(
    "data/features/tusz_xgb_220/feature_extraction_summary.csv"
)

OUTER_SPLITS_FILE = Path(
    "experiments/xgboost/patient_cv_folds_balanced_v1.json"
)

OUTPUT_FILE = Path(
    "experiments/xgboost/patient_cv_inner_splits_v1.json"
)

INNER_VALIDATION_PATIENTS = 8
SEED = 42
N_CANDIDATES = 100_000


def load_patient_stats() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}

    with SUMMARY_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            stats[row["patient_id"]] = {
                "n_windows": int(row["n_windows"]),
                "interictal": int(row["n_interictal"]),
                "pre_ictal": int(row["n_pre_ictal"]),
                "ictal": int(row["n_ictal"]),
            }

    if len(stats) != 40:
        raise RuntimeError(
            f"Expected statistics for 40 patients, found {len(stats)}"
        )

    return stats


def aggregate(
    patient_ids: list[str],
    stats: dict[str, dict[str, int]],
) -> tuple[np.ndarray, int]:
    counts = np.zeros(3, dtype=np.int64)
    windows = 0

    for patient_id in patient_ids:
        row = stats[patient_id]

        counts += np.array(
            [
                row["interictal"],
                row["pre_ictal"],
                row["ictal"],
            ],
            dtype=np.int64,
        )

        windows += row["n_windows"]

    return counts, windows


def candidate_score(
    validation_patients: list[str],
    outer_train_patients: list[str],
    stats: dict[str, dict[str, int]],
) -> float:
    val_counts, val_windows = aggregate(
        validation_patients,
        stats,
    )

    outer_counts, outer_windows = aggregate(
        outer_train_patients,
        stats,
    )

    # All three original classes must exist in inner validation.
    if np.any(val_counts == 0):
        return np.inf

    pre_patients = sum(
        stats[p]["pre_ictal"] > 0
        for p in validation_patients
    )

    ictal_patients = sum(
        stats[p]["ictal"] > 0
        for p in validation_patients
    )

    # We want the threshold-selection set to contain several
    # independent patients contributing each minority class.
    if pre_patients < 4 or ictal_patients < 4:
        return np.inf

    val_fraction = val_windows / outer_windows

    # Keep the inner validation set reasonably sized. In particular,
    # this prevents the 203,033-window patient from becoming the
    # majority of an inner validation set.
    if val_fraction < 0.10 or val_fraction > 0.40:
        return np.inf

    outer_prop = (
        outer_counts.astype(np.float64)
        / outer_counts.sum()
    )

    val_prop = (
        val_counts.astype(np.float64)
        / val_counts.sum()
    )

    relative_error = (
        (val_prop - outer_prop)
        / np.maximum(outer_prop, 1e-12)
    )

    class_weights = np.array(
        [0.25, 1.0, 1.0],
        dtype=np.float64,
    )

    class_score = float(
        np.sum(
            class_weights
            * relative_error**2
        )
    )

    # 8 of 32 patients = 25%, so prefer around 25% of windows too,
    # but class composition remains the stronger objective.
    window_score = float(
        ((val_fraction - 0.25) / 0.25) ** 2
    )

    # Slightly reward broader minority-patient representation.
    presence_penalty = (
        (8 - pre_patients) * 0.01
        + (8 - ictal_patients) * 0.01
    )

    return (
        class_score
        + 0.15 * window_score
        + presence_penalty
    )


def main() -> None:
    stats = load_patient_stats()

    with OUTER_SPLITS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        outer = json.load(f)

    folds = outer["folds"]

    if len(folds) != 5:
        raise RuntimeError(
            f"Expected 5 outer folds, found {len(folds)}"
        )

    output_folds = []

    print("=" * 80)
    print("INNER PATIENT-LEVEL VALIDATION SPLIT SEARCH")
    print("=" * 80)
    print(
        "Design: 24 inner-train patients + "
        "8 inner-validation patients"
    )
    print(
        "Outer held-out patients remain completely untouched."
    )

    for fold in folds:
        outer_fold = int(fold["fold"])

        outer_train_patients = sorted(
            fold["train_patients"]
        )

        outer_held_out_patients = sorted(
            fold["held_out_patients"]
        )

        if len(outer_train_patients) != 32:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 32 training patients"
            )

        if len(outer_held_out_patients) != 8:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 8 held-out patients"
            )

        rng = np.random.default_rng(
            SEED + outer_fold * 1000
        )

        patients_array = np.array(
            outer_train_patients,
            dtype=object,
        )

        best_score = np.inf
        best_validation = None

        for _ in range(N_CANDIDATES):
            chosen = rng.choice(
                patients_array,
                size=INNER_VALIDATION_PATIENTS,
                replace=False,
            )

            validation_patients = sorted(
                chosen.tolist()
            )

            score = candidate_score(
                validation_patients,
                outer_train_patients,
                stats,
            )

            if score < best_score:
                best_score = score
                best_validation = validation_patients

        if best_validation is None:
            raise RuntimeError(
                f"Outer fold {outer_fold}: no valid inner split found"
            )

        best_validation_set = set(best_validation)

        inner_train = sorted(
            p
            for p in outer_train_patients
            if p not in best_validation_set
        )

        if len(inner_train) != 24:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 24 inner-train patients"
            )

        if set(inner_train) & set(best_validation):
            raise RuntimeError(
                f"Outer fold {outer_fold}: inner leakage detected"
            )

        if (
            set(inner_train) & set(outer_held_out_patients)
            or set(best_validation) & set(outer_held_out_patients)
        ):
            raise RuntimeError(
                f"Outer fold {outer_fold}: outer held-out leakage detected"
            )

        inner_train_counts, inner_train_windows = aggregate(
            inner_train,
            stats,
        )

        inner_val_counts, inner_val_windows = aggregate(
            best_validation,
            stats,
        )

        pre_patients = sum(
            stats[p]["pre_ictal"] > 0
            for p in best_validation
        )

        ictal_patients = sum(
            stats[p]["ictal"] > 0
            for p in best_validation
        )

        val_prop = (
            inner_val_counts
            / inner_val_counts.sum()
            * 100
        )

        print()
        print(f"OUTER FOLD {outer_fold}")
        print("-" * 80)
        print(
            f"Inner train patients      : {len(inner_train)}"
        )
        print(
            f"Inner validation patients : {len(best_validation)}"
        )
        print(
            f"Inner train windows       : {inner_train_windows:,}"
        )
        print(
            f"Inner validation windows  : {inner_val_windows:,}"
        )
        print(
            f"Inner validation Interictal: "
            f"{inner_val_counts[0]:,} ({val_prop[0]:.2f}%)"
        )
        print(
            f"Inner validation Pre-Ictal : "
            f"{inner_val_counts[1]:,} ({val_prop[1]:.2f}%)"
        )
        print(
            f"Inner validation Ictal      : "
            f"{inner_val_counts[2]:,} ({val_prop[2]:.2f}%)"
        )
        print(
            f"Patients with Pre-Ictal     : {pre_patients}"
        )
        print(
            f"Patients with Ictal         : {ictal_patients}"
        )
        print(
            "Inner validation patients   : "
            + ", ".join(best_validation)
        )
        print(
            "Outer held-out patients      : "
            + ", ".join(outer_held_out_patients)
        )

        output_folds.append(
            {
                "outer_fold": outer_fold,
                "inner_train_patients": inner_train,
                "inner_validation_patients": best_validation,
                "outer_held_out_patients": outer_held_out_patients,
                "inner_train_windows": int(inner_train_windows),
                "inner_validation_windows": int(inner_val_windows),
                "inner_train_counts": {
                    "interictal": int(inner_train_counts[0]),
                    "pre_ictal": int(inner_train_counts[1]),
                    "ictal": int(inner_train_counts[2]),
                },
                "inner_validation_counts": {
                    "interictal": int(inner_val_counts[0]),
                    "pre_ictal": int(inner_val_counts[1]),
                    "ictal": int(inner_val_counts[2]),
                },
                "inner_validation_minority_patient_counts": {
                    "with_pre_ictal": int(pre_patients),
                    "with_ictal": int(ictal_patients),
                },
                "search_score": float(best_score),
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "method": (
            "Frozen outer 5-fold patient CV with one "
            "patient-grouped inner validation split per outer fold"
        ),
        "purpose": (
            "Inner validation is used only for threshold selection. "
            "Outer held-out patients are not used for model or threshold selection."
        ),
        "seed": SEED,
        "candidate_searches_per_outer_fold": N_CANDIDATES,
        "inner_train_patients": 24,
        "inner_validation_patients": 8,
        "folds": output_folds,
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
    print("[PASS] INNER SPLITS CREATED")
    print("[PASS] NO INNER PATIENT LEAKAGE")
    print("[PASS] OUTER HELD-OUT PATIENTS REMAIN UNTOUCHED")
    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
