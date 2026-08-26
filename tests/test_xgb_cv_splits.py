from __future__ import annotations

import json
from pathlib import Path


OUTER_FILE = Path(
    "experiments/xgboost/patient_cv_folds_balanced_v1.json"
)

INNER_FILE = Path(
    "experiments/xgboost/patient_cv_inner_splits_v1.json"
)


def test_outer_and_inner_patient_cv_contract():
    with OUTER_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        outer = json.load(f)

    with INNER_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        inner = json.load(f)

    outer_by_fold = {
        int(row["fold"]): row
        for row in outer["folds"]
    }

    inner_by_fold = {
        int(row["outer_fold"]): row
        for row in inner["folds"]
    }

    assert set(outer_by_fold) == {1, 2, 3, 4, 5}
    assert set(inner_by_fold) == {1, 2, 3, 4, 5}

    all_outer_held_out = []

    for fold_number in range(1, 6):
        outer_fold = outer_by_fold[fold_number]
        inner_fold = inner_by_fold[fold_number]

        outer_train = set(
            outer_fold["train_patients"]
        )

        outer_test = set(
            outer_fold["held_out_patients"]
        )

        inner_train = set(
            inner_fold["inner_train_patients"]
        )

        inner_val = set(
            inner_fold["inner_validation_patients"]
        )

        assert len(outer_train) == 32
        assert len(outer_test) == 8

        assert len(inner_train) == 24
        assert len(inner_val) == 8

        assert not (outer_train & outer_test)
        assert not (inner_train & inner_val)

        assert not (inner_train & outer_test)
        assert not (inner_val & outer_test)

        assert inner_train | inner_val == outer_train

        counts = inner_fold[
            "inner_validation_counts"
        ]

        assert counts["interictal"] > 0
        assert counts["pre_ictal"] > 0
        assert counts["ictal"] > 0

        minority = inner_fold[
            "inner_validation_minority_patient_counts"
        ]

        assert minority["with_pre_ictal"] >= 4
        assert minority["with_ictal"] >= 4

        all_outer_held_out.extend(
            sorted(outer_test)
        )

    assert len(all_outer_held_out) == 40
    assert len(set(all_outer_held_out)) == 40
