from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


DEFAULT_FEATURE_DIR = Path(
    "data/features/tusz_xgb_220"
)

DEFAULT_OUTER_SPLITS = Path(
    "experiments/xgboost/patient_cv_folds_balanced_v1.json"
)

DEFAULT_INNER_SPLITS = Path(
    "experiments/xgboost/patient_cv_inner_splits_v1.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "reports/xgboost_hierarchy/xgb_baseline_v1"
)

SEED = 42

# Fixed baseline hyperparameters.
# These are intentionally not tuned against outer held-out folds.
BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": SEED,
}


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def stratified_smoke_indices(
    y: np.ndarray,
    max_per_class: int = 300,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []

    for label in [0, 1, 2]:
        idx = np.flatnonzero(y == label)

        if len(idx) > max_per_class:
            idx = rng.choice(
                idx,
                size=max_per_class,
                replace=False,
            )

        selected.append(
            np.asarray(idx, dtype=np.int64)
        )

    if not selected:
        return np.arange(len(y))

    result = np.concatenate(selected)
    result.sort()

    return result


def load_patients(
    patient_ids: list[str],
    feature_dir: Path,
    smoke: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_parts = []
    y_parts = []
    patient_parts = []
    window_parts = []

    for patient_id in patient_ids:
        path = (
            feature_dir
            / f"{patient_id}_features.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing feature file: {path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            X = data["X"].astype(
                np.float32,
                copy=False,
            )
            y = data["y"].astype(
                np.int64,
                copy=False,
            )
            window_index = data[
                "window_index"
            ].astype(
                np.int64,
                copy=False,
            )

            stored_patient = str(
                data["patient_id"]
            )

        if stored_patient != patient_id:
            raise RuntimeError(
                f"Patient mismatch in {path}: "
                f"{stored_patient} != {patient_id}"
            )

        if X.shape != (len(y), 220):
            raise RuntimeError(
                f"{patient_id}: unexpected X shape {X.shape}"
            )

        if smoke:
            idx = stratified_smoke_indices(
                y,
                max_per_class=300,
                seed=SEED,
            )
            X = X[idx]
            y = y[idx]
            window_index = window_index[idx]

        X_parts.append(X)
        y_parts.append(y)

        patient_parts.append(
            np.full(
                len(y),
                patient_id,
                dtype="<U8",
            )
        )

        window_parts.append(
            window_index
        )

    X_all = np.concatenate(
        X_parts,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    y_all = np.concatenate(
        y_parts,
        axis=0,
    )

    patients_all = np.concatenate(
        patient_parts,
        axis=0,
    )

    windows_all = np.concatenate(
        window_parts,
        axis=0,
    )

    return (
        X_all,
        y_all,
        patients_all,
        windows_all,
    )


def fit_standardizer(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = X_train.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std[std < 1e-8] = 1.0

    return mean, std


def apply_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    X = np.asarray(
        X,
        dtype=np.float32,
    ).copy()

    X -= mean
    X /= std

    return X


def make_model(
    n_jobs: int,
    smoke: bool,
) -> XGBClassifier:
    params = dict(BASE_PARAMS)
    params["n_jobs"] = n_jobs

    if smoke:
        params["n_estimators"] = 20
        params["max_depth"] = 3

    return XGBClassifier(
        **params
    )


def binary_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    denominator = tn + fp

    if denominator == 0:
        return 0.0

    return float(
        tn / denominator
    )


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (
        probability >= threshold
    ).astype(np.int64)

    result = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            binary_specificity(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "n": int(len(y_true)),
        "positive_count": int(
            np.sum(y_true == 1)
        ),
        "negative_count": int(
            np.sum(y_true == 0)
        ),
    }

    return result


def select_stage1_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    minimum_specificity: float = 0.80,
) -> tuple[float, list[dict], bool]:
    rows = []
    candidates = []

    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):
        metrics = binary_metrics(
            y_true,
            probability,
            float(threshold),
        )

        rows.append(metrics)

        if (
            metrics["specificity"]
            >= minimum_specificity
        ):
            candidates.append(metrics)

    fallback = False

    if candidates:
        best = max(
            candidates,
            key=lambda row: (
                row["balanced_accuracy"],
                row["recall_sensitivity"],
                row["f1"],
            ),
        )
    else:
        fallback = True

        best = max(
            rows,
            key=lambda row: (
                row["balanced_accuracy"],
                row["specificity"],
                row["recall_sensitivity"],
            ),
        )

    return (
        float(best["threshold"]),
        rows,
        fallback,
    )


def select_stage2_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, list[dict]]:
    rows = []

    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):
        y_pred = (
            probability >= threshold
        ).astype(np.int64)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=[0, 1],
                zero_division=0,
            )
        )

        row = {
            "threshold": float(threshold),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "pre_ictal_recall": float(
                recall[0]
            ),
            "ictal_recall": float(
                recall[1]
            ),
            "pre_ictal_f1": float(
                f1[0]
            ),
            "ictal_f1": float(
                f1[1]
            ),
        }

        rows.append(row)

    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["ictal_recall"],
            row["macro_f1"],
        ),
    )

    return (
        float(best["threshold"]),
        rows,
    )


def multiclass_specificity(
    cm: np.ndarray,
) -> tuple[list[float], float]:
    total = cm.sum()
    specificities = []

    for class_index in range(3):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = total - tp - fn - fp

        denominator = tn + fp

        specificity = (
            tn / denominator
            if denominator > 0
            else 0.0
        )

        specificities.append(
            float(specificity)
        )

    return (
        specificities,
        float(np.mean(specificities)),
    )


def end_to_end_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    specificities, macro_specificity = (
        multiclass_specificity(cm)
    )

    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=[0, 1, 2],
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = None

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_specificity": (
            macro_specificity
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_auc_ovr": macro_auc,
        "confusion_matrix": (
            cm.astype(int).tolist()
        ),
        "per_class": {
            "interictal": {
                "precision": float(
                    precision[0]
                ),
                "recall": float(
                    recall[0]
                ),
                "specificity": float(
                    specificities[0]
                ),
                "f1": float(
                    f1[0]
                ),
                "support": int(
                    support[0]
                ),
            },
            "pre_ictal": {
                "precision": float(
                    precision[1]
                ),
                "recall": float(
                    recall[1]
                ),
                "specificity": float(
                    specificities[1]
                ),
                "f1": float(
                    f1[1]
                ),
                "support": int(
                    support[1]
                ),
            },
            "ictal": {
                "precision": float(
                    precision[2]
                ),
                "recall": float(
                    recall[2]
                ),
                "specificity": float(
                    specificities[2]
                ),
                "f1": float(
                    f1[2]
                ),
                "support": int(
                    support[2]
                ),
            },
        },
    }


def patient_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
) -> list[dict]:
    """
    Patient-level metrics.

    Important: some patients contain no Pre-Ictal and/or no Ictal windows.
    Missing-class recall is therefore recorded as None rather than 0 so the
    later patient-balanced average does not unfairly penalize a patient for a
    class that was absent from that patient's ground truth.
    """
    rows = []

    for patient_id in sorted(
        np.unique(patient_ids).tolist()
    ):
        mask = patient_ids == patient_id

        yt = y_true[mask]
        yp = y_pred[mask]

        cm = confusion_matrix(
            yt,
            yp,
            labels=[0, 1, 2],
        )

        recalls = []
        f1_values = []
        class_recalls = {}

        class_names = [
            "interictal",
            "pre_ictal",
            "ictal",
        ]

        for class_index, class_name in enumerate(class_names):
            support = int(
                np.sum(yt == class_index)
            )

            if support == 0:
                class_recalls[class_name] = None
                continue

            tp = int(
                cm[class_index, class_index]
            )

            recall = tp / support
            recalls.append(float(recall))
            class_recalls[class_name] = float(recall)

            class_f1 = f1_score(
                (yt == class_index).astype(np.int64),
                (yp == class_index).astype(np.int64),
                zero_division=0,
            )

            f1_values.append(float(class_f1))

        patient_balanced_accuracy = (
            float(np.mean(recalls))
            if recalls
            else None
        )

        patient_macro_f1_present = (
            float(np.mean(f1_values))
            if f1_values
            else None
        )

        rows.append(
            {
                "patient_id": patient_id,
                "n_windows": int(
                    np.sum(mask)
                ),
                "balanced_accuracy_present_classes": (
                    patient_balanced_accuracy
                ),
                "macro_f1_present_classes": (
                    patient_macro_f1_present
                ),
                "interictal_recall": (
                    class_recalls["interictal"]
                ),
                "pre_ictal_recall": (
                    class_recalls["pre_ictal"]
                ),
                "ictal_recall": (
                    class_recalls["ictal"]
                ),
            }
        )

    return rows


def save_json(
    path: Path,
    obj: dict | list,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
        )


def run_fold(
    fold_number: int,
    outer_row: dict,
    inner_row: dict,
    feature_dir: Path,
    output_dir: Path,
    n_jobs: int,
    smoke: bool,
) -> dict:
    print()
    print("=" * 88)
    print(
        f"OUTER FOLD {fold_number}"
        + (" [SMOKE]" if smoke else "")
    )
    print("=" * 88)

    inner_train_patients = (
        inner_row[
            "inner_train_patients"
        ]
    )

    inner_val_patients = (
        inner_row[
            "inner_validation_patients"
        ]
    )

    outer_train_patients = (
        outer_row[
            "train_patients"
        ]
    )

    outer_test_patients = (
        outer_row[
            "held_out_patients"
        ]
    )

    print(
        "[1/6] Loading inner-train patients..."
    )

    (
        X_inner_train,
        y_inner_train,
        _,
        _,
    ) = load_patients(
        inner_train_patients,
        feature_dir,
        smoke=smoke,
    )

    print(
        "[2/6] Loading inner-validation patients..."
    )

    (
        X_inner_val,
        y_inner_val,
        _,
        _,
    ) = load_patients(
        inner_val_patients,
        feature_dir,
        smoke=smoke,
    )

    mean, std = fit_standardizer(
        X_inner_train
    )

    X_inner_train = apply_standardizer(
        X_inner_train,
        mean,
        std,
    )

    X_inner_val = apply_standardizer(
        X_inner_val,
        mean,
        std,
    )

    # ------------------------
    # Stage 1 threshold selection
    # ------------------------
    print(
        "[3/6] Stage 1 inner training "
        "(Interictal vs Alarm)..."
    )

    y1_train = (
        y_inner_train > 0
    ).astype(np.int64)

    y1_val = (
        y_inner_val > 0
    ).astype(np.int64)

    stage1_inner_model = make_model(
        n_jobs=n_jobs,
        smoke=smoke,
    )

    stage1_inner_model.fit(
        X_inner_train,
        y1_train,
    )

    p_alarm_val = (
        stage1_inner_model
        .predict_proba(
            X_inner_val
        )[:, 1]
    )

    (
        stage1_threshold,
        stage1_sweep,
        stage1_fallback,
    ) = select_stage1_threshold(
        y1_val,
        p_alarm_val,
        minimum_specificity=0.80,
    )

    stage1_inner_metrics = (
        binary_metrics(
            y1_val,
            p_alarm_val,
            stage1_threshold,
        )
    )

    print(
        "      Stage 1 threshold:",
        stage1_threshold,
    )

    print(
        "      Stage 1 inner AUC:",
        f"{stage1_inner_metrics['roc_auc']:.4f}",
    )

    print(
        "      Stage 1 inner BA:",
        f"{stage1_inner_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "      Stage 1 inner sens/spec:",
        f"{stage1_inner_metrics['recall_sensitivity']:.4f}",
        "/",
        f"{stage1_inner_metrics['specificity']:.4f}",
    )

    # ------------------------
    # Stage 2 threshold selection
    # ------------------------
    print(
        "[4/6] Stage 2 inner training "
        "(Pre-Ictal vs Ictal)..."
    )

    stage2_train_mask = (
        y_inner_train > 0
    )

    stage2_val_mask = (
        y_inner_val > 0
    )

    X2_train = (
        X_inner_train[
            stage2_train_mask
        ]
    )

    y2_train = (
        y_inner_train[
            stage2_train_mask
        ] == 2
    ).astype(np.int64)

    X2_val = (
        X_inner_val[
            stage2_val_mask
        ]
    )

    y2_val = (
        y_inner_val[
            stage2_val_mask
        ] == 2
    ).astype(np.int64)

    stage2_inner_model = make_model(
        n_jobs=n_jobs,
        smoke=smoke,
    )

    stage2_inner_model.fit(
        X2_train,
        y2_train,
    )

    p_ictal_val = (
        stage2_inner_model
        .predict_proba(
            X2_val
        )[:, 1]
    )

    (
        stage2_threshold,
        stage2_sweep,
    ) = select_stage2_threshold(
        y2_val,
        p_ictal_val,
    )

    stage2_best_row = next(
        row
        for row in stage2_sweep
        if abs(
            row["threshold"]
            - stage2_threshold
        ) < 1e-9
    )

    try:
        stage2_auc = float(
            roc_auc_score(
                y2_val,
                p_ictal_val,
            )
        )

        stage2_pr_auc = float(
            average_precision_score(
                y2_val,
                p_ictal_val,
            )
        )
    except ValueError:
        stage2_auc = None
        stage2_pr_auc = None

    print(
        "      Stage 2 threshold:",
        stage2_threshold,
    )

    print(
        "      Stage 2 inner AUC:",
        (
            f"{stage2_auc:.4f}"
            if stage2_auc is not None
            else "NA"
        ),
    )

    print(
        "      Stage 2 inner BA:",
        f"{stage2_best_row['balanced_accuracy']:.4f}",
    )

    del (
        stage1_inner_model,
        stage2_inner_model,
        X_inner_train,
        X_inner_val,
        X2_train,
        X2_val,
        y_inner_train,
        y_inner_val,
        y1_train,
        y1_val,
        y2_train,
        y2_val,
        p_alarm_val,
        p_ictal_val,
    )

    gc.collect()

    # ------------------------
    # Refit on all outer training patients
    # ------------------------
    print(
        "[5/6] Refit frozen configuration "
        "on all 32 outer-training patients..."
    )

    (
        X_outer_train,
        y_outer_train,
        _,
        _,
    ) = load_patients(
        outer_train_patients,
        feature_dir,
        smoke=smoke,
    )

    (
        X_outer_test,
        y_outer_test,
        test_patient_ids,
        test_window_indices,
    ) = load_patients(
        outer_test_patients,
        feature_dir,
        smoke=smoke,
    )

    outer_mean, outer_std = (
        fit_standardizer(
            X_outer_train
        )
    )

    X_outer_train = (
        apply_standardizer(
            X_outer_train,
            outer_mean,
            outer_std,
        )
    )

    X_outer_test = (
        apply_standardizer(
            X_outer_test,
            outer_mean,
            outer_std,
        )
    )

    y1_outer_train = (
        y_outer_train > 0
    ).astype(np.int64)

    final_stage1 = make_model(
        n_jobs=n_jobs,
        smoke=smoke,
    )

    final_stage1.fit(
        X_outer_train,
        y1_outer_train,
    )

    stage2_outer_train_mask = (
        y_outer_train > 0
    )

    y2_outer_train = (
        y_outer_train[
            stage2_outer_train_mask
        ] == 2
    ).astype(np.int64)

    final_stage2 = make_model(
        n_jobs=n_jobs,
        smoke=smoke,
    )

    final_stage2.fit(
        X_outer_train[
            stage2_outer_train_mask
        ],
        y2_outer_train,
    )

    print(
        "[6/6] Frozen outer held-out evaluation..."
    )

    p_alarm_test = (
        final_stage1
        .predict_proba(
            X_outer_test
        )[:, 1]
    )

    p_ictal_given_alarm = (
        final_stage2
        .predict_proba(
            X_outer_test
        )[:, 1]
    )

    y1_test = (
        y_outer_test > 0
    ).astype(np.int64)

    stage1_test_metrics = (
        binary_metrics(
            y1_test,
            p_alarm_test,
            stage1_threshold,
        )
    )

    # Stage 2 standalone diagnostic on true seizure-related
    # held-out windows only.
    true_stage2_mask = (
        y_outer_test > 0
    )

    y2_test = (
        y_outer_test[
            true_stage2_mask
        ] == 2
    ).astype(np.int64)

    p2_true = (
        p_ictal_given_alarm[
            true_stage2_mask
        ]
    )

    y2_pred = (
        p2_true
        >= stage2_threshold
    ).astype(np.int64)

    stage2_precision, stage2_recall, stage2_f1, _ = (
        precision_recall_fscore_support(
            y2_test,
            y2_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    try:
        stage2_test_auc = float(
            roc_auc_score(
                y2_test,
                p2_true,
            )
        )

        stage2_test_pr_auc = float(
            average_precision_score(
                y2_test,
                p2_true,
            )
        )
    except ValueError:
        stage2_test_auc = None
        stage2_test_pr_auc = None

    stage2_test_metrics = {
        "threshold": float(
            stage2_threshold
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y2_test,
                y2_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y2_test,
                y2_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": stage2_test_auc,
        "pr_auc": stage2_test_pr_auc,
        "pre_ictal_recall": float(
            stage2_recall[0]
        ),
        "ictal_recall": float(
            stage2_recall[1]
        ),
        "pre_ictal_f1": float(
            stage2_f1[0]
        ),
        "ictal_f1": float(
            stage2_f1[1]
        ),
        "n_true_seizure_related": int(
            len(y2_test)
        ),
    }

    # End-to-end hierarchy.
    y_pred = np.zeros(
        len(y_outer_test),
        dtype=np.int64,
    )

    routed_alarm = (
        p_alarm_test
        >= stage1_threshold
    )

    routed_ictal = (
        p_ictal_given_alarm
        >= stage2_threshold
    )

    y_pred[
        routed_alarm
        & ~routed_ictal
    ] = 1

    y_pred[
        routed_alarm
        & routed_ictal
    ] = 2

    probabilities = np.column_stack(
        [
            1.0 - p_alarm_test,
            p_alarm_test
            * (
                1.0
                - p_ictal_given_alarm
            ),
            p_alarm_test
            * p_ictal_given_alarm,
        ]
    ).astype(np.float32)

    e2e_metrics = (
        end_to_end_metrics(
            y_outer_test,
            y_pred,
            probabilities,
        )
    )

    per_patient = (
        patient_level_metrics(
            y_outer_test,
            y_pred,
            probabilities,
            test_patient_ids,
        )
    )

    fold_dir = (
        output_dir
        / f"fold_{fold_number}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save final models in portable XGBoost JSON format.
    final_stage1.save_model(
        str(
            fold_dir
            / "stage1_xgb.json"
        )
    )

    final_stage2.save_model(
        str(
            fold_dir
            / "stage2_xgb.json"
        )
    )

    np.savez_compressed(
        fold_dir / "standardizer.npz",
        mean=outer_mean,
        std=outer_std,
    )

    np.savez_compressed(
        fold_dir
        / "outer_heldout_predictions.npz",
        y_true=y_outer_test,
        y_pred=y_pred,
        probabilities=probabilities,
        p_alarm=p_alarm_test.astype(
            np.float32
        ),
        p_ictal_given_alarm=(
            p_ictal_given_alarm.astype(
                np.float32
            )
        ),
        patient_id=test_patient_ids,
        window_index=test_window_indices,
    )

    save_json(
        fold_dir
        / "stage1_threshold_sweep.json",
        stage1_sweep,
    )

    save_json(
        fold_dir
        / "stage2_threshold_sweep.json",
        stage2_sweep,
    )

    fold_result = {
        "experiment": (
            "hierarchical_xgboost_baseline_no_smote"
        ),
        "smoke": bool(smoke),
        "outer_fold": int(
            fold_number
        ),
        "seed": SEED,
        "xgboost_params": {
            **BASE_PARAMS,
            "n_jobs": n_jobs,
            "smoke_override": bool(
                smoke
            ),
        },
        "outer_train_patients": (
            outer_train_patients
        ),
        "inner_train_patients": (
            inner_train_patients
        ),
        "inner_validation_patients": (
            inner_val_patients
        ),
        "outer_held_out_patients": (
            outer_test_patients
        ),
        "stage1": {
            "selection_policy": (
                "inner validation threshold maximizing "
                "balanced accuracy subject to specificity >= 0.80"
            ),
            "threshold": float(
                stage1_threshold
            ),
            "specificity_constraint_fallback": (
                bool(stage1_fallback)
            ),
            "inner_validation": (
                stage1_inner_metrics
            ),
            "outer_heldout": (
                stage1_test_metrics
            ),
        },
        "stage2": {
            "selection_policy": (
                "inner validation threshold maximizing "
                "balanced accuracy, tie-break Ictal recall then macro F1"
            ),
            "threshold": float(
                stage2_threshold
            ),
            "inner_validation": {
                **stage2_best_row,
                "roc_auc": stage2_auc,
                "pr_auc": stage2_pr_auc,
            },
            "outer_heldout_true_seizure_related": (
                stage2_test_metrics
            ),
        },
        "end_to_end_outer_heldout": (
            e2e_metrics
        ),
        "patient_level_outer_heldout": (
            per_patient
        ),
    }

    save_json(
        fold_dir
        / "fold_metrics.json",
        fold_result,
    )

    print()
    print(
        "Outer held-out end-to-end:"
    )

    print(
        "  Balanced accuracy:",
        f"{e2e_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "  Macro F1:",
        f"{e2e_metrics['macro_f1']:.4f}",
    )

    print(
        "  Macro AUC:",
        (
            f"{e2e_metrics['macro_auc_ovr']:.4f}"
            if e2e_metrics[
                "macro_auc_ovr"
            ] is not None
            else "NA"
        ),
    )

    print(
        "  Recalls (Int/Pre/Ictal):",
        f"{e2e_metrics['per_class']['interictal']['recall']:.4f}",
        "/",
        f"{e2e_metrics['per_class']['pre_ictal']['recall']:.4f}",
        "/",
        f"{e2e_metrics['per_class']['ictal']['recall']:.4f}",
    )

    print(
        "  Confusion matrix:",
        e2e_metrics[
            "confusion_matrix"
        ],
    )

    del (
        final_stage1,
        final_stage2,
        X_outer_train,
        X_outer_test,
        y_outer_train,
        y_outer_test,
        y1_outer_train,
        y2_outer_train,
        p_alarm_test,
        p_ictal_given_alarm,
        probabilities,
        y_pred,
    )

    gc.collect()

    return fold_result


def summarize_results(
    results: list[dict],
) -> dict:
    if not results:
        return {}

    fold_metrics = [
        row[
            "end_to_end_outer_heldout"
        ]
        for row in results
    ]

    def mean_std(key: str) -> dict:
        values = np.array(
            [
                row[key]
                for row in fold_metrics
                if row[key] is not None
            ],
            dtype=np.float64,
        )

        return {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            ),
        }

    patient_rows = []

    for row in results:
        patient_rows.extend(
            row[
                "patient_level_outer_heldout"
            ]
        )

    patient_summary = {}

    patient_keys = [
        "balanced_accuracy_present_classes",
        "macro_f1_present_classes",
        "interictal_recall",
        "pre_ictal_recall",
        "ictal_recall",
    ]

    for key in patient_keys:
        values = np.array(
            [
                patient[key]
                for patient in patient_rows
                if patient[key] is not None
            ],
            dtype=np.float64,
        )

        patient_summary[key] = {
            "mean": float(
                np.mean(values)
            ) if len(values) else None,
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            ) if len(values) else None,
            "n_patients_contributing": int(
                len(values)
            ),
        }

    return {
        "fold_level_equal_weight": {
            "balanced_accuracy": (
                mean_std(
                    "balanced_accuracy"
                )
            ),
            "macro_f1": (
                mean_std(
                    "macro_f1"
                )
            ),
            "macro_auc_ovr": (
                mean_std(
                    "macro_auc_ovr"
                )
            ),
        },
        "patient_balanced_summary": (
            patient_summary
        ),
        "n_completed_folds": len(
            results
        ),
        "n_patient_rows": len(
            patient_rows
        ),
    }


def pooled_oof_metrics(
    mode_dir: Path,
    folds: list[int],
) -> dict | None:
    if len(folds) != 5:
        return None

    y_true_parts = []
    y_pred_parts = []
    probability_parts = []
    patient_parts = []

    for fold_number in folds:
        path = (
            mode_dir
            / f"fold_{fold_number}"
            / "outer_heldout_predictions.npz"
        )

        if not path.exists():
            return None

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            y_true_parts.append(
                data["y_true"]
            )
            y_pred_parts.append(
                data["y_pred"]
            )
            probability_parts.append(
                data["probabilities"]
            )
            patient_parts.append(
                data["patient_id"]
            )

    y_true = np.concatenate(
        y_true_parts
    )
    y_pred = np.concatenate(
        y_pred_parts
    )
    probabilities = np.concatenate(
        probability_parts
    )
    patients = np.concatenate(
        patient_parts
    )

    metrics = end_to_end_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    metrics["n_windows"] = int(
        len(y_true)
    )
    metrics["n_unique_patients"] = int(
        len(np.unique(patients))
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
    )

    parser.add_argument(
        "--outer-splits",
        type=Path,
        default=DEFAULT_OUTER_SPLITS,
    )

    parser.add_argument(
        "--inner-splits",
        type=Path,
        default=DEFAULT_INNER_SPLITS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--outer-fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help=(
            "Run one outer fold only. "
            "Omit to run all five."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast pipeline test using a small "
            "stratified sample per patient and "
            "20 trees. NOT for scientific results."
        ),
    )

    args = parser.parse_args()

    outer = load_json(
        args.outer_splits
    )

    inner = load_json(
        args.inner_splits
    )

    outer_by_fold = {
        int(row["fold"]): row
        for row in outer["folds"]
    }

    inner_by_fold = {
        int(row["outer_fold"]): row
        for row in inner["folds"]
    }

    folds_to_run = (
        [args.outer_fold]
        if args.outer_fold is not None
        else [1, 2, 3, 4, 5]
    )

    mode_dir = (
        args.output_dir
        / (
            "SMOKE"
            if args.smoke
            else "FULL"
        )
    )

    mode_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        mode_dir
        / "experiment_config.json",
        {
            "experiment": (
                "hierarchical_xgboost_baseline_no_smote"
            ),
            "smoke": bool(
                args.smoke
            ),
            "seed": SEED,
            "feature_dir": str(
                args.feature_dir
            ),
            "outer_splits": str(
                args.outer_splits
            ),
            "inner_splits": str(
                args.inner_splits
            ),
            "xgboost_params": {
                **BASE_PARAMS,
                "n_jobs": (
                    args.n_jobs
                ),
            },
            "stage1_threshold_policy": (
                "Specificity-constrained, minimum 0.80, "
                "selected on inner validation only"
            ),
            "stage2_threshold_policy": (
                "Maximize balanced accuracy on inner validation only; "
                "tie-break Ictal recall then macro F1"
            ),
            "smote": False,
        },
    )

    results = []

    start_all = time.time()

    for fold_number in folds_to_run:
        if (
            fold_number not in outer_by_fold
            or fold_number not in inner_by_fold
        ):
            raise RuntimeError(
                f"Missing split definition for fold {fold_number}"
            )

        result = run_fold(
            fold_number=fold_number,
            outer_row=outer_by_fold[
                fold_number
            ],
            inner_row=inner_by_fold[
                fold_number
            ],
            feature_dir=args.feature_dir,
            output_dir=mode_dir,
            n_jobs=args.n_jobs,
            smoke=args.smoke,
        )

        results.append(result)

        summary = summarize_results(
            results
        )

        save_json(
            mode_dir
            / "running_summary.json",
            summary,
        )

    elapsed = time.time() - start_all

    final_summary = summarize_results(
        results
    )

    final_summary[
        "elapsed_seconds"
    ] = float(elapsed)

    final_summary[
        "folds_run"
    ] = folds_to_run

    pooled = pooled_oof_metrics(
        mode_dir,
        folds_to_run,
    )

    if pooled is not None:
        final_summary[
            "pooled_out_of_fold"
        ] = pooled

    save_json(
        mode_dir
        / "final_summary.json",
        final_summary,
    )

    print()
    print("=" * 88)
    print("RUN COMPLETE")
    print("=" * 88)
    print(
        "Folds:",
        folds_to_run,
    )
    print(
        "Elapsed seconds:",
        round(elapsed, 2),
    )
    print(
        "Output:",
        mode_dir,
    )

    if len(results) == 5:
        print()
        print(
            "Equal-weight fold mean balanced accuracy:",
            f"{final_summary['fold_level_equal_weight']['balanced_accuracy']['mean']:.4f}",
        )
        print(
            "Equal-weight fold mean macro F1:",
            f"{final_summary['fold_level_equal_weight']['macro_f1']['mean']:.4f}",
        )
        print(
            "Patient-balanced mean macro F1:",
            f"{final_summary['patient_balanced_summary']['macro_f1_present_classes']['mean']:.4f}",
        )


if __name__ == "__main__":
    main()
