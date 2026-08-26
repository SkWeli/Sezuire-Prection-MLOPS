"""Sanity tests for the immutable three-class EEG metric contract."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.evaluation.metrics import (
    apply_temporal_smoothing,
    compute_metrics_from_outputs,
    compute_multiclass_auc,
    probabilities_to_predictions,
)

CLASS_NAMES = {
    0: "Interictal",
    1: "Pre-Ictal",
    2: "Ictal",
}


def test_class_contract():
    assert CLASS_NAMES == {
        0: "Interictal",
        1: "Pre-Ictal",
        2: "Ictal",
    }


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.05, 0.90, 0.05],
            [0.10, 0.85, 0.05],
            [0.05, 0.05, 0.90],
            [0.05, 0.10, 0.85],
        ]
    )

    assert np.mean(y_true == y_pred) == pytest.approx(1.0)
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred, average="macro") == pytest.approx(1.0)
    assert compute_multiclass_auc(y_true, y_prob) == pytest.approx(1.0)


def test_completely_wrong_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 1, 2, 2, 0, 0])

    assert np.mean(y_true == y_pred) == pytest.approx(0.0)
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(0.0)
    assert f1_score(y_true, y_pred, average="macro", zero_division=0) == pytest.approx(0.0)


def test_constant_interictal_prediction():
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.zeros_like(y_true)

    accuracy = np.mean(y_true == y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    assert accuracy == pytest.approx(0.60)
    assert balanced_accuracy == pytest.approx(1.0 / 3.0)
    assert macro_f1 < accuracy
    assert cm[1, 1] == 0
    assert cm[2, 2] == 0


def test_preictal_and_ictal_swap():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 2, 2, 1, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    assert cm.tolist() == [[2, 0, 0], [0, 0, 2], [0, 2, 0]]
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_known_auc_ranking():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.80, 0.15, 0.05],
            [0.05, 0.90, 0.05],
            [0.10, 0.75, 0.15],
            [0.02, 0.08, 0.90],
            [0.05, 0.15, 0.80],
        ]
    )
    assert compute_multiclass_auc(y_true, y_prob) == pytest.approx(1.0)


def test_equal_probabilities_are_chance_auc():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.full((len(y_true), 3), 1.0 / 3.0)
    assert compute_multiclass_auc(y_true, y_prob) == pytest.approx(0.5)


def test_probability_columns_follow_class_order():
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ]
    )
    assert np.argmax(y_prob, axis=1).tolist() == [0, 1, 2]


def test_alarm_threshold_changes_predictions():
    y_prob = np.array(
        [
            [0.45, 0.35, 0.20],  # alarm sum = 0.55
            [0.70, 0.20, 0.10],  # alarm sum = 0.30
            [0.20, 0.25, 0.55],  # alarm sum = 0.80, Ictal wins
        ]
    )

    assert probabilities_to_predictions(y_prob, 0.50).tolist() == [1, 0, 2]
    assert probabilities_to_predictions(y_prob, 0.80).tolist() == [0, 0, 2]
    assert probabilities_to_predictions(y_prob, None).tolist() == [0, 0, 2]


def test_multiclass_smoothing_preserves_three_probability_columns():
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.10, 0.80],
            [0.05, 0.05, 0.90],
            [0.10, 0.80, 0.10],
            [0.90, 0.05, 0.05],
        ]
    )

    smoothed = apply_temporal_smoothing(y_prob, window_size=3)
    assert smoothed.shape == (5, 3)
    assert np.allclose(smoothed.sum(axis=1), 1.0)
    assert np.isfinite(smoothed).all()
    assert np.any(np.argmax(smoothed, axis=1) == 2)


def test_metrics_report_minority_recalls_separately():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.10, 0.80],
            [0.60, 0.10, 0.30],
        ]
    )

    metrics = compute_metrics_from_outputs(
        y_true,
        y_prob,
        decision_threshold=0.5,
        window_step_s=2.0,
    )

    assert metrics["preictal_recall"] == pytest.approx(0.5)
    assert metrics["ictal_recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [1, 1, 0], [1, 0, 1]]


def test_malformed_auc_input_raises_instead_of_hiding_bug():
    y_true = np.array([0, 1, 2])
    malformed = np.array([0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="shape"):
        compute_multiclass_auc(y_true, malformed)
