"""Contract tests for metrics.py and reporting inputs."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.evaluation.metrics import (
    compute_majority_class_baseline,
    compute_metrics_from_outputs,
)


PRIMARY_KEYS = {
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "auc",
    "per_class_precision",
    "per_class_recall",
    "per_class_specificity",
    "per_class_f1",
    "per_class_support",
    "confusion_matrix",
}

ALARM_KEYS = {
    "decision_threshold",
    "alarm_accuracy",
    "alarm_balanced_accuracy",
    "alarm_precision",
    "alarm_recall_sensitivity",
    "alarm_specificity",
    "alarm_f1",
    "alarm_auc",
    "alarm_youden_j",
    "alarm_prediction_rate",
    "alarm_true_rate",
    "alarm_confusion_matrix",
    "tp",
    "tn",
    "fp",
    "fn",
    "evaluated_duration_seconds",
    "evaluated_duration_hours",
    "false_alarms_per_hour",
}


def _assert_complete(metrics, require_majority=False):
    required = PRIMARY_KEYS | ALARM_KEYS
    if require_majority:
        required |= {"majority_class"}

    missing = sorted(required - set(metrics))
    assert not missing, f"Missing reporting keys: {missing}"

    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
    assert np.asarray(metrics["alarm_confusion_matrix"]).shape == (2, 2)

    for key in (
        "per_class_precision",
        "per_class_recall",
        "per_class_specificity",
        "per_class_f1",
        "per_class_support",
    ):
        assert len(metrics[key]) == 3


def test_complete_model_metric_contract():
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_prob = np.eye(3, dtype=np.float64)[y_true]

    metrics = compute_metrics_from_outputs(
        y_true,
        y_prob,
        decision_threshold=0.5,
    )

    _assert_complete(metrics)
    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["alarm_confusion_matrix"] == [[2, 0], [0, 4]]


def test_primary_metrics_do_not_change_with_alarm_threshold():
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_prob = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.40, 0.40, 0.20],
            [0.20, 0.50, 0.30],
            [0.20, 0.30, 0.50],
            [0.70, 0.20, 0.10],
            [0.10, 0.30, 0.60],
        ],
        dtype=np.float64,
    )

    low = compute_metrics_from_outputs(
        y_true,
        y_prob,
        decision_threshold=0.20,
    )
    high = compute_metrics_from_outputs(
        y_true,
        y_prob,
        decision_threshold=0.90,
    )

    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc",
        "confusion_matrix",
    ):
        assert low[key] == high[key]


def test_majority_baseline_has_complete_reporting_contract():
    features = torch.zeros((6, 1, 20, 512), dtype=torch.float32)

    train_dataset = TensorDataset(
        features,
        torch.tensor([0, 0, 0, 0, 1, 2], dtype=torch.long),
    )
    test_dataset = TensorDataset(
        features,
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
    )

    metrics = compute_majority_class_baseline(
        train_dataset,
        test_dataset,
        window_step_s=2.0,
    )

    _assert_complete(metrics, require_majority=True)
    assert metrics["majority_class"] == 0
