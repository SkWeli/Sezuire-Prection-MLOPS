import numpy as np

from src.evaluation.metrics import calculate_metrics_from_probabilities


def _example():
    y_true = np.array([0, 1, 2, 0], dtype=np.int64)
    y_prob = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.40, 0.50, 0.10],
            [0.20, 0.20, 0.60],
            [0.30, 0.40, 0.30],
        ],
        dtype=np.float64,
    )
    return y_true, y_prob


def test_primary_multiclass_metrics_are_independent_of_alarm_threshold():
    y_true, y_prob = _example()
    low = calculate_metrics_from_probabilities(
        y_true, y_prob, average_loss=0.2, window_step_s=2.0, decision_threshold=0.25
    )
    high = calculate_metrics_from_probabilities(
        y_true, y_prob, average_loss=0.2, window_step_s=2.0, decision_threshold=0.75
    )

    assert low["confusion_matrix"] == high["confusion_matrix"]
    assert low["accuracy"] == high["accuracy"]
    assert low["f1"] == high["f1"]
    assert low["alarm_confusion_matrix"] != high["alarm_confusion_matrix"]


def test_metrics_include_per_class_and_alarm_outputs():
    y_true, y_prob = _example()
    metrics = calculate_metrics_from_probabilities(
        y_true, y_prob, average_loss=0.2, window_step_s=2.0, decision_threshold=0.5
    )

    assert len(metrics["per_class_precision"]) == 3
    assert len(metrics["per_class_recall"]) == 3
    assert len(metrics["per_class_f1"]) == 3
    assert len(metrics["per_class_specificity"]) == 3
    assert 0.0 <= metrics["alarm_f1"] <= 1.0
    assert 0.0 <= metrics["alarm_specificity"] <= 1.0
