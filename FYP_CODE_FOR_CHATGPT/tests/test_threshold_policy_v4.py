from src.evaluation.metrics import select_best_threshold_from_rows


def _row(threshold, sensitivity, specificity, f1, false_alarms):
    return {
        "threshold": threshold,
        "alarm_sensitivity": sensitivity,
        "alarm_specificity": specificity,
        "alarm_f1": f1,
        "alarm_balanced_accuracy": (sensitivity + specificity) / 2.0,
        "alarm_youden_j": sensitivity + specificity - 1.0,
        "false_alarms_per_hour": false_alarms,
    }


def test_balanced_accuracy_rejects_all_alarm_threshold():
    rows = [
        _row(0.10, sensitivity=1.00, specificity=0.00, f1=0.31, false_alarms=1400),
        _row(0.55, sensitivity=0.72, specificity=0.70, f1=0.29, false_alarms=350),
        _row(0.80, sensitivity=0.30, specificity=0.92, f1=0.20, false_alarms=100),
    ]

    selected, constraint_met = select_best_threshold_from_rows(
        rows, selection_policy="balanced_accuracy"
    )

    assert constraint_met is True
    assert selected["threshold"] == 0.55


def test_specificity_constrained_policy_respects_minimum():
    rows = [
        _row(0.40, sensitivity=0.90, specificity=0.50, f1=0.40, false_alarms=700),
        _row(0.70, sensitivity=0.60, specificity=0.82, f1=0.35, false_alarms=180),
        _row(0.80, sensitivity=0.40, specificity=0.91, f1=0.28, false_alarms=90),
    ]

    selected, constraint_met = select_best_threshold_from_rows(
        rows,
        selection_policy="specificity_constrained",
        min_specificity=0.80,
    )

    assert constraint_met is True
    assert selected["threshold"] == 0.70
