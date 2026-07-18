"""Evaluation metrics for multiclass EEG seizure detection and alarm analysis."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CLASS_NAMES = ("Interictal", "Pre-Ictal", "Ictal")


def compute_multiclass_auc(y_true, y_prob):
    """Return macro one-vs-rest AUC, or NaN when it is not defined."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
        return float("nan")

    n_classes = y_prob.shape[1]
    if n_classes < 2 or len(np.unique(y_true)) != n_classes:
        return float("nan")

    try:
        return float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=np.arange(n_classes),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def compute_binary_auc(y_true, positive_scores):
    """Return binary AUC, or NaN if only one target class is present."""
    y_true = np.asarray(y_true, dtype=np.int64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y_true, positive_scores))
    except ValueError:
        return float("nan")


def multiclass_predictions_from_probabilities(probabilities):
    """Primary three-class prediction: always use class-probability argmax."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected probabilities with shape (n_samples, n_classes>=2), "
            f"got {probabilities.shape}."
        )
    return np.argmax(probabilities, axis=1).astype(np.int64)


def probabilities_to_predictions(probabilities, decision_threshold=0.5):
    """Backward-compatible thresholded three-class helper.

    New evaluation code does not use this function for the primary multiclass
    task. It is retained so older callers and tests do not break.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected probabilities with shape (n_samples, n_classes>=2), "
            f"got {probabilities.shape}."
        )
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")
    if probabilities.shape[1] == 2:
        return (probabilities[:, 1] >= decision_threshold).astype(np.int64)
    alarm_probability = probabilities[:, 1:].sum(axis=1)
    alarm_subclass = 1 + np.argmax(probabilities[:, 1:], axis=1)
    return np.where(alarm_probability >= decision_threshold, alarm_subclass, 0).astype(np.int64)


def alarm_predictions_from_probabilities(probabilities, decision_threshold=0.5):
    """Secondary binary alarm prediction: Interictal=0, Pre-Ictal/Ictal=1."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected probabilities with shape (n_samples, n_classes>=2), "
            f"got {probabilities.shape}."
        )
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")

    if probabilities.shape[1] == 2:
        alarm_probability = probabilities[:, 1]
    else:
        alarm_probability = probabilities[:, 1:].sum(axis=1)

    return (alarm_probability >= decision_threshold).astype(np.int64)


def apply_temporal_smoothing(predictions, window_size=5):
    """Apply centred majority voting without collapsing multiclass labels."""
    predictions = np.asarray(predictions, dtype=np.int64)

    if window_size <= 1 or predictions.size == 0:
        return predictions.copy()
    if window_size % 2 == 0:
        raise ValueError("smoothing window_size must be odd.")

    n_classes = int(predictions.max()) + 1
    radius = window_size // 2
    smoothed = predictions.copy()

    for index in range(predictions.size):
        start = max(0, index - radius)
        stop = min(predictions.size, index + radius + 1)
        local_values = predictions[start:stop]
        counts = np.bincount(local_values, minlength=n_classes)
        winners = np.flatnonzero(counts == counts.max())
        smoothed[index] = (
            predictions[index]
            if predictions[index] in winners
            else int(winners[0])
        )

    return smoothed


def _specificity_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    total = confusion.sum()
    values = []

    for class_index in range(confusion.shape[0]):
        tp = confusion[class_index, class_index]
        fn = confusion[class_index, :].sum() - tp
        fp = confusion[:, class_index].sum() - tp
        tn = total - tp - fn - fp
        denominator = tn + fp
        values.append(float(tn / denominator) if denominator > 0 else float("nan"))

    return float(np.nanmean(values)), values


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


def calculate_metrics_from_probabilities(
    y_true,
    y_prob,
    average_loss,
    window_step_s,
    decision_threshold=0.5,
    smoothing_window=0,
):
    """
    Calculate two explicitly separated evaluations.

    Primary evaluation:
        Three-class Interictal / Pre-Ictal / Ictal prediction using argmax.

    Secondary evaluation:
        Binary alarm/no-alarm prediction using a validation-selected threshold
        on P(Pre-Ictal) + P(Ictal).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty dataset split.")
    if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
        raise ValueError("Probability matrix shape does not match target labels.")

    n_classes = y_prob.shape[1]
    labels = np.arange(n_classes)

    # PRIMARY MULTICLASS TASK: independent of alarm threshold.
    y_pred = multiclass_predictions_from_probabilities(y_prob)
    if smoothing_window > 1:
        y_pred = apply_temporal_smoothing(y_pred, window_size=smoothing_window)

    multiclass_confusion = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_specificity, per_class_specificity = _specificity_from_confusion(
        multiclass_confusion
    )

    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(per_class_f1))

    # SECONDARY ALARM TASK: thresholded separately.
    true_alarm = (y_true > 0).astype(np.int64)
    alarm_probability = (
        y_prob[:, 1]
        if n_classes == 2
        else y_prob[:, 1:].sum(axis=1)
    )
    predicted_alarm = alarm_predictions_from_probabilities(
        y_prob, decision_threshold=decision_threshold
    )
    if smoothing_window > 1:
        predicted_alarm = apply_temporal_smoothing(
            predicted_alarm, window_size=smoothing_window
        )

    alarm_confusion = confusion_matrix(true_alarm, predicted_alarm, labels=[0, 1])
    tn, fp, fn, tp = alarm_confusion.ravel()

    alarm_precision = _safe_divide(tp, tp + fp)
    alarm_recall = _safe_divide(tp, tp + fn)
    alarm_specificity = _safe_divide(tn, tn + fp)
    alarm_f1 = _safe_divide(2 * alarm_precision * alarm_recall, alarm_precision + alarm_recall)
    alarm_balanced_accuracy = (alarm_recall + alarm_specificity) / 2.0
    alarm_accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    evaluated_duration_seconds = float(y_true.size * window_step_s)
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0
    false_alarms_per_hour = (
        float(fp / evaluated_duration_hours)
        if evaluated_duration_hours > 0
        else 0.0
    )

    return {
        # Primary multiclass metrics.
        "loss": float(average_loss),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": macro_recall,
        "precision": macro_precision,
        "recall_sensitivity": macro_recall,
        "specificity": macro_specificity,
        "f1": macro_f1,
        "auc": compute_multiclass_auc(y_true, y_prob),
        "confusion_matrix": multiclass_confusion.tolist(),
        "per_class_precision": [float(value) for value in per_class_precision],
        "per_class_recall": [float(value) for value in per_class_recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "per_class_specificity": [float(value) for value in per_class_specificity],
        # Secondary alarm metrics.
        "decision_threshold": float(decision_threshold),
        "alarm_accuracy": alarm_accuracy,
        "alarm_balanced_accuracy": alarm_balanced_accuracy,
        "alarm_precision": alarm_precision,
        "alarm_recall_sensitivity": alarm_recall,
        "alarm_specificity": alarm_specificity,
        "alarm_f1": alarm_f1,
        "alarm_auc": compute_binary_auc(true_alarm, alarm_probability),
        "alarm_confusion_matrix": alarm_confusion.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }


def collect_model_outputs(model, loader, criterion):
    """Run one inference pass and return labels, probabilities, and average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_scores = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_true.extend(yb.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation loader contains zero samples.")

    return {
        "y_true": np.asarray(all_true, dtype=np.int64),
        "y_prob": np.asarray(all_scores, dtype=np.float64),
        "average_loss": total_loss / total_samples,
    }


def evaluate_model(
    model,
    loader,
    criterion,
    window_step_s=2.0,
    decision_threshold=0.5,
    smoothing_window=0,
):
    """Evaluate primary multiclass and secondary alarm tasks."""
    outputs = collect_model_outputs(model, loader, criterion)
    return calculate_metrics_from_probabilities(
        y_true=outputs["y_true"],
        y_prob=outputs["y_prob"],
        average_loss=outputs["average_loss"],
        window_step_s=window_step_s,
        decision_threshold=decision_threshold,
        smoothing_window=smoothing_window,
    )


def find_best_threshold(
    model,
    loader,
    criterion,
    window_step_s=2.0,
    metric_name="alarm_f1",
    min_threshold=0.05,
    max_threshold=0.95,
    step=0.01,
    smoothing_window=0,
):
    """Tune only the secondary alarm threshold on the validation split."""
    outputs = collect_model_outputs(model, loader, criterion)
    thresholds = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics = None
    sweep_rows = []

    for threshold in thresholds:
        metrics = calculate_metrics_from_probabilities(
            y_true=outputs["y_true"],
            y_prob=outputs["y_prob"],
            average_loss=outputs["average_loss"],
            window_step_s=window_step_s,
            decision_threshold=float(threshold),
            smoothing_window=smoothing_window,
        )
        score = float(metrics.get(metric_name, np.nan))

        row = {
            "threshold": float(threshold),
            "macro_f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "interictal_recall": metrics["per_class_recall"][0],
            "preictal_recall": metrics["per_class_recall"][1],
            "ictal_recall": metrics["per_class_recall"][2],
            "alarm_accuracy": metrics["alarm_accuracy"],
            "alarm_balanced_accuracy": metrics["alarm_balanced_accuracy"],
            "alarm_precision": metrics["alarm_precision"],
            "alarm_sensitivity": metrics["alarm_recall_sensitivity"],
            "alarm_specificity": metrics["alarm_specificity"],
            "alarm_f1": metrics["alarm_f1"],
            "alarm_auc": metrics["alarm_auc"],
            "false_alarms_per_hour": metrics["false_alarms_per_hour"],
        }
        sweep_rows.append(row)

        if not np.isfinite(score):
            continue

        is_better = score > best_score + 1e-12
        is_equal_with_fewer_false_alarms = (
            abs(score - best_score) <= 1e-12
            and best_metrics is not None
            and metrics["false_alarms_per_hour"]
            < best_metrics["false_alarms_per_hour"] - 1e-12
        )
        is_equal_and_closer_to_default = (
            abs(score - best_score) <= 1e-12
            and best_metrics is not None
            and abs(
                metrics["false_alarms_per_hour"]
                - best_metrics["false_alarms_per_hour"]
            ) <= 1e-12
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        )

        if is_better or is_equal_with_fewer_false_alarms or is_equal_and_closer_to_default:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError(
            f"Unable to select a threshold because validation metric '{metric_name}' "
            "was unavailable for every candidate threshold."
        )

    return {
        "best_threshold": best_threshold,
        "best_score": float(best_score),
        "metric_name": metric_name,
        "best_metrics": best_metrics,
        "sweep_rows": sweep_rows,
    }


def get_labels_from_split(split_dataset):
    """Extract labels from TensorDataset or torch.utils.data.Subset."""
    if hasattr(split_dataset, "dataset") and hasattr(split_dataset, "indices"):
        labels_tensor = split_dataset.dataset.tensors[1]
        return labels_tensor[split_dataset.indices].cpu().numpy()
    if hasattr(split_dataset, "tensors"):
        return split_dataset.tensors[1].cpu().numpy()
    raise TypeError(f"Unsupported dataset type: {type(split_dataset)}")


def compute_majority_class_baseline(train_dataset, test_dataset, window_step_s=2.0):
    """Evaluate a fixed majority-class baseline using identical metric definitions."""
    train_labels = np.asarray(get_labels_from_split(train_dataset), dtype=np.int64)
    test_labels = np.asarray(get_labels_from_split(test_dataset), dtype=np.int64)

    if train_labels.size == 0 or test_labels.size == 0:
        raise ValueError("Majority-class baseline requires non-empty train and test splits.")

    n_classes = max(3, int(max(train_labels.max(), test_labels.max())) + 1)
    train_class_counts = np.bincount(train_labels, minlength=n_classes)
    majority_class = int(np.argmax(train_class_counts))

    probabilities = np.zeros((test_labels.size, n_classes), dtype=np.float64)
    probabilities[:, majority_class] = 1.0

    metrics = calculate_metrics_from_probabilities(
        y_true=test_labels,
        y_prob=probabilities,
        average_loss=float("nan"),
        window_step_s=window_step_s,
        decision_threshold=0.5,
        smoothing_window=0,
    )
    metrics.update(
        {
            "majority_class": majority_class,
            "train_class_counts": train_class_counts.tolist(),
            "train_non_seizure_count": int(train_class_counts[0]),
            "train_seizure_count": int(train_class_counts[1:].sum()),
        }
    )
    return metrics
