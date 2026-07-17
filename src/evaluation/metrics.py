"""Evaluation metrics for EEG seizure detection and prediction."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CLASS_NAMES = ("Interictal", "Pre-Ictal", "Ictal")


def compute_multiclass_auc(y_true, y_prob):
    """Return macro multiclass AUC, or NaN when it is not mathematically defined."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
        return float("nan")

    n_classes = y_prob.shape[1]
    present_classes = np.unique(y_true)

    # Multiclass ROC AUC needs every model output class to be represented in y_true.
    if n_classes < 2 or len(present_classes) != n_classes:
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


def probabilities_to_predictions(probabilities, decision_threshold=0.5):
    """
    Convert class probabilities into predictions using an alarm threshold.

    For the three-class task:
      0 = interictal
      1 = pre-ictal
      2 = ictal

    The threshold controls whether a window is considered seizure-related:
      P(pre-ictal) + P(ictal) >= threshold -> choose class 1 or 2
      otherwise -> class 0

    This makes validation threshold tuning meaningful while preserving the
    distinction between pre-ictal and ictal predictions.
    """
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected probabilities with shape (n_samples, n_classes>=2), "
            f"got {probabilities.shape}."
        )

    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")

    n_classes = probabilities.shape[1]

    if n_classes == 2:
        return (probabilities[:, 1] >= decision_threshold).astype(np.int64)

    alarm_probability = probabilities[:, 1:].sum(axis=1)
    alarm_subclass = 1 + np.argmax(probabilities[:, 1:], axis=1)

    return np.where(
        alarm_probability >= decision_threshold,
        alarm_subclass,
        0,
    ).astype(np.int64)


def apply_temporal_smoothing(predictions, window_size=5):
    """Apply a centred multiclass majority vote without collapsing class 2 into class 1."""
    predictions = np.asarray(predictions, dtype=np.int64)

    if window_size <= 1 or predictions.size == 0:
        return predictions.copy()

    if window_size % 2 == 0:
        raise ValueError("smoothing window_size must be odd so the vote is centred.")

    n_classes = int(predictions.max()) + 1
    radius = window_size // 2
    smoothed = predictions.copy()

    for index in range(predictions.size):
        start = max(0, index - radius)
        stop = min(predictions.size, index + radius + 1)
        local_values = predictions[start:stop]
        counts = np.bincount(local_values, minlength=n_classes)
        winners = np.flatnonzero(counts == counts.max())

        # Preserve the original centre class when it is part of a tie.
        if predictions[index] in winners:
            smoothed[index] = predictions[index]
        else:
            smoothed[index] = int(winners[0])

    return smoothed


def _macro_specificity(confusion):
    """Calculate one-vs-rest specificity for each class and return the macro mean."""
    confusion = np.asarray(confusion, dtype=np.int64)
    total = confusion.sum()
    specificities = []

    for class_index in range(confusion.shape[0]):
        tp = confusion[class_index, class_index]
        fn = confusion[class_index, :].sum() - tp
        fp = confusion[:, class_index].sum() - tp
        tn = total - tp - fn - fp
        denominator = tn + fp
        specificities.append(tn / denominator if denominator > 0 else np.nan)

    return float(np.nanmean(specificities)), [float(value) for value in specificities]


def _calculate_metrics(
    y_true,
    y_pred,
    y_prob,
    average_loss,
    window_step_s,
    decision_threshold,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty dataset split.")

    n_classes = y_prob.shape[1]
    labels = np.arange(n_classes)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    precision = float(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    recall = float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    specificity, per_class_specificity = _macro_specificity(confusion)

    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    # Binary alarm-level aggregation used for false alarms/hour and TP/TN/FP/FN.
    true_alarm = (y_true > 0).astype(np.int64)
    predicted_alarm = (y_pred > 0).astype(np.int64)
    alarm_confusion = confusion_matrix(true_alarm, predicted_alarm, labels=[0, 1])
    tn, fp, fn, tp = alarm_confusion.ravel()

    evaluated_duration_seconds = float(y_true.size * window_step_s)
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0
    false_alarms_per_hour = (
        float(fp / evaluated_duration_hours)
        if evaluated_duration_hours > 0
        else 0.0
    )

    return {
        "loss": float(average_loss),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": compute_multiclass_auc(y_true, y_prob),
        "decision_threshold": float(decision_threshold),
        "confusion_matrix": confusion.tolist(),
        "alarm_confusion_matrix": alarm_confusion.tolist(),
        "per_class_recall": [float(value) for value in per_class_recall],
        "per_class_specificity": per_class_specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }


def evaluate_model(
    model,
    loader,
    criterion,
    window_step_s=2.0,
    decision_threshold=0.5,
    smoothing_window=0,
):
    """Evaluate a trained multiclass model on a validation or test loader."""
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

            probabilities = torch.softmax(outputs, dim=1)
            all_true.extend(yb.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation loader contains zero samples.")

    y_true = np.asarray(all_true, dtype=np.int64)
    y_prob = np.asarray(all_scores, dtype=np.float64)
    y_pred = probabilities_to_predictions(y_prob, decision_threshold)

    if smoothing_window > 1:
        y_pred = apply_temporal_smoothing(y_pred, window_size=smoothing_window)

    return _calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        average_loss=total_loss / total_samples,
        window_step_s=window_step_s,
        decision_threshold=decision_threshold,
    )


def find_best_threshold(
    model,
    loader,
    criterion,
    window_step_s=2.0,
    metric_name="f1",
    min_threshold=0.05,
    max_threshold=0.95,
    step=0.01,
    smoothing_window=0,
):
    """Select the best alarm threshold using the validation set only."""
    thresholds = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_model(
            model=model,
            loader=loader,
            criterion=criterion,
            window_step_s=window_step_s,
            decision_threshold=float(threshold),
            smoothing_window=smoothing_window,
        )
        score = float(metrics.get(metric_name, np.nan))

        if not np.isfinite(score):
            continue

        is_better = score > best_score + 1e-12
        is_equal_but_safer = (
            abs(score - best_score) <= 1e-12
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        )

        if is_better or is_equal_but_safer:
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
    }


def get_labels_from_split(split_dataset):
    """Extract labels from TensorDataset or torch.utils.data.Subset."""
    if hasattr(split_dataset, "dataset") and hasattr(split_dataset, "indices"):
        labels_tensor = split_dataset.dataset.tensors[1]
        split_labels = labels_tensor[split_dataset.indices]
        return split_labels.cpu().numpy()

    if hasattr(split_dataset, "tensors"):
        labels_tensor = split_dataset.tensors[1]
        return labels_tensor.cpu().numpy()

    raise TypeError(f"Unsupported dataset type: {type(split_dataset)}")


def compute_majority_class_baseline(train_dataset, test_dataset, window_step_s=2.0):
    """Evaluate a true majority-class baseline with the same metric definitions as the model."""
    train_labels = np.asarray(get_labels_from_split(train_dataset), dtype=np.int64)
    test_labels = np.asarray(get_labels_from_split(test_dataset), dtype=np.int64)

    if train_labels.size == 0 or test_labels.size == 0:
        raise ValueError("Majority-class baseline requires non-empty train and test splits.")

    n_classes = max(3, int(max(train_labels.max(), test_labels.max())) + 1)
    train_class_counts = np.bincount(train_labels, minlength=n_classes)
    majority_class = int(np.argmax(train_class_counts))
    baseline_predictions = np.full(test_labels.shape, majority_class, dtype=np.int64)

    baseline_probabilities = np.zeros((test_labels.size, n_classes), dtype=np.float64)
    baseline_probabilities[:, majority_class] = 1.0

    metrics = _calculate_metrics(
        y_true=test_labels,
        y_pred=baseline_predictions,
        y_prob=baseline_probabilities,
        average_loss=float("nan"),
        window_step_s=window_step_s,
        decision_threshold=0.5,
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
