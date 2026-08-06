"""
Evaluation metrics for three-class EEG seizure-state classification.

Immutable class contract:
    0 = Interictal
    1 = Pre-Ictal
    2 = Ictal

This module keeps multiclass classification metrics separate from model
training. It also provides a validation-only alarm-threshold search.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_IDS = np.array([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("Interictal", "Pre-Ictal", "Ictal")


def _as_numpy_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    return array


def _validate_probabilities(y_prob: Any, n_samples: int) -> np.ndarray:
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if probabilities.shape != (n_samples, 3):
        raise ValueError(
            "y_prob must have shape (n_samples, 3) in class order "
            "[Interictal, Pre-Ictal, Ictal]; "
            f"got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob contains NaN or infinite values.")
    if np.any(probabilities < -1e-8):
        raise ValueError("y_prob contains negative values.")

    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum.")

    # Renormalise tiny floating-point drift, but do not hide malformed inputs.
    probabilities = probabilities / row_sums
    return probabilities


def compute_multiclass_auc(y_true: Any, y_prob: Any) -> float:
    """
    Compute macro one-vs-rest ROC AUC for the available classes.

    A class is included only when both positive and negative examples are
    present. This avoids returning a misleading value when a small split lacks
    one class. Malformed probability arrays raise a clear exception instead of
    silently returning 0.5.
    """
    truth = _as_numpy_1d(y_true, "y_true").astype(np.int64, copy=False)
    probabilities = _validate_probabilities(y_prob, len(truth))

    invalid_labels = sorted(set(np.unique(truth).tolist()) - set(CLASS_IDS.tolist()))
    if invalid_labels:
        raise ValueError(f"y_true contains invalid class IDs: {invalid_labels}.")

    auc_values = []
    for class_id in CLASS_IDS:
        binary_truth = (truth == class_id).astype(np.int64)
        if np.unique(binary_truth).size < 2:
            continue
        auc_values.append(
            float(roc_auc_score(binary_truth, probabilities[:, class_id]))
        )

    if not auc_values:
        return float("nan")
    return float(np.mean(auc_values))


def apply_temporal_smoothing(
    probabilities: Any,
    window_size: int = 5,
) -> np.ndarray:
    """
    Smooth class probabilities with a centred moving average.

    Smoothing class IDs directly is invalid for multiclass classification.
    This function smooths each probability column independently, preserves all
    three classes, and renormalises rows afterward.

    Important: the caller must avoid smoothing across patient or recording
    boundaries. Use window_size=1 or 0 when those boundaries are unavailable.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            "probabilities must have shape (n_samples, 3); "
            f"got {probabilities.shape}."
        )
    if window_size <= 1 or len(probabilities) <= 1:
        return _validate_probabilities(probabilities, len(probabilities))
    if window_size < 0:
        raise ValueError("window_size must be non-negative.")

    # An odd window keeps the centred interpretation predictable.
    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(probabilities, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)

    smoothed = np.empty_like(probabilities, dtype=np.float64)
    for class_id in CLASS_IDS:
        smoothed[:, class_id] = np.convolve(
            padded[:, class_id], kernel, mode="valid"
        )

    return _validate_probabilities(smoothed, len(smoothed))


def probabilities_to_predictions(
    y_prob: Any,
    decision_threshold: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Convert three-class probabilities into class predictions.

    When decision_threshold is None, use normal three-class argmax.

    Otherwise, treat Pre-Ictal and Ictal as alarm classes:
      * alarm probability = P(Pre-Ictal) + P(Ictal)
      * below threshold -> Interictal (0)
      * at/above threshold -> whichever alarm class has higher probability

    This gives the threshold a real, auditable meaning while retaining the
    distinction between classes 1 and 2.
    """
    probabilities = np.asarray(y_prob, dtype=np.float64)
    probabilities = _validate_probabilities(probabilities, len(probabilities))

    if decision_threshold is None:
        return np.argmax(probabilities, axis=1).astype(np.int64)

    threshold = float(decision_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")

    alarm_probability = probabilities[:, 1] + probabilities[:, 2]
    alarm_class = np.argmax(probabilities[:, 1:3], axis=1).astype(np.int64) + 1
    return np.where(alarm_probability >= threshold, alarm_class, 0).astype(np.int64)


def _collect_model_outputs(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_probabilities = []

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Each loader batch must provide (inputs, labels).")

            xb, yb = batch[0].to(device), batch[1].to(device)
            logits = model(xb)

            if logits.ndim != 2 or logits.shape[1] != 3:
                raise ValueError(
                    "Model output must have shape (batch, 3); "
                    f"got {tuple(logits.shape)}."
                )

            loss = criterion(logits, yb)
            batch_size = int(yb.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            probabilities = torch.softmax(logits, dim=1)
            all_true.append(yb.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation loader contained zero samples.")

    y_true = np.concatenate(all_true).astype(np.int64, copy=False)
    y_prob = np.concatenate(all_probabilities).astype(np.float64, copy=False)
    average_loss = total_loss / total_samples
    return average_loss, y_true, y_prob, total_samples


def compute_metrics_from_outputs(
    y_true: Any,
    y_prob: Any,
    *,
    average_loss: float = float("nan"),
    window_step_s: float = 2.0,
    decision_threshold: Optional[float] = 0.5,
    smoothing_window: int = 0,
) -> Dict[str, Any]:
    """Compute the complete project metric dictionary from labels and probabilities."""
    truth = _as_numpy_1d(y_true, "y_true").astype(np.int64, copy=False)
    probabilities = _validate_probabilities(y_prob, len(truth))

    invalid_labels = sorted(set(np.unique(truth).tolist()) - set(CLASS_IDS.tolist()))
    if invalid_labels:
        raise ValueError(f"y_true contains invalid class IDs: {invalid_labels}.")

    if smoothing_window > 1:
        probabilities = apply_temporal_smoothing(
            probabilities, window_size=smoothing_window
        )

    predictions = probabilities_to_predictions(
        probabilities, decision_threshold=decision_threshold
    )

    cm = confusion_matrix(truth, predictions, labels=CLASS_IDS)
    accuracy = float(np.mean(truth == predictions))
    balanced_accuracy = float(balanced_accuracy_score(truth, predictions))
    macro_precision = float(
        precision_score(truth, predictions, labels=CLASS_IDS, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(truth, predictions, labels=CLASS_IDS, average="macro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(truth, predictions, labels=CLASS_IDS, average="macro", zero_division=0)
    )
    macro_auc = compute_multiclass_auc(truth, probabilities)

    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=CLASS_IDS,
        zero_division=0,
    )

    # Binary alarm interpretation: class 0 = background, classes 1/2 = alarm.
    true_alarm = truth > 0
    predicted_alarm = predictions > 0

    tp = int(np.sum(true_alarm & predicted_alarm))
    tn = int(np.sum(~true_alarm & ~predicted_alarm))
    fp = int(np.sum(~true_alarm & predicted_alarm))
    fn = int(np.sum(true_alarm & ~predicted_alarm))

    alarm_sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    interictal_specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    total_samples = len(truth)
    evaluated_duration_seconds = float(total_samples * window_step_s)
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0
    false_alarms_per_hour = (
        fp / evaluated_duration_hours if evaluated_duration_hours > 0 else float("nan")
    )

    return {
        "loss": float(average_loss),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": macro_precision,
        "recall_sensitivity": macro_recall,  # retained for reporting compatibility
        "specificity": float(interictal_specificity),
        "f1": macro_f1,
        "auc": macro_auc,
        "decision_threshold": (
            float(decision_threshold) if decision_threshold is not None else float("nan")
        ),
        "alarm_sensitivity": float(alarm_sensitivity),
        "interictal_specificity": float(interictal_specificity),
        "preictal_precision": float(per_precision[1]),
        "preictal_recall": float(per_recall[1]),
        "preictal_f1": float(per_f1[1]),
        "ictal_precision": float(per_precision[2]),
        "ictal_recall": float(per_recall[2]),
        "ictal_f1": float(per_f1[2]),
        "per_class_precision": per_precision.astype(float).tolist(),
        "per_class_recall": per_recall.astype(float).tolist(),
        "per_class_f1": per_f1.astype(float).tolist(),
        "per_class_support": per_support.astype(int).tolist(),
        "predicted_class_counts": np.bincount(predictions, minlength=3).astype(int).tolist(),
        "true_class_counts": np.bincount(truth, minlength=3).astype(int).tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": float(false_alarms_per_hour),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    window_step_s: float = 2.0,
    decision_threshold: Optional[float] = 0.5,
    smoothing_window: int = 0,
) -> Dict[str, Any]:
    """Evaluate a model once and compute leakage-safe three-class metrics."""
    average_loss, y_true, y_prob, _ = _collect_model_outputs(model, loader, criterion)
    return compute_metrics_from_outputs(
        y_true,
        y_prob,
        average_loss=average_loss,
        window_step_s=window_step_s,
        decision_threshold=decision_threshold,
        smoothing_window=smoothing_window,
    )


def find_best_threshold(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    window_step_s: float = 2.0,
    metric_name: str = "f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
    smoothing_window: int = 0,
) -> Dict[str, Any]:
    """
    Select an alarm threshold using validation data only.

    Model inference is performed once. Threshold candidates are then evaluated
    against the same cached probabilities, avoiding dozens of repeated passes.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold cannot exceed max_threshold.")

    average_loss, y_true, y_prob, _ = _collect_model_outputs(model, loader, criterion)
    if smoothing_window > 1:
        y_prob = apply_temporal_smoothing(y_prob, smoothing_window)

    thresholds = np.arange(min_threshold, max_threshold + step / 2.0, step)
    best_threshold = None
    best_score = -np.inf
    best_metrics = None

    for threshold in thresholds:
        metrics = compute_metrics_from_outputs(
            y_true,
            y_prob,
            average_loss=average_loss,
            window_step_s=window_step_s,
            decision_threshold=float(threshold),
            smoothing_window=0,
        )

        if metric_name not in metrics:
            raise KeyError(f"Unknown threshold-selection metric: {metric_name}")
        score = float(metrics[metric_name])
        if not np.isfinite(score):
            continue

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("No finite threshold-selection score was produced.")

    return {
        "best_threshold": best_threshold,
        "best_score": float(best_score),
        "metric_name": metric_name,
        "best_metrics": best_metrics,
    }


def get_labels_from_split(split_dataset: Any) -> np.ndarray:
    """Retrieve labels from TensorDataset, Subset, or a lazy dataset API."""
    if hasattr(split_dataset, "get_all_labels"):
        labels = split_dataset.get_all_labels()
        return np.asarray(labels, dtype=np.int64)

    if hasattr(split_dataset, "labels"):
        labels = split_dataset.labels
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        return np.asarray(labels, dtype=np.int64)

    if hasattr(split_dataset, "dataset") and hasattr(split_dataset, "indices"):
        parent_labels = get_labels_from_split(split_dataset.dataset)
        return np.asarray(parent_labels)[np.asarray(split_dataset.indices)]

    if hasattr(split_dataset, "tensors") and len(split_dataset.tensors) >= 2:
        labels = split_dataset.tensors[1]
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        return np.asarray(labels, dtype=np.int64)

    raise TypeError(f"Unsupported dataset type: {type(split_dataset)}")


def compute_majority_class_baseline(
    train_dataset: Any,
    test_dataset: Any,
    window_step_s: float = 2.0,
) -> Dict[str, Any]:
    """Compute a real three-class majority baseline using the training majority."""
    train_labels = get_labels_from_split(train_dataset)
    test_labels = get_labels_from_split(test_dataset)

    train_class_counts = np.bincount(train_labels, minlength=3)
    majority_class = int(np.argmax(train_class_counts))

    predictions = np.full(len(test_labels), majority_class, dtype=np.int64)
    probabilities = np.zeros((len(test_labels), 3), dtype=np.float64)
    probabilities[:, majority_class] = 1.0

    metrics = compute_metrics_from_outputs(
        test_labels,
        probabilities,
        average_loss=float("nan"),
        window_step_s=window_step_s,
        decision_threshold=None,
        smoothing_window=0,
    )
    metrics.update(
        {
            "majority_class": majority_class,
            "train_interictal_count": int(train_class_counts[0]),
            "train_preictal_count": int(train_class_counts[1]),
            "train_ictal_count": int(train_class_counts[2]),
            # Legacy fields retained for existing report code.
            "train_non_seizure_count": int(train_class_counts[0]),
            "train_seizure_count": int(train_class_counts[1] + train_class_counts[2]),
        }
    )
    return metrics
