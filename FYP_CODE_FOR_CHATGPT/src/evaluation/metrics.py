"""
Evaluation metrics for three-class EEG seizure-state classification.

Immutable class contract:
    0 = Interictal
    1 = Pre-Ictal
    2 = Ictal

Primary task:
    Three-class prediction using argmax.

Secondary alarm task:
    Interictal versus alarm, where:
        P(alarm) = P(Pre-Ictal) + P(Ictal)

Alarm thresholds are selected using validation data only.
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

    return probabilities / row_sums


def compute_multiclass_auc(y_true: Any, y_prob: Any) -> float:
    """Compute macro one-vs-rest ROC AUC for classes available in the split."""
    truth = _as_numpy_1d(y_true, "y_true").astype(np.int64, copy=False)
    probabilities = _validate_probabilities(y_prob, len(truth))

    invalid_labels = sorted(
        set(np.unique(truth).tolist()) - set(CLASS_IDS.tolist())
    )
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

    return float(np.mean(auc_values)) if auc_values else float("nan")


def apply_temporal_smoothing(
    probabilities: Any,
    window_size: int = 5,
) -> np.ndarray:
    """
    Smooth each probability column independently with a centred moving average.

    The caller must not smooth across patient or recording boundaries.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            "probabilities must have shape (n_samples, 3); "
            f"got {probabilities.shape}."
        )

    if window_size < 0:
        raise ValueError("window_size must be non-negative.")

    if window_size <= 1 or len(probabilities) <= 1:
        return _validate_probabilities(probabilities, len(probabilities))

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(probabilities, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)

    smoothed = np.empty_like(probabilities, dtype=np.float64)
    for class_id in CLASS_IDS:
        smoothed[:, class_id] = np.convolve(
            padded[:, class_id],
            kernel,
            mode="valid",
        )

    return _validate_probabilities(smoothed, len(smoothed))


def probabilities_to_predictions(
    y_prob: Any,
    decision_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Convert probabilities into three-class predictions.

    None:
        Standard three-class argmax.

    Numeric threshold:
        Below threshold -> Interictal.
        At/above threshold -> whichever alarm class has the larger probability.

    This helper is retained for compatibility. Primary multiclass metrics always
    use argmax inside compute_metrics_from_outputs().
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

    return np.where(
        alarm_probability >= threshold,
        alarm_class,
        0,
    ).astype(np.int64)


def _collect_model_outputs(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Run model inference once and return loss, labels and probabilities."""
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

            xb = batch[0].to(device)
            yb = batch[1].to(device)

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

    return total_loss / total_samples, y_true, y_prob, total_samples


def _compute_alarm_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    window_step_s: float,
) -> Dict[str, Any]:
    """Compute binary interictal-versus-alarm metrics."""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Alarm threshold must be between 0 and 1.")

    true_alarm = truth > 0
    alarm_probability = probabilities[:, 1] + probabilities[:, 2]
    predicted_alarm = alarm_probability >= threshold

    tp = int(np.sum(true_alarm & predicted_alarm))
    tn = int(np.sum(~true_alarm & ~predicted_alarm))
    fp = int(np.sum(~true_alarm & predicted_alarm))
    fn = int(np.sum(true_alarm & ~predicted_alarm))

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    if np.isfinite(sensitivity) and precision + sensitivity > 0:
        alarm_f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
    else:
        alarm_f1 = 0.0

    if np.isfinite(sensitivity) and np.isfinite(specificity):
        alarm_balanced_accuracy = (sensitivity + specificity) / 2.0
        alarm_youden_j = sensitivity + specificity - 1.0
    else:
        alarm_balanced_accuracy = float("nan")
        alarm_youden_j = float("nan")

    binary_truth = true_alarm.astype(np.int64)
    if np.unique(binary_truth).size == 2:
        alarm_auc = float(roc_auc_score(binary_truth, alarm_probability))
    else:
        alarm_auc = float("nan")

    total_samples = len(truth)
    evaluated_duration_seconds = float(total_samples * window_step_s)
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0

    false_alarms_per_hour = (
        fp / evaluated_duration_hours
        if evaluated_duration_hours > 0
        else float("nan")
    )

    return {
        "alarm_decision_threshold": threshold,
        "alarm_accuracy": float(np.mean(true_alarm == predicted_alarm)),
        "alarm_balanced_accuracy": float(alarm_balanced_accuracy),
        "alarm_precision": float(precision),
        "alarm_recall_sensitivity": float(sensitivity),
        "alarm_sensitivity": float(sensitivity),
        "alarm_specificity": float(specificity),
        "alarm_f1": float(alarm_f1),
        "alarm_auc": float(alarm_auc),
        "alarm_youden_j": float(alarm_youden_j),
        "alarm_prediction_rate": float(np.mean(predicted_alarm)),
        "alarm_true_rate": float(np.mean(true_alarm)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "alarm_confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": float(false_alarms_per_hour),
    }


def compute_metrics_from_outputs(
    y_true: Any,
    y_prob: Any,
    *,
    average_loss: float = float("nan"),
    window_step_s: float = 2.0,
    decision_threshold: Optional[float] = 0.5,
    smoothing_window: int = 0,
) -> Dict[str, Any]:
    """
    Compute primary multiclass metrics and secondary alarm metrics.

    Primary multiclass metrics are always based on argmax and therefore remain
    independent of the alarm operating threshold.
    """
    truth = _as_numpy_1d(y_true, "y_true").astype(np.int64, copy=False)
    probabilities = _validate_probabilities(y_prob, len(truth))

    invalid_labels = sorted(
        set(np.unique(truth).tolist()) - set(CLASS_IDS.tolist())
    )
    if invalid_labels:
        raise ValueError(f"y_true contains invalid class IDs: {invalid_labels}.")

    if window_step_s <= 0:
        raise ValueError("window_step_s must be positive.")

    if smoothing_window > 1:
        probabilities = apply_temporal_smoothing(
            probabilities,
            window_size=smoothing_window,
        )

    # Primary task: threshold-independent three-class argmax.
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)

    cm = confusion_matrix(truth, predictions, labels=CLASS_IDS)
    accuracy = float(np.mean(truth == predictions))

    # One-vs-rest specificity for each class.
    per_class_specificity = []

    for class_index in CLASS_IDS:
        true_positive = cm[class_index, class_index]
        false_negative = cm[class_index, :].sum() - true_positive
        false_positive = cm[:, class_index].sum() - true_positive
        true_negative = cm.sum() - (
            true_positive
            + false_negative
            + false_positive
        )

        denominator = true_negative + false_positive

        specificity_value = (
            true_negative / denominator
            if denominator > 0
            else float("nan")
        )

        per_class_specificity.append(
            float(specificity_value)
        )

    finite_specificities = [
        value for value in per_class_specificity if np.isfinite(value)
    ]
    macro_specificity = (
        float(np.mean(finite_specificities))
        if finite_specificities
        else float("nan")
    )

    balanced_accuracy = float(balanced_accuracy_score(truth, predictions))
    macro_precision = float(
        precision_score(
            truth,
            predictions,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )
    )
    macro_recall = float(
        recall_score(
            truth,
            predictions,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )
    )
    macro_f1 = float(
        f1_score(
            truth,
            predictions,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )
    )
    macro_auc = compute_multiclass_auc(truth, probabilities)

    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            truth,
            predictions,
            labels=CLASS_IDS,
            zero_division=0,
        )
    )

    metrics: Dict[str, Any] = {
        "loss": float(average_loss),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": macro_precision,
        "recall_sensitivity": macro_recall,
        "specificity": macro_specificity,
        "macro_specificity": macro_specificity,
        "f1": macro_f1,
        "auc": macro_auc,
        "decision_threshold": (
            float(decision_threshold)
            if decision_threshold is not None
            else float("nan")
        ),
        "preictal_precision": float(per_precision[1]),
        "preictal_recall": float(per_recall[1]),
        "preictal_f1": float(per_f1[1]),
        "ictal_precision": float(per_precision[2]),
        "ictal_recall": float(per_recall[2]),
        "ictal_f1": float(per_f1[2]),
        "per_class_precision": per_precision.astype(float).tolist(),
        "per_class_recall": per_recall.astype(float).tolist(),
        "per_class_specificity": per_class_specificity,
        "per_class_f1": per_f1.astype(float).tolist(),
        "per_class_support": per_support.astype(int).tolist(),
        "predicted_class_counts": np.bincount(
            predictions,
            minlength=3,
        ).astype(int).tolist(),
        "true_class_counts": np.bincount(
            truth,
            minlength=3,
        ).astype(int).tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }

    if decision_threshold is not None:
        alarm_metrics = _compute_alarm_metrics(
            truth,
            probabilities,
            float(decision_threshold),
            window_step_s,
        )
        metrics.update(alarm_metrics)

        # Compatibility aliases used by existing reporting code.
        # "specificity" remains the primary three-class macro specificity.
        metrics["alarm_sensitivity"] = alarm_metrics[
            "alarm_recall_sensitivity"
        ]
        metrics["interictal_specificity"] = alarm_metrics[
            "alarm_specificity"
        ]
    else:
        evaluated_duration_seconds = float(len(truth) * window_step_s)
        evaluated_duration_hours = evaluated_duration_seconds / 3600.0
        metrics.update(
            {
                "alarm_decision_threshold": float("nan"),
                "alarm_accuracy": float("nan"),
                "alarm_balanced_accuracy": float("nan"),
                "alarm_precision": float("nan"),
                "alarm_recall_sensitivity": float("nan"),
                "alarm_specificity": float("nan"),
                "alarm_f1": float("nan"),
                "alarm_auc": float("nan"),
                "alarm_youden_j": float("nan"),
                "alarm_prediction_rate": float("nan"),
                "alarm_true_rate": float(np.mean(truth > 0)),
                "alarm_sensitivity": float("nan"),
                "interictal_specificity": float("nan"),
                "alarm_confusion_matrix": [[0, 0], [0, 0]],
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "evaluated_duration_seconds": evaluated_duration_seconds,
                "evaluated_duration_hours": evaluated_duration_hours,
                "false_alarms_per_hour": float("nan"),
            }
        )

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    window_step_s: float = 2.0,
    decision_threshold: Optional[float] = 0.5,
    smoothing_window: int = 0,
) -> Dict[str, Any]:
    """Evaluate a model once and compute primary and secondary metrics."""
    average_loss, y_true, y_prob, _ = _collect_model_outputs(
        model,
        loader,
        criterion,
    )

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
    selection_policy: str = "balanced_accuracy",
    min_specificity: float = 0.80,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
    smoothing_window: int = 0,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Select a validation-only binary alarm threshold.

    Supported policies:
        balanced_accuracy
        youden_j
        f1
        specificity_constrained

    metric_name is retained as a compatibility alias for older callers.
    """
    if step <= 0:
        raise ValueError("step must be positive.")

    if min_threshold > max_threshold:
        raise ValueError("min_threshold cannot exceed max_threshold.")

    if not 0.0 <= min_specificity <= 1.0:
        raise ValueError("min_specificity must be between 0 and 1.")

    if metric_name is not None:
        compatibility_map = {
            "balanced_accuracy": "balanced_accuracy",
            "alarm_balanced_accuracy": "balanced_accuracy",
            "youden_j": "youden_j",
            "alarm_youden_j": "youden_j",
            "f1": "f1",
            "alarm_f1": "f1",
        }
        if metric_name not in compatibility_map:
            raise ValueError(
                f"Unsupported metric_name: {metric_name}"
            )
        selection_policy = compatibility_map[metric_name]

    valid_policies = {
        "balanced_accuracy",
        "youden_j",
        "f1",
        "specificity_constrained",
    }
    if selection_policy not in valid_policies:
        raise ValueError(
            f"selection_policy must be one of {sorted(valid_policies)}, "
            f"got {selection_policy!r}."
        )

    average_loss, y_true, y_prob, _ = _collect_model_outputs(
        model,
        loader,
        criterion,
    )

    if smoothing_window > 1:
        y_prob = apply_temporal_smoothing(
            y_prob,
            window_size=smoothing_window,
        )

    thresholds = np.arange(
        min_threshold,
        max_threshold + step / 2.0,
        step,
    )

    sweep_rows = []
    candidates = []

    for threshold in thresholds:
        alarm_metrics = _compute_alarm_metrics(
            y_true,
            y_prob,
            float(threshold),
            window_step_s,
        )

        row = {
            "threshold": float(threshold),
            **alarm_metrics,
        }
        sweep_rows.append(row)

        if selection_policy == "balanced_accuracy":
            score = alarm_metrics["alarm_balanced_accuracy"]
            eligible = True
        elif selection_policy == "youden_j":
            score = alarm_metrics["alarm_youden_j"]
            eligible = True
        elif selection_policy == "f1":
            score = alarm_metrics["alarm_f1"]
            eligible = True
        else:
            score = alarm_metrics["alarm_recall_sensitivity"]
            eligible = (
                np.isfinite(alarm_metrics["alarm_specificity"])
                and alarm_metrics["alarm_specificity"] >= min_specificity
            )

        if eligible and np.isfinite(score):
            candidates.append(
                (
                    float(score),
                    float(alarm_metrics["alarm_specificity"]),
                    -float(alarm_metrics["alarm_prediction_rate"]),
                    float(threshold),
                    alarm_metrics,
                )
            )

    specificity_constraint_met = True

    if selection_policy == "specificity_constrained" and not candidates:
        specificity_constraint_met = False

        # Fallback: highest specificity, then sensitivity, then lower alarm rate.
        for row in sweep_rows:
            specificity = row["alarm_specificity"]
            sensitivity = row["alarm_recall_sensitivity"]

            if not np.isfinite(specificity):
                continue

            candidates.append(
                (
                    float(specificity),
                    float(sensitivity)
                    if np.isfinite(sensitivity)
                    else -np.inf,
                    -float(row["alarm_prediction_rate"]),
                    float(row["threshold"]),
                    {
                        key: value
                        for key, value in row.items()
                        if key != "threshold"
                    },
                )
            )

    if not candidates:
        raise RuntimeError("No finite threshold-selection score was produced.")

    # Deterministic tie-break:
    # 1. selection score
    # 2. specificity
    # 3. lower predicted alarm rate
    # 4. higher threshold
    best = max(candidates, key=lambda item: item[:4])

    best_score, _, _, best_threshold, best_alarm_metrics = best

    # Add threshold-independent primary metrics to the selected alarm metrics.
    best_metrics = compute_metrics_from_outputs(
        y_true,
        y_prob,
        average_loss=average_loss,
        window_step_s=window_step_s,
        decision_threshold=best_threshold,
        smoothing_window=0,
    )

    return {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "selection_policy": selection_policy,
        "metric_name": selection_policy,
        "best_metrics": best_metrics,
        "sweep_rows": sweep_rows,
        "specificity_constraint_met": bool(specificity_constraint_met),
        "min_specificity": float(min_specificity),
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

    probabilities = np.zeros((len(test_labels), 3), dtype=np.float64)
    probabilities[:, majority_class] = 1.0

    metrics = compute_metrics_from_outputs(
        test_labels,
        probabilities,
        average_loss=float("nan"),
        window_step_s=window_step_s,
        decision_threshold=0.5,
        smoothing_window=0,
    )

    metrics.update(
        {
            "majority_class": majority_class,
            "train_interictal_count": int(train_class_counts[0]),
            "train_preictal_count": int(train_class_counts[1]),
            "train_ictal_count": int(train_class_counts[2]),
            "train_non_seizure_count": int(train_class_counts[0]),
            "train_seizure_count": int(
                train_class_counts[1] + train_class_counts[2]
            ),
        }
    )

    return metrics
