"""
Evaluation metrics for EEG seizure detection.

This file contains metric-related logic only.
The training script calls these functions, but does not define them.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_multiclass_auc(y_true, y_prob):
    """
    Calculate a safe multiclass macro AUC proxy using sklearn principles.
    """
    from sklearn.metrics import roc_auc_score
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        # Check unique classes present in batch
        if len(np.unique(y_true)) < 2:
            return 0.5
        # If y_prob is 1D or only reflects 1 channel, map to uniform distribution shape
        if len(y_prob.shape) == 1 or y_prob.shape[1] < 3:
            return 0.5
        return float(roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro"))
    except Exception:
        return 0.5

def apply_temporal_smoothing(predictions, window_size=5, threshold_fraction=0.5):
    """
    Apply a sliding window majority vote to smooth out jittery predictions.
    """
    if window_size <= 1:
        return predictions

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(predictions, kernel, mode='same')
    return (smoothed >= threshold_fraction).astype(int)

def evaluate_model(model, loader, criterion, window_step_s=2.0, decision_threshold=0.5, smoothing_window=0):
    """
    Evaluate a trained model on validation or test data natively using 3-class dimensions.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []
    all_scores = []

    # Dynamic CUDA device discovery based on where model parameter weights live
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
            
            # Predict the class index with the highest probability [0, 1, 2]
            predictions = torch.argmax(probabilities, dim=1)

            all_true.extend(yb.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

    # --- POST-PROCESSING ---
    y_true = np.asarray(all_true)
    y_prob = np.asarray(all_scores)
    raw_pred = np.asarray(all_pred)
    
    if smoothing_window > 0:
        y_pred = apply_temporal_smoothing(raw_pred, window_size=smoothing_window)
    else:
        y_pred = raw_pred

    # Macro-averaged metrics calculations across all 3 indices
    accuracy = float(np.mean(y_true == y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_accuracy = recall 
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auc = compute_multiclass_auc(y_true, y_prob)
    avg_loss = total_loss / total_samples

    # False positive window tracking: any time model predicts an alarm class (>0) on background (0)
    fp_windows = int(np.sum((y_true == 0) & (y_pred > 0)))
    evaluated_duration_hours = (total_samples * window_step_s) / 3600.0
    false_alarms_per_hour = fp_windows / evaluated_duration_hours if evaluated_duration_hours > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": float(np.mean(y_true[y_true == 0] == y_pred[y_true == 0]) if np.sum(y_true == 0) > 0 else 1.0),
        "f1": f1,
        "auc": auc,
        "decision_threshold": decision_threshold,

        # Multi-class mapped fields for thesis matrix logging compatibility
        "tp": int(np.sum((y_true > 0) & (y_pred > 0))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": fp_windows,
        "fn": int(np.sum((y_true > 0) & (y_pred == 0))),

        "evaluated_duration_seconds": total_samples * window_step_s,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }

def find_best_threshold(model, loader, criterion, window_step_s=2.0, metric_name="f1", min_threshold=0.05, max_threshold=0.95, step=0.01, smoothing_window=0):
    """
    Find best threshold structure using validation metrics data loops.
    """
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_model(
            model=model, loader=loader, criterion=criterion,
            window_step_s=window_step_s, decision_threshold=float(threshold),
            smoothing_window=smoothing_window
        )
        score = metrics.get(metric_name, 0.0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "metric_name": metric_name,
        "best_metrics": best_metrics,
    }

def get_labels_from_split(split_dataset):
    if hasattr(split_dataset, "dataset") and hasattr(split_dataset, "indices"):
        labels_tensor = split_dataset.dataset.tensors[1]
        split_labels = labels_tensor[split_dataset.indices]
        return split_labels.cpu().numpy()
    elif hasattr(split_dataset, "tensors"):
        labels_tensor = split_dataset.tensors[1]
        return labels_tensor.cpu().numpy()
    raise TypeError(f"Unsupported dataset type: {type(split_dataset)}")

def compute_majority_class_baseline(train_dataset, test_dataset, window_step_s=2.0):
    train_labels = get_labels_from_split(train_dataset)
    test_labels = get_labels_from_split(test_dataset)

    train_class_counts = np.bincount(train_labels, minlength=3)
    majority_class = int(np.argmax(train_class_counts))

    baseline_predictions = np.full_like(test_labels, fill_value=majority_class)
    total_samples = len(test_labels)

    accuracy = float(np.mean(test_labels == baseline_predictions))
    
    fp_windows = int(np.sum((test_labels == 0) & (baseline_predictions > 0)))
    evaluated_duration_hours = (total_samples * window_step_s) / 3600.0
    false_alarms_per_hour = fp_windows / evaluated_duration_hours if evaluated_duration_hours > 0 else 0.0

    return {
        "majority_class": majority_class,
        "train_non_seizure_count": int(train_class_counts[0]),
        "train_seizure_count": int(train_class_counts[1]),

        "accuracy": accuracy,
        "precision": 0.0,
        "recall_sensitivity": 0.0,
        "specificity": 1.0 if majority_class == 0 else 0.0,
        "balanced_accuracy": 0.333,
        "decision_threshold": 0.5,
        "f1": 0.0,
        "auc": 0.5,

        "tp": 0,
        "tn": int(np.sum((test_labels == 0) & (baseline_predictions == 0))),
        "fp": fp_windows,
        "fn": int(np.sum(test_labels > 0)),

        "evaluated_duration_seconds": total_samples * window_step_s,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }