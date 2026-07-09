"""
Evaluation metrics for EEG seizure detection.

This file contains metric-related logic only.
The training script calls these functions, but does not define them.

Why this separation is useful:
- train.py stays focused on training.
- evaluation logic can be reused later by evaluate.py.
- metrics remain consistent across CNN, EEGNet, TCN, and future models.
"""

import numpy as np
import torch


def compute_binary_auc(y_true, y_score):
    """
    Calculate binary AUC without requiring sklearn.

    y_true:
        Ground-truth labels.
        0 = non-seizure, 1 = seizure.

    y_score:
        Predicted probability score for class 1, seizure.

    Why AUC is useful:
        AUC shows how well the model separates seizure and non-seizure windows,
        even when the decision threshold changes.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # AUC is not mathematically valid if only one class exists.
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort prediction scores so ranks can be calculated.
    order = np.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = np.empty(len(y_score), dtype=float)

    # Assign average ranks for tied scores.
    i = 0
    while i < len(sorted_scores):
        j = i

        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1

        # Rank values are 1-based, so we add 2 before dividing.
        average_rank = (i + j + 2) / 2.0
        ranks[order[i:j + 1]] = average_rank

        i = j + 1

    positive_rank_sum = np.sum(ranks[y_true == 1])

    auc = (
        positive_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    ) / (n_pos * n_neg)

    return float(auc)

def apply_temporal_smoothing(predictions, window_size=5, threshold_fraction=0.5):
    """
    Apply a sliding window majority vote to smooth out jittery predictions.
    
    window_size: Number of consecutive windows to check.
    threshold_fraction: What fraction of windows must be 'seizure' to trigger an alarm.
                        0.5 means majority rule.
    """
    if window_size <= 1:
        return predictions

    # Create a kernel for the moving average
    kernel = np.ones(window_size) / window_size
    
    # Convolve the binary predictions (0s and 1s)
    # mode='same' keeps the output array length identical to input
    smoothed = np.convolve(predictions, kernel, mode='same')
    
    # Threshold the smoothed average to get final binary output
    return (smoothed >= threshold_fraction).astype(int)

def evaluate_model(model, loader, criterion, window_step_s=2.0, decision_threshold=0.5, smoothing_window=0):
    """
    Evaluate a trained model on validation or test data.

    This function does not update model weights.
    It calculates:
    - loss
    - accuracy
    - precision
    - recall/sensitivity
    - specificity
    - F1-score
    - AUC
    - confusion matrix values
    - false alarms per hour

    window_step_s:
        Time step between two consecutive windows.
        Example: 4-second windows with 50% overlap gives 2 seconds.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []
    all_scores = []

    # No gradients are needed during evaluation.
    # This makes evaluation faster and uses less memory.
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)

            batch_size = xb.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.softmax(outputs, dim=1)
            seizure_scores = probabilities[:, 1]

            predictions = (seizure_scores >= decision_threshold).long()

            all_true.extend(yb.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_scores.extend(seizure_scores.cpu().numpy())

    # --- POST-PROCESSING STEP ---
    y_true = np.asarray(all_true)
    y_score = np.asarray(all_scores)
    
    # Convert the full list of predictions to a numpy array first
    raw_pred = np.asarray(all_pred)
    
    # Apply temporal smoothing to the entire sequence if requested
    if smoothing_window > 0:
        y_pred = apply_temporal_smoothing(raw_pred, window_size=smoothing_window)
    else:
        y_pred = raw_pred
    # ----------------------------

    # Binary confusion matrix values.
    # Positive class = seizure.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced accuracy gives equal importance to seizure and non-seizure classes.
    # This is useful when the dataset is imbalanced.
    balanced_accuracy = (recall + specificity) / 2.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc = compute_binary_auc(y_true, y_score)
    avg_loss = total_loss / total_samples

    # Estimate evaluated EEG duration.
    # Example: 350 windows * 2 seconds = 700 seconds.
    evaluated_duration_seconds = total_samples * window_step_s
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0

    # False alarms are false positive seizure predictions.
    false_alarms_per_hour = (
        fp / evaluated_duration_hours
        if evaluated_duration_hours > 0
        else 0.0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "decision_threshold": decision_threshold,

        # Confusion matrix values
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        # False alarm analysis
        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }

def find_best_threshold(
    model,
    loader,
    criterion,
    window_step_s=2.0,
    metric_name="f1",
    min_threshold=0.05,
    max_threshold=0.95,
    step=0.01,
    smoothing_window=0
):
    """
    Find the best seizure decision threshold using validation data.

    Why threshold tuning is needed:
    - The default threshold 0.5 may not be suitable for imbalanced EEG data.
    - A low threshold can predict too many seizures.
    - A high threshold can reduce false alarms but may miss seizures.
    - Validation data is used to choose the threshold before final test evaluation.

    metric_name:
        The metric used to select the best threshold.
        For this project, balanced_accuracy is useful because it considers both:
        - sensitivity
        - specificity
    """

    thresholds = np.arange(
        min_threshold,
        max_threshold + step,
        step
    )

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_model(
            model=model,
            loader=loader,
            criterion=criterion,
            window_step_s=window_step_s,
            decision_threshold=float(threshold),
            smoothing_window=smoothing_window
        )

        # Calculate F2-score dynamically if requested
        # F2 weighs recall twice as high as precision.
        if metric_name == "f2":
            p = metrics["precision"]
            r = metrics["recall_sensitivity"]
            score = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
        else:
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
    """
    Extract labels from a dataset split.

    Why this helper exists:
        Class imbalance handling needs access to the training labels
        so class weights can be calculated from the TRAIN split only.

    Important implementation detail:
        This project now supports two split strategies:

        1. Window-level split:
           Uses torch.utils.data.random_split(), which returns Subset objects.
           A Subset stores:
               - split_dataset.dataset  -> the original TensorDataset
               - split_dataset.indices  -> indices belonging to this split

        2. Patient-level split:
           Builds TensorDataset objects directly for train/val/test after
           separating patients first. In this case, the split itself is
           already a TensorDataset and does NOT have .dataset or .indices.

        Therefore this function must handle both:
        - Subset
        - TensorDataset

    Returns:
        Numpy array of labels for the given split.
    """

    # Case 1:
    # random_split() returns a Subset object.
    # We need to use the stored indices to extract only the labels
    # belonging to this split from the parent TensorDataset.
    if hasattr(split_dataset, "dataset") and hasattr(split_dataset, "indices"):
        labels_tensor = split_dataset.dataset.tensors[1]
        split_labels = labels_tensor[split_dataset.indices]
        return split_labels.cpu().numpy()

    # Case 2:
    # Patient-level splitting builds a TensorDataset directly.
    # In that case, labels are already stored as the second tensor.
    elif hasattr(split_dataset, "tensors"):
        labels_tensor = split_dataset.tensors[1]
        return labels_tensor.cpu().numpy()

    # Any other dataset type is unsupported for now.
    raise TypeError(
        "Unsupported dataset type for label extraction. "
        f"Got: {type(split_dataset)}"
    )

def compute_majority_class_baseline(train_dataset, test_dataset, window_step_s=2.0):
    """
    Compute majority-class baseline performance.

    Majority-class baseline:
    1. Find the most common class in the training set.
    2. Predict that class for every test sample.

    Why this matters:
        It checks whether the CNN performs better than a simple class-imbalance rule.
    """

    train_labels = get_labels_from_split(train_dataset)
    test_labels = get_labels_from_split(test_dataset)

    # Count labels in the training data.
    # class 0 = non-seizure, class 1 = seizure.
    train_class_counts = np.bincount(train_labels, minlength=2)

    majority_class = int(np.argmax(train_class_counts))

    # Predict the same majority class for every test sample.
    baseline_predictions = np.full_like(
        test_labels,
        fill_value=majority_class
    )

    # AUC needs a score for seizure class.
    # Since this baseline is constant, the score is either all 0 or all 1.
    if majority_class == 1:
        baseline_scores = np.ones_like(test_labels, dtype=float)
    else:
        baseline_scores = np.zeros_like(test_labels, dtype=float)

    tp = int(np.sum((test_labels == 1) & (baseline_predictions == 1)))
    tn = int(np.sum((test_labels == 0) & (baseline_predictions == 0)))
    fp = int(np.sum((test_labels == 0) & (baseline_predictions == 1)))
    fn = int(np.sum((test_labels == 1) & (baseline_predictions == 0)))

    total_samples = len(test_labels)

    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc = compute_binary_auc(test_labels, baseline_scores)

    evaluated_duration_seconds = total_samples * window_step_s
    evaluated_duration_hours = evaluated_duration_seconds / 3600.0

    false_alarms_per_hour = (
        fp / evaluated_duration_hours
        if evaluated_duration_hours > 0
        else 0.0
    )

    return {
        "majority_class": majority_class,
        "train_non_seizure_count": int(train_class_counts[0]),
        "train_seizure_count": int(train_class_counts[1]),

        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "decision_threshold": 0.5,
        "f1": f1,
        "auc": auc,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "evaluated_duration_seconds": evaluated_duration_seconds,
        "evaluated_duration_hours": evaluated_duration_hours,
        "false_alarms_per_hour": false_alarms_per_hour,
    }