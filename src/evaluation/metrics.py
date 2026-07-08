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


def evaluate_model(model, loader, criterion, window_step_s=2.0, decision_threshold=0.5):
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

            # Convert logits into probabilities.
            # Class 1 probability is used as the seizure score for AUC.
            probabilities = torch.softmax(outputs, dim=1)
            seizure_scores = probabilities[:, 1]

            # Convert seizure probability into final class prediction.
            # A threshold of 0.5 is the default, but later we tune this using validation data.
            predictions = (seizure_scores >= decision_threshold).long()

            all_true.extend(yb.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_scores.extend(seizure_scores.cpu().numpy())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_score = np.asarray(all_scores)

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
    metric_name="balanced_accuracy",
    min_threshold=0.05,
    max_threshold=0.95,
    step=0.01
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
            decision_threshold=float(threshold)
        )

        score = metrics[metric_name]

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
    Extract labels from a dataset split created by random_split().

    random_split() returns a Subset object.
    The labels are still stored in the original TensorDataset.
    """

    labels_tensor = split_dataset.dataset.tensors[1]

    labels = [
        int(labels_tensor[index].item())
        for index in split_dataset.indices
    ]

    return np.asarray(labels)


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