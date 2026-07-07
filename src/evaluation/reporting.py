"""
Reporting utilities for EEG seizure detection experiments.

This file handles:
- logging metrics to MLflow
- saving final test reports
- saving CNN vs baseline result tables

Why this is separate:
    train.py should focus on training.
    Reporting and artifact saving are supporting responsibilities.
"""

import csv
from pathlib import Path

import mlflow
import numpy as np


def log_evaluation_metrics_to_mlflow(prefix, metrics, step=None):
    """
    Log evaluation metrics to MLflow using a consistent prefix.

    Example:
        prefix="val"      -> val_accuracy, val_f1
        prefix="test"     -> test_accuracy, test_f1
        prefix="baseline" -> baseline_accuracy, baseline_f1
    """

    metric_keys = [
        "loss",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluated_duration_seconds",
        "evaluated_duration_hours",
        "false_alarms_per_hour",
    ]

    mlflow_metrics = {}

    for key in metric_keys:
        if key not in metrics:
            continue

        value = float(metrics[key])

        # AUC can be NaN if a split contains only one class.
        # MLflow should only receive valid numeric values.
        if not np.isfinite(value):
            continue

        mlflow_metrics[f"{prefix}_{key}"] = value

    if step is None:
        mlflow.log_metrics(mlflow_metrics)
    else:
        mlflow.log_metrics(mlflow_metrics, step=step)


def save_test_metrics_report(metrics, output_dir, run_name, baseline_metrics=None):
    """
    Save final test results as a readable text report.

    This report is useful for:
    - thesis writing
    - viva preparation
    - MLflow artifacts
    - final result discussion
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{run_name}_test_metrics.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Final Test Evaluation Report\n")
        file.write("============================\n\n")

        file.write(f"Test loss                    : {metrics['loss']:.4f}\n")
        file.write(f"Test accuracy                : {metrics['accuracy']:.4f}\n")
        file.write(f"Precision                    : {metrics['precision']:.4f}\n")
        file.write(f"Recall / Sensitivity         : {metrics['recall_sensitivity']:.4f}\n")
        file.write(f"Specificity                  : {metrics['specificity']:.4f}\n")
        file.write(f"F1-score                     : {metrics['f1']:.4f}\n")

        if np.isfinite(metrics["auc"]):
            file.write(f"AUC                          : {metrics['auc']:.4f}\n")
        else:
            file.write("AUC                          : Not available\n")

        file.write(f"Evaluated duration hours     : {metrics['evaluated_duration_hours']:.4f}\n")
        file.write(f"False alarms per hour        : {metrics['false_alarms_per_hour']:.4f}\n")

        file.write("\nConfusion Matrix\n")
        file.write("----------------\n")
        file.write("                 Predicted\n")
        file.write("               Non-Seizure  Seizure\n")
        file.write(f"Actual Non-Seizure   {metrics['tn']:>6}   {metrics['fp']:>6}\n")
        file.write(f"Actual Seizure       {metrics['fn']:>6}   {metrics['tp']:>6}\n")

        if baseline_metrics is not None:
            file.write("\nMajority-Class Baseline\n")
            file.write("-----------------------\n")
            file.write(
                f"Majority class                : {baseline_metrics['majority_class']} "
                f"(0=non-seizure, 1=seizure)\n"
            )
            file.write(f"Baseline accuracy             : {baseline_metrics['accuracy']:.4f}\n")
            file.write(f"Baseline precision            : {baseline_metrics['precision']:.4f}\n")
            file.write(f"Baseline recall/sensitivity   : {baseline_metrics['recall_sensitivity']:.4f}\n")
            file.write(f"Baseline specificity          : {baseline_metrics['specificity']:.4f}\n")
            file.write(f"Baseline F1-score             : {baseline_metrics['f1']:.4f}\n")

            if np.isfinite(baseline_metrics["auc"]):
                file.write(f"Baseline AUC                  : {baseline_metrics['auc']:.4f}\n")
            else:
                file.write("Baseline AUC                  : Not available\n")

            file.write(
                f"Baseline false alarms/hour    : "
                f"{baseline_metrics['false_alarms_per_hour']:.4f}\n"
            )

            file.write("\nModel vs Baseline\n")
            file.write("-----------------\n")
            file.write(
                f"Accuracy difference           : "
                f"{metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}\n"
            )
            file.write(
                f"F1-score difference           : "
                f"{metrics['f1'] - baseline_metrics['f1']:+.4f}\n"
            )
            file.write(
                f"False alarms/hour difference  : "
                f"{metrics['false_alarms_per_hour'] - baseline_metrics['false_alarms_per_hour']:+.4f}\n"
            )

        file.write("\nReported accuracy source: held-out test set\n")
        file.write("False alarm type       : window-level false positives per hour\n")

    return report_path


def save_cnn_baseline_results_table(test_metrics, baseline_metrics, output_dir, run_name):
    """
    Save CNN vs majority-class baseline results as CSV and Markdown.

    CSV:
        Useful for later analysis and final tables.

    Markdown:
        Useful for README, thesis draft, and documentation.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{run_name}_cnn_baseline_results.csv"
    md_path = output_dir / f"{run_name}_cnn_baseline_results.md"

    fieldnames = [
        "model",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "auc",
        "false_alarms_per_hour",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    rows = [
        {
            "model": "SeizureCNN",
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall_sensitivity": test_metrics["recall_sensitivity"],
            "specificity": test_metrics["specificity"],
            "f1_score": test_metrics["f1"],
            "auc": test_metrics["auc"],
            "false_alarms_per_hour": test_metrics["false_alarms_per_hour"],
            "tp": test_metrics["tp"],
            "tn": test_metrics["tn"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
        },
        {
            "model": "MajorityClassBaseline",
            "accuracy": baseline_metrics["accuracy"],
            "precision": baseline_metrics["precision"],
            "recall_sensitivity": baseline_metrics["recall_sensitivity"],
            "specificity": baseline_metrics["specificity"],
            "f1_score": baseline_metrics["f1"],
            "auc": baseline_metrics["auc"],
            "false_alarms_per_hour": baseline_metrics["false_alarms_per_hour"],
            "tp": baseline_metrics["tp"],
            "tn": baseline_metrics["tn"],
            "fp": baseline_metrics["fp"],
            "fn": baseline_metrics["fn"],
        },
    ]

    # Save machine-readable CSV file.
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    # Save human-readable Markdown table.
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write("# CNN Baseline Result Table\n\n")
        md_file.write("| Model | Accuracy | Precision | Recall/Sensitivity | Specificity | F1-score | AUC | False Alarms/Hour | TP | TN | FP | FN |\n")
        md_file.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for row in rows:
            auc_value = row["auc"]

            if np.isfinite(auc_value):
                auc_text = f"{auc_value:.4f}"
            else:
                auc_text = "N/A"

            md_file.write(
                f"| {row['model']} "
                f"| {row['accuracy']:.4f} "
                f"| {row['precision']:.4f} "
                f"| {row['recall_sensitivity']:.4f} "
                f"| {row['specificity']:.4f} "
                f"| {row['f1_score']:.4f} "
                f"| {auc_text} "
                f"| {row['false_alarms_per_hour']:.4f} "
                f"| {row['tp']} "
                f"| {row['tn']} "
                f"| {row['fp']} "
                f"| {row['fn']} |\n"
            )

    return csv_path, md_path