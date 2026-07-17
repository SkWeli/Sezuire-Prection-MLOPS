"""Reporting and MLflow utilities for EEG experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import mlflow
import numpy as np


DEFAULT_CLASS_NAMES = ["Interictal", "Pre-Ictal", "Ictal"]


def log_evaluation_metrics_to_mlflow(prefix, metrics, step=None):
    """Log scalar evaluation metrics using a consistent MLflow prefix."""
    metric_keys = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "decision_threshold",
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
        if np.isfinite(value):
            mlflow_metrics[f"{prefix}_{key}"] = value

    if not mlflow_metrics:
        return

    if step is None:
        mlflow.log_metrics(mlflow_metrics)
    else:
        mlflow.log_metrics(mlflow_metrics, step=step)


def _write_full_confusion_matrix(file, confusion, class_names):
    file.write("\nThree-Class Confusion Matrix\n")
    file.write("----------------------------\n")
    header = "Actual \\ Predicted".ljust(24) + "".join(
        name.rjust(14) for name in class_names
    )
    file.write(header + "\n")

    for class_name, row in zip(class_names, confusion):
        file.write(class_name.ljust(24) + "".join(str(int(value)).rjust(14) for value in row) + "\n")


def save_test_metrics_report(
    metrics,
    output_dir,
    run_name,
    baseline_metrics=None,
    class_names=None,
):
    """Save a thesis-ready final test report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_name}_test_metrics.txt"

    class_names = class_names or DEFAULT_CLASS_NAMES
    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    alarm_confusion = np.asarray(metrics["alarm_confusion_matrix"], dtype=int)

    with report_path.open("w", encoding="utf-8") as file:
        file.write("Final Test Evaluation Report\n")
        file.write("============================\n\n")
        file.write(f"Test loss                    : {metrics['loss']:.4f}\n")
        file.write(f"Test accuracy                : {metrics['accuracy']:.4f}\n")
        file.write(f"Balanced accuracy            : {metrics['balanced_accuracy']:.4f}\n")
        file.write(f"Decision threshold           : {metrics['decision_threshold']:.4f}\n")
        file.write(f"Macro precision              : {metrics['precision']:.4f}\n")
        file.write(f"Macro recall / sensitivity   : {metrics['recall_sensitivity']:.4f}\n")
        file.write(f"Macro specificity            : {metrics['specificity']:.4f}\n")
        file.write(f"Macro F1-score               : {metrics['f1']:.4f}\n")

        if np.isfinite(metrics["auc"]):
            file.write(f"Macro multiclass AUC         : {metrics['auc']:.4f}\n")
        else:
            file.write("Macro multiclass AUC         : Not available\n")

        file.write(f"Evaluated duration hours     : {metrics['evaluated_duration_hours']:.4f}\n")
        file.write(f"False-positive windows/hour  : {metrics['false_alarms_per_hour']:.4f}\n")

        _write_full_confusion_matrix(file, confusion, class_names)

        file.write("\nAlarm-Level Binary Aggregation\n")
        file.write("------------------------------\n")
        file.write("Alarm = Pre-Ictal or Ictal\n")
        file.write("                         Predicted No Alarm  Predicted Alarm\n")
        file.write(
            f"Actual No Alarm          {alarm_confusion[0, 0]:18d}  {alarm_confusion[0, 1]:15d}\n"
        )
        file.write(
            f"Actual Alarm             {alarm_confusion[1, 0]:18d}  {alarm_confusion[1, 1]:15d}\n"
        )

        if baseline_metrics is not None:
            baseline_class = int(baseline_metrics["majority_class"])
            baseline_name = (
                class_names[baseline_class]
                if baseline_class < len(class_names)
                else str(baseline_class)
            )

            file.write("\nMajority-Class Baseline\n")
            file.write("-----------------------\n")
            file.write(f"Majority class                : {baseline_class} ({baseline_name})\n")
            file.write(f"Baseline accuracy             : {baseline_metrics['accuracy']:.4f}\n")
            file.write(f"Baseline macro precision      : {baseline_metrics['precision']:.4f}\n")
            file.write(f"Baseline macro recall         : {baseline_metrics['recall_sensitivity']:.4f}\n")
            file.write(f"Baseline macro specificity    : {baseline_metrics['specificity']:.4f}\n")
            file.write(f"Baseline macro F1-score       : {baseline_metrics['f1']:.4f}\n")

            if np.isfinite(baseline_metrics["auc"]):
                file.write(f"Baseline macro AUC            : {baseline_metrics['auc']:.4f}\n")
            else:
                file.write("Baseline macro AUC            : Not available\n")

            file.write(
                f"Baseline false-positive windows/hour: "
                f"{baseline_metrics['false_alarms_per_hour']:.4f}\n"
            )

            file.write("\nModel vs Baseline\n")
            file.write("-----------------\n")
            file.write(
                f"Accuracy difference           : "
                f"{metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}\n"
            )
            file.write(
                f"Macro F1 difference           : "
                f"{metrics['f1'] - baseline_metrics['f1']:+.4f}\n"
            )
            file.write(
                f"False-positive windows/hour difference: "
                f"{metrics['false_alarms_per_hour'] - baseline_metrics['false_alarms_per_hour']:+.4f}\n"
            )

        file.write("\nReported accuracy source: held-out test set\n")
        file.write("Threshold source: validation split only\n")
        file.write("False alarm type: window-level false positives per hour\n")

    return report_path


def save_confusion_matrix_plot(metrics, output_dir, run_name, class_names=None):
    """Save the full multiclass confusion matrix as PNG and CSV."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names or DEFAULT_CLASS_NAMES

    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    png_path = output_dir / f"{run_name}_confusion_matrix.png"
    csv_path = output_dir / f"{run_name}_confusion_matrix.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Actual / Predicted", *class_names])
        for class_name, row in zip(class_names, confusion):
            writer.writerow([class_name, *row.tolist()])

    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=class_names,
    )
    display.plot(ax=axis, values_format="d")
    axis.set_title(f"{run_name} — Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return png_path, csv_path


def save_cnn_baseline_results_table(
    test_metrics,
    baseline_metrics,
    output_dir,
    run_name,
    model_name="SeizureCNN",
):
    """Save model-versus-majority-baseline results as CSV and Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{run_name}_model_baseline_results.csv"
    md_path = output_dir / f"{run_name}_model_baseline_results.md"

    fieldnames = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "decision_threshold",
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

    def _row(name, values):
        return {
            "model": name,
            "accuracy": values["accuracy"],
            "balanced_accuracy": values["balanced_accuracy"],
            "decision_threshold": values.get("decision_threshold", 0.5),
            "precision": values["precision"],
            "recall_sensitivity": values["recall_sensitivity"],
            "specificity": values["specificity"],
            "f1_score": values["f1"],
            "auc": values["auc"],
            "false_alarms_per_hour": values["false_alarms_per_hour"],
            "tp": values["tp"],
            "tn": values["tn"],
            "fp": values["fp"],
            "fn": values["fn"],
        }

    rows = [
        _row(model_name, test_metrics),
        _row("MajorityClassBaseline", baseline_metrics),
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as md_file:
        md_file.write("# Model vs Majority-Class Baseline\n\n")
        md_file.write(
            "| Model | Accuracy | Balanced Accuracy | Threshold | Precision | Recall/Sensitivity | "
            "Specificity | F1-score | AUC | False-Positive Windows/Hour | TP | TN | FP | FN |\n"
        )
        md_file.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for row in rows:
            auc_text = f"{row['auc']:.4f}" if np.isfinite(row["auc"]) else "N/A"
            md_file.write(
                f"| {row['model']} "
                f"| {row['accuracy']:.4f} "
                f"| {row['balanced_accuracy']:.4f} "
                f"| {row['decision_threshold']:.4f} "
                f"| {row['precision']:.4f} "
                f"| {row['recall_sensitivity']:.4f} "
                f"| {row['specificity']:.4f} "
                f"| {row['f1_score']:.4f} "
                f"| {auc_text} "
                f"| {row['false_alarms_per_hour']:.4f} "
                f"| {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} |\n"
            )

    return csv_path, md_path
