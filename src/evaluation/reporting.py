"""Reporting, tables, plots, and MLflow utilities for EEG experiments."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import mlflow
import numpy as np


DEFAULT_CLASS_NAMES = ["Interictal", "Pre-Ictal", "Ictal"]


def _metric_safe_name(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def log_evaluation_metrics_to_mlflow(prefix, metrics, step=None, class_names=None):
    """Log scalar primary, alarm-level, and per-class metrics to MLflow."""
    scalar_keys = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "decision_threshold",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc",
        "alarm_accuracy",
        "alarm_balanced_accuracy",
        "alarm_precision",
        "alarm_recall_sensitivity",
        "alarm_specificity",
        "alarm_f1",
        "alarm_youden_j",
        "alarm_prediction_rate",
        "alarm_auc",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluated_duration_seconds",
        "evaluated_duration_hours",
        "false_alarms_per_hour",
    ]

    mlflow_metrics = {}
    for key in scalar_keys:
        if key not in metrics:
            continue
        value = float(metrics[key])
        if np.isfinite(value):
            mlflow_metrics[f"{prefix}_{key}"] = value

    class_names = class_names or DEFAULT_CLASS_NAMES
    per_class_fields = {
        "precision": metrics.get("per_class_precision"),
        "recall": metrics.get("per_class_recall"),
        "f1": metrics.get("per_class_f1"),
        "specificity": metrics.get("per_class_specificity"),
    }
    for metric_name, values in per_class_fields.items():
        if values is None:
            continue
        for class_name, value in zip(class_names, values):
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                mlflow_metrics[
                    f"{prefix}_{_metric_safe_name(class_name)}_{metric_name}"
                ] = numeric_value

    if not mlflow_metrics:
        return
    if step is None:
        mlflow.log_metrics(mlflow_metrics)
    else:
        mlflow.log_metrics(mlflow_metrics, step=step)


def _write_full_confusion_matrix(file, confusion, class_names):
    file.write("\nThree-Class Confusion Matrix\n")
    file.write("----------------------------\n")
    file.write(
        "Actual \\ Predicted".ljust(24)
        + "".join(name.rjust(14) for name in class_names)
        + "\n"
    )
    for class_name, row in zip(class_names, confusion):
        file.write(
            class_name.ljust(24)
            + "".join(str(int(value)).rjust(14) for value in row)
            + "\n"
        )


def _write_per_class_table(file, metrics, class_names):
    file.write("\nPer-Class Metrics\n")
    file.write("-----------------\n")
    file.write(
        "Class".ljust(18)
        + "Precision".rjust(12)
        + "Recall".rjust(12)
        + "Specificity".rjust(14)
        + "F1".rjust(12)
        + "\n"
    )
    for index, class_name in enumerate(class_names):
        file.write(
            class_name.ljust(18)
            + f"{metrics['per_class_precision'][index]:.4f}".rjust(12)
            + f"{metrics['per_class_recall'][index]:.4f}".rjust(12)
            + f"{metrics['per_class_specificity'][index]:.4f}".rjust(14)
            + f"{metrics['per_class_f1'][index]:.4f}".rjust(12)
            + "\n"
        )


def save_test_metrics_report(
    metrics,
    output_dir,
    run_name,
    baseline_metrics=None,
    class_names=None,
):
    """Save a thesis-ready report separating multiclass and alarm evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_name}_test_metrics.txt"

    class_names = class_names or DEFAULT_CLASS_NAMES
    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    alarm_confusion = np.asarray(metrics["alarm_confusion_matrix"], dtype=int)

    with report_path.open("w", encoding="utf-8") as file:
        file.write("Final Held-Out Test Evaluation Report\n")
        file.write("=====================================\n\n")
        file.write("Primary Task: Three-Class Classification\n")
        file.write("----------------------------------------\n")
        file.write(f"Test loss                    : {metrics['loss']:.4f}\n")
        file.write(f"Accuracy                     : {metrics['accuracy']:.4f}\n")
        file.write(f"Balanced accuracy            : {metrics['balanced_accuracy']:.4f}\n")
        file.write(f"Macro precision              : {metrics['precision']:.4f}\n")
        file.write(f"Macro recall / sensitivity   : {metrics['recall_sensitivity']:.4f}\n")
        file.write(f"Macro specificity            : {metrics['specificity']:.4f}\n")
        file.write(f"Macro F1-score               : {metrics['f1']:.4f}\n")
        file.write(
            "Macro multiclass AUC         : "
            + (f"{metrics['auc']:.4f}" if np.isfinite(metrics["auc"]) else "Not available")
            + "\n"
        )

        _write_per_class_table(file, metrics, class_names)
        _write_full_confusion_matrix(file, confusion, class_names)

        file.write("\nSecondary Task: Alarm / No-Alarm\n")
        file.write("--------------------------------\n")
        file.write("Alarm = Pre-Ictal or Ictal\n")
        file.write(f"Validation-selected threshold : {metrics['decision_threshold']:.4f}\n")
        if metrics.get("alarm_threshold_policy"):
            file.write(f"Threshold-selection policy    : {metrics['alarm_threshold_policy']}\n")
        if metrics.get("alarm_threshold_policy") == "specificity_constrained":
            file.write(
                f"Minimum validation specificity: "
                f"{metrics.get('alarm_min_specificity', float('nan')):.4f}\n"
            )
        file.write(f"Alarm accuracy                : {metrics['alarm_accuracy']:.4f}\n")
        file.write(f"Alarm balanced accuracy       : {metrics['alarm_balanced_accuracy']:.4f}\n")
        file.write(f"Alarm precision               : {metrics['alarm_precision']:.4f}\n")
        file.write(f"Alarm sensitivity             : {metrics['alarm_recall_sensitivity']:.4f}\n")
        file.write(f"Alarm specificity             : {metrics['alarm_specificity']:.4f}\n")
        file.write(f"Alarm F1-score                : {metrics['alarm_f1']:.4f}\n")
        file.write(f"Alarm Youden J                : {metrics['alarm_youden_j']:.4f}\n")
        file.write(f"Predicted-alarm window rate   : {metrics['alarm_prediction_rate']:.4f}\n")
        file.write(
            "Alarm AUC                     : "
            + (f"{metrics['alarm_auc']:.4f}" if np.isfinite(metrics["alarm_auc"]) else "Not available")
            + "\n"
        )
        file.write(f"Evaluated duration hours      : {metrics['evaluated_duration_hours']:.4f}\n")
        file.write(f"False-positive windows/hour   : {metrics['false_alarms_per_hour']:.4f}\n")
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
            file.write(f"Baseline balanced accuracy    : {baseline_metrics['balanced_accuracy']:.4f}\n")
            file.write(f"Baseline macro precision      : {baseline_metrics['precision']:.4f}\n")
            file.write(f"Baseline macro recall         : {baseline_metrics['recall_sensitivity']:.4f}\n")
            file.write(f"Baseline macro specificity    : {baseline_metrics['specificity']:.4f}\n")
            file.write(f"Baseline macro F1-score       : {baseline_metrics['f1']:.4f}\n")
            file.write(
                "Baseline macro AUC            : "
                + (
                    f"{baseline_metrics['auc']:.4f}"
                    if np.isfinite(baseline_metrics["auc"])
                    else "Not available"
                )
                + "\n"
            )
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
                f"Balanced accuracy difference  : "
                f"{metrics['balanced_accuracy'] - baseline_metrics['balanced_accuracy']:+.4f}\n"
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
        file.write("Primary multiclass prediction rule: argmax, independent of alarm threshold\n")
        file.write("Alarm threshold source: validation split only\n")
        file.write("False alarm type: window-level false positives per hour\n")

    return report_path


def save_confusion_matrix_plot(metrics, output_dir, run_name, class_names=None):
    """Save the primary multiclass confusion matrix as PNG and CSV."""
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
    axis.set_title(f"{run_name} — Held-Out Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return png_path, csv_path


def save_threshold_sweep_table(sweep_rows, output_dir, run_name):
    """Save the validation alarm-threshold trade-off as CSV and Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_threshold_sweep.csv"
    md_path = output_dir / f"{run_name}_threshold_sweep.md"

    if not sweep_rows:
        raise ValueError("Threshold sweep contains no rows.")

    fieldnames = list(sweep_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)

    selected_fields = [
        "threshold",
        "alarm_f1",
        "alarm_sensitivity",
        "alarm_specificity",
        "alarm_balanced_accuracy",
        "alarm_youden_j",
        "alarm_prediction_rate",
        "false_alarms_per_hour",
    ]
    with md_path.open("w", encoding="utf-8") as md_file:
        md_file.write("# Validation Alarm Threshold Sweep\n\n")
        md_file.write(
            "| Threshold | Alarm F1 | Sensitivity | Specificity | Balanced Accuracy "
            "| Youden J | Predicted Alarm Rate | False Alarms/Hour |\n"
        )
        md_file.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sweep_rows:
            md_file.write(
                f"| {row[selected_fields[0]]:.2f} "
                f"| {row[selected_fields[1]]:.4f} "
                f"| {row[selected_fields[2]]:.4f} "
                f"| {row[selected_fields[3]]:.4f} "
                f"| {row[selected_fields[4]]:.4f} "
                f"| {row[selected_fields[5]]:.4f} "
                f"| {row[selected_fields[6]]:.4f} "
                f"| {row[selected_fields[7]]:.4f} |\n"
            )

    return csv_path, md_path


def save_split_manifest(split_patient_ids, output_dir, run_name):
    """Save exact patient assignments so the experiment can be reproduced."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_patient_split.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "patient_id"])
        writer.writeheader()
        for split_name in ("train", "validation", "test"):
            for patient_id in split_patient_ids.get(split_name, []):
                writer.writerow({"split": split_name, "patient_id": patient_id})

    return csv_path


def save_cnn_baseline_results_table(
    test_metrics,
    baseline_metrics,
    output_dir,
    run_name,
    model_name="SeizureCNN",
):
    """Save a concise model-versus-baseline summary as CSV and Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_model_baseline_results.csv"
    md_path = output_dir / f"{run_name}_model_baseline_results.md"

    fieldnames = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "auc",
        "decision_threshold",
        "alarm_accuracy",
        "alarm_balanced_accuracy",
        "alarm_precision",
        "alarm_sensitivity",
        "alarm_specificity",
        "alarm_f1",
        "alarm_youden_j",
        "alarm_prediction_rate",
        "alarm_auc",
        "false_alarms_per_hour",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    def row(name, values):
        return {
            "model": name,
            "accuracy": values["accuracy"],
            "balanced_accuracy": values["balanced_accuracy"],
            "precision": values["precision"],
            "recall_sensitivity": values["recall_sensitivity"],
            "specificity": values["specificity"],
            "f1_score": values["f1"],
            "auc": values["auc"],
            "decision_threshold": values.get("decision_threshold", 0.5),
            "alarm_accuracy": values["alarm_accuracy"],
            "alarm_balanced_accuracy": values["alarm_balanced_accuracy"],
            "alarm_precision": values["alarm_precision"],
            "alarm_sensitivity": values["alarm_recall_sensitivity"],
            "alarm_specificity": values["alarm_specificity"],
            "alarm_f1": values["alarm_f1"],
            "alarm_youden_j": values["alarm_youden_j"],
            "alarm_prediction_rate": values["alarm_prediction_rate"],
            "alarm_auc": values["alarm_auc"],
            "false_alarms_per_hour": values["false_alarms_per_hour"],
            "tp": values["tp"],
            "tn": values["tn"],
            "fp": values["fp"],
            "fn": values["fn"],
        }

    rows = [row(model_name, test_metrics), row("MajorityClassBaseline", baseline_metrics)]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as md_file:
        md_file.write("# Model vs Majority-Class Baseline\n\n")
        md_file.write(
            "| Model | Accuracy | Balanced Accuracy | Macro F1 | Macro AUC | Alarm F1 "
            "| Alarm Sensitivity | Alarm Specificity | Predicted Alarm Rate | False Alarms/Hour |\n"
        )
        md_file.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for values in rows:
            macro_auc = f"{values['auc']:.4f}" if np.isfinite(values["auc"]) else "N/A"
            md_file.write(
                f"| {values['model']} | {values['accuracy']:.4f} "
                f"| {values['balanced_accuracy']:.4f} | {values['f1_score']:.4f} "
                f"| {macro_auc} | {values['alarm_f1']:.4f} "
                f"| {values['alarm_sensitivity']:.4f} | {values['alarm_specificity']:.4f} "
                f"| {values['alarm_prediction_rate']:.4f} "
                f"| {values['false_alarms_per_hour']:.4f} |\n"
            )

    return csv_path, md_path
