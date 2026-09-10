"""
Comprehensive Intent Classification Evaluation Metrics.
Calculates Accuracy, Macro F1, Per-intent Precision/Recall/F1, and Confusion Matrix.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.config import ALL_INTENTS


def evaluate_intent_classifier(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes complete evaluation metrics for intent classification.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
            classification_report
        )
    except ImportError:
        raise ImportError("scikit-learn is required for metric computation. Run `pip install scikit-learn`.")

    unique_labels = labels or sorted(list(set(y_true + y_pred)))
    
    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro metrics (unweighted mean - critical for class imbalance)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, average="macro", zero_division=0
    )
    
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, label in enumerate(unique_labels):
        per_class_metrics[label] = {
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1_score": round(float(f1_class[i]), 4),
            "support": int(support_class[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_dict = {
        "labels": unique_labels,
        "matrix": cm.tolist()
    }

    report_str = classification_report(y_true, y_pred, labels=unique_labels, zero_division=0)

    # Majority baseline comparison
    majority_label = max(set(y_true), key=y_true.count)
    majority_preds = [majority_label] * len(y_true)
    majority_acc = float(accuracy_score(y_true, majority_preds))
    _, _, majority_f1_macro, _ = precision_recall_fscore_support(
        y_true, majority_preds, labels=unique_labels, average="macro", zero_division=0
    )

    results = {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm_dict,
        "classification_report_str": report_str,
        "majority_baseline": {
            "majority_class": majority_label,
            "accuracy": round(majority_acc, 4),
            "macro_f1": round(float(majority_f1_macro), 4)
        }
    }

    return results


def format_intent_metrics_table(metrics: Dict[str, Any]) -> str:
    """Formats metrics dictionary into a clean markdown table."""
    headers = ["Intent", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for intent, scores in metrics["per_class"].items():
        rows.append([
            intent,
            f"{scores['precision']:.3f}",
            f"{scores['recall']:.3f}",
            f"{scores['f1_score']:.3f}",
            scores["support"]
        ])
    table = tabulate(rows, headers=headers, tablefmt="github")
    
    summary = (
        f"**Accuracy:** {metrics['accuracy']:.4f}\n"
        f"**Macro F1:** {metrics['macro_f1']:.4f}\n"
        f"**Weighted F1:** {metrics['weighted_f1']:.4f}\n\n"
        f"**Majority Class Baseline Accuracy:** {metrics['majority_baseline']['accuracy']:.4f} "
        f"(Macro F1: {metrics['majority_baseline']['macro_f1']:.4f})\n\n"
        f"### Per-Class Performance:\n\n{table}"
    )
    return summary
