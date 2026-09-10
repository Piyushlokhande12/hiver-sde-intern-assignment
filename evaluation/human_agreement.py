"""
Human Agreement and Inter-Annotator Reliability Evaluation.
Computes percentage agreement and Cohen's Kappa score.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import HUMAN_EVAL_PATH


def cohens_kappa(rater1: List[str], rater2: List[str]) -> float:
    """
    Computes Cohen's Kappa coefficient between two raters.
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    categories = list(set(rater1 + rater2))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n = len(rater1)
    k = len(categories)

    # Confusion matrix between raters
    cm = np.zeros((k, k), dtype=int)
    for r1, r2 in zip(rater1, rater2):
        cm[cat_to_idx[r1]][cat_to_idx[r2]] += 1

    # Observed agreement
    p_o = np.trace(cm) / n

    # Expected agreement
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    p_e = np.sum((row_sums * col_sums) / (n * n))

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


def evaluate_human_agreement(file_path: Path = HUMAN_EVAL_PATH) -> Dict[str, Any]:
    """
    Loads human agreement evaluation dataset and computes inter-annotator statistics.
    """
    if not file_path.exists():
        return {
            "error": f"Human agreement file not found at {file_path}",
            "raw_agreement": 0.0,
            "cohens_kappa": 0.0
        }

    df = pd.read_csv(file_path)
    r1 = df["annotator1_intent"].astype(str).tolist()
    r2 = df["annotator2_intent"].astype(str).tolist()

    raw_agree = float(np.mean([a == b for a, b in zip(r1, r2)]))
    kappa = cohens_kappa(r1, r2)

    return {
        "n_samples": len(df),
        "raw_agreement_percentage": round(raw_agree * 100, 2),
        "cohens_kappa": round(kappa, 4),
        "interpretation": _interpret_kappa(kappa)
    }


def _interpret_kappa(kappa: float) -> str:
    if kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"
