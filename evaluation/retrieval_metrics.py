"""
Historical Retrieval Evaluation Metrics:
Recall@1, Recall@3, Recall@5, Mean Reciprocal Rank (MRR), and Topical Alignment.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.retrieval.tfidf_retriever import HistoricalSupportRetriever
from src.classification.agent_classifier import classify_intent


def evaluate_retriever(
    retriever: HistoricalSupportRetriever,
    golden_df: pd.DataFrame,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluates historical support retrieval against the golden evaluation dataset.
    Measures Recall@1, Recall@3, Recall@5, MRR, and Intent Consistency.
    """
    # Filter valid evaluation rows
    valid_df = golden_df[golden_df["valid"] == True].reset_index(drop=True)
    
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    reciprocal_ranks = []
    similarities_top1 = []

    for _, row in valid_df.iterrows():
        query = str(row.get("customer_text", ""))
        true_intent = str(row.get("intent", ""))

        results = retriever.retrieve(query, top_k=top_k)
        if not results:
            recall_at_1.append(0)
            recall_at_3.append(0)
            recall_at_5.append(0)
            reciprocal_ranks.append(0)
            continue

        similarities_top1.append(results[0]["similarity_score"])

        # Check intent match of retrieved customer queries
        match_ranks = []
        for rank, res in enumerate(results, start=1):
            retrieved_intent, _ = classify_intent(res["customer_message"])
            if retrieved_intent == true_intent:
                match_ranks.append(rank)

        # Recall@K
        r1 = 1 if 1 in match_ranks else 0
        r3 = 1 if any(r <= 3 for r in match_ranks) else 0
        r5 = 1 if any(r <= 5 for r in match_ranks) else 0

        recall_at_1.append(r1)
        recall_at_3.append(r3)
        recall_at_5.append(r5)

        if match_ranks:
            reciprocal_ranks.append(1.0 / min(match_ranks))
        else:
            reciprocal_ranks.append(0.0)

    results_dict = {
        "n_evaluated": len(valid_df),
        "recall_at_1": round(float(np.mean(recall_at_1)), 4),
        "recall_at_3": round(float(np.mean(recall_at_3)), 4),
        "recall_at_5": round(float(np.mean(recall_at_5)), 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "mean_top1_similarity": round(float(np.mean(similarities_top1)), 4) if similarities_top1 else 0.0
    }

    return results_dict


def format_retrieval_table(metrics: Dict[str, Any]) -> str:
    """Formats retrieval metrics into a markdown table."""
    headers = ["Metric", "Score"]
    rows = [
        ["Recall@1", f"{metrics['recall_at_1']:.4f}"],
        ["Recall@3", f"{metrics['recall_at_3']:.4f}"],
        ["Recall@5", f"{metrics['recall_at_5']:.4f}"],
        ["Mean Reciprocal Rank (MRR)", f"{metrics['mrr']:.4f}"],
        ["Mean Top-1 Cosine Similarity", f"{metrics['mean_top1_similarity']:.4f}"]
    ]
    return tabulate(rows, headers=headers, tablefmt="github")
