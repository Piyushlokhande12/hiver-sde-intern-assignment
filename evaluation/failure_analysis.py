"""
Failure Mode Analysis & Diagnostic Reporting.
Identifies the Top 5 Error Categories, slice statistics, and actionable insights.
"""
from typing import Dict, List, Any
import pandas as pd
from tabulate import tabulate

from src.classification.agent_classifier import classify_intent
from src.retrieval.tfidf_retriever import HistoricalSupportRetriever
from src.agent.support_agent import AIAgent


TOP_5_FAILURE_MODES = [
    {
        "id": 1,
        "name": "Context-Deprived Mid-Thread Tweets",
        "description": "Customer tweets referring to earlier email/phone conversations ('I already replied', 'Did you get my DM?'). Classifier struggles due to missing conversation history.",
        "mitigation": "Integrate Twitter conversation thread reconstruction before classification; route contextless replies to dedicated thread-continuation queue."
    },
    {
        "id": 2,
        "name": "Intent Boundary Overlap (Cancellation vs Refund vs Account)",
        "description": "Cross-intent queries such as 'Cancel Prime membership and refund $14.99' or 'Accidental purchase return'.",
        "mitigation": "Adopt multi-label classification or hierarchical taxonomy (Order -> Delivery/Cancel; Billing -> Refund/Charge)."
    },
    {
        "id": 3,
        "name": "Cross-Lingual & Regional Marketplace Misalignment",
        "description": "Tweets in French, German, Japanese, Spanish mentioning regional domain links (amazon.fr, amazon.de). Lexical TF-IDF has lower recall across non-English vocabulary.",
        "mitigation": "Deploy language detection filter + multilingual embeddings (e.g. multilingual E5 or XLM-RoBERTa)."
    },
    {
        "id": 4,
        "name": "Lexical Keyword Bias in TF-IDF Retrieval",
        "description": "TF-IDF matching frequent words ('package', 'Amazon', 'order') rather than specific failure symptoms ('stuck at customs', 'shattered ceramic').",
        "mitigation": "Upgrade retrieval baseline to hybrid BM25 + dense semantic retrieval."
    },
    {
        "id": 5,
        "name": "False Escalation on High-Frustration Routine Queries",
        "description": "Customers expressing strong irritation ('worst service ever, where is my order') triggering frustration escalation even when self-service tracking link is sufficient.",
        "mitigation": "Decouple sentiment polarity from intent routing; auto-handle routine queries with empathetic apology while escalating only true complex claims."
    }
]


def run_failure_analysis(
    golden_df: pd.DataFrame,
    agent: AIAgent
) -> Dict[str, Any]:
    """
    Analyzes misclassifications and escalation decisions across the golden dataset.
    """
    valid_df = golden_df[golden_df["valid"] == True].reset_index(drop=True)
    
    intent_errors = []
    escalation_breakdown = {"auto_handle": 0, "escalate": 0}

    for _, row in valid_df.iterrows():
        text = str(row.get("customer_text", ""))
        true_intent = str(row.get("intent", ""))
        pred_intent, conf = classify_intent(text)
        
        output = agent.process(text)
        decision = output.get("decision", "escalate")
        escalation_breakdown[decision] = escalation_breakdown.get(decision, 0) + 1

        if pred_intent != true_intent:
            intent_errors.append({
                "customer_text": text,
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "confidence": conf,
                "decision": decision,
                "reason": output.get("reason")
            })

    return {
        "total_evaluated": len(valid_df),
        "total_intent_errors": len(intent_errors),
        "intent_error_rate": round(len(intent_errors) / len(valid_df), 4) if valid_df.shape[0] > 0 else 0,
        "escalation_distribution": escalation_breakdown,
        "top_5_failure_modes": TOP_5_FAILURE_MODES,
        "sample_error_cases": intent_errors[:10]
    }
