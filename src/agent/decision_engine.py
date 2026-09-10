"""
Deterministic Decision Engine for Auto-Handling vs Human Escalation.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from src.config import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    HIGH_RISK_KEYWORDS,
    SENSITIVE_ACCOUNT_KEYWORDS,
    FRUSTRATION_KEYWORDS,
)


@dataclass
class DecisionOutcome:
    decision: str  # "auto_handle" | "escalate"
    reason: str
    risk_flags: List[str]
    intent_confidence_pass: bool
    evidence_grounding_pass: bool


def detect_risk_flags(text: str) -> List[str]:
    """
    Checks for high risk, sensitive account, or frustration keywords.
    """
    t = text.lower()
    flags = []

    for kw in HIGH_RISK_KEYWORDS:
        if kw in t:
            flags.append(f"high_risk_keyword:{kw}")

    for kw in SENSITIVE_ACCOUNT_KEYWORDS:
        if kw in t:
            flags.append(f"sensitive_account_action:{kw}")

    for kw in FRUSTRATION_KEYWORDS:
        if kw in t:
            flags.append(f"repeat_frustration:{kw}")

    return flags


def evaluate_decision(
    customer_text: str,
    intent: str,
    confidence: float,
    evidence: List[Dict[str, Any]],
    confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD
) -> DecisionOutcome:
    """
    Deterministic decision pipeline to choose between AUTO_HANDLE and ESCALATE.
    Produces clear, auditable human reasons for every decision.
    """
    risk_flags = detect_risk_flags(customer_text)
    
    # 1. Check for Invalid Intent
    if intent == "INVALID":
        return DecisionOutcome(
            decision="escalate",
            reason="Input classified as invalid, incomplete context, or non-actionable conversational noise.",
            risk_flags=risk_flags,
            intent_confidence_pass=False,
            evidence_grounding_pass=False
        )

    # 2. Check for High-Risk / Security / Safety Keywords
    high_risk_triggers = [f for f in risk_flags if f.startswith("high_risk_keyword:")]
    if high_risk_triggers:
        return DecisionOutcome(
            decision="escalate",
            reason=f"High-risk safety, legal, or fraud concern detected ({', '.join(high_risk_triggers)}) requiring human specialist verification.",
            risk_flags=risk_flags,
            intent_confidence_pass=True,
            evidence_grounding_pass=True
        )

    # 3. Check for Sensitive Account Credentials
    sensitive_triggers = [f for f in risk_flags if f.startswith("sensitive_account_action:")]
    if sensitive_triggers:
        return DecisionOutcome(
            decision="escalate",
            reason=f"Request involves sensitive account credential management or authentication ({', '.join(sensitive_triggers)}).",
            risk_flags=risk_flags,
            intent_confidence_pass=True,
            evidence_grounding_pass=True
        )

    # 4. Check for Repeated Dissatisfaction / Escalation Request
    frustration_triggers = [f for f in risk_flags if f.startswith("repeat_frustration:")]
    if frustration_triggers:
        return DecisionOutcome(
            decision="escalate",
            reason=f"Customer expresses repeated dissatisfaction or requested human assistance ({', '.join(frustration_triggers)}).",
            risk_flags=risk_flags,
            intent_confidence_pass=True,
            evidence_grounding_pass=True
        )

    # 5. Check Intent Classifier Confidence
    intent_pass = confidence >= confidence_threshold
    if not intent_pass:
        return DecisionOutcome(
            decision="escalate",
            reason=f"Intent classification confidence ({confidence:.2f}) is below safe threshold ({confidence_threshold:.2f}) for intent '{intent}'.",
            risk_flags=risk_flags,
            intent_confidence_pass=False,
            evidence_grounding_pass=False
        )

    # 6. Check Historical Evidence Grounding
    max_sim = max([e.get("similarity_score", 0.0) for e in evidence], default=0.0)
    evidence_pass = max_sim >= similarity_threshold
    if not evidence_pass:
        return DecisionOutcome(
            decision="escalate",
            reason=f"Insufficient historical support precedent (top similarity score {max_sim:.2f} < threshold {similarity_threshold:.2f}).",
            risk_flags=risk_flags,
            intent_confidence_pass=True,
            evidence_grounding_pass=False
        )

    # 7. Safe to Auto-Handle
    return DecisionOutcome(
        decision="auto_handle",
        reason=f"Routine {intent} issue with high classifier confidence ({confidence:.2f}) and verified historical support evidence (top similarity {max_sim:.2f}).",
        risk_flags=risk_flags,
        intent_confidence_pass=True,
        evidence_grounding_pass=True
    )
