"""
Unit tests for deterministic decision engine.
"""
from src.agent.decision_engine import evaluate_decision


def test_invalid_intent_escalation():
    outcome = evaluate_decision(
        customer_text="thanks a lot",
        intent="INVALID",
        confidence=0.95,
        evidence=[]
    )
    assert outcome.decision == "escalate"


def test_high_risk_keyword_escalation():
    outcome = evaluate_decision(
        customer_text="I want to report an unauthorized stolen card fraud on my account",
        intent="payment_issue",
        confidence=0.92,
        evidence=[{"customer_message": "foo", "historical_response": "bar", "similarity_score": 0.85}]
    )
    assert outcome.decision == "escalate"
    assert any("high_risk_keyword" in f for f in outcome.risk_flags)


def test_sensitive_account_action_escalation():
    outcome = evaluate_decision(
        customer_text="Please assist with my password reset and 2fa security key",
        intent="account_issue",
        confidence=0.90,
        evidence=[{"customer_message": "foo", "historical_response": "bar", "similarity_score": 0.80}]
    )
    assert outcome.decision == "escalate"


def test_low_confidence_escalation():
    outcome = evaluate_decision(
        customer_text="Something weird happened with my box",
        intent="other",
        confidence=0.45,
        evidence=[{"customer_message": "foo", "historical_response": "bar", "similarity_score": 0.50}]
    )
    assert outcome.decision == "escalate"


def test_insufficient_evidence_escalation():
    outcome = evaluate_decision(
        customer_text="My order is delayed",
        intent="order_delivery",
        confidence=0.90,
        evidence=[{"customer_message": "foo", "historical_response": "bar", "similarity_score": 0.15}]
    )
    assert outcome.decision == "escalate"


def test_auto_handle_success():
    outcome = evaluate_decision(
        customer_text="Where can I track my late package?",
        intent="order_delivery",
        confidence=0.88,
        evidence=[{"customer_message": "How do I track my delivery?", "historical_response": "Check Your Orders page.", "similarity_score": 0.75}]
    )
    assert outcome.decision == "auto_handle"
    assert "Routine order_delivery issue" in outcome.reason
