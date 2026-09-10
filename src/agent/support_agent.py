"""
End-to-End AI Customer Support Agent Orchestrator.
"""
import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.classification.agent_classifier import classify_intent
from src.retrieval.tfidf_retriever import get_retriever
from src.agent.decision_engine import evaluate_decision
from src.agent.prompt_templates import GROUNDED_SYSTEM_PROMPT, build_response_prompt
from src.config import DEFAULT_ANTHROPIC_MODEL, RETRIEVAL_TOP_K


class AgentResponse(BaseModel):
    intent: str
    confidence: float
    decision: str
    reason: str
    draft_reply: str
    evidence: list


class AIAgent:
    """
    Production-ready AI Customer Support Agent for AmazonHelp.
    """
    def __init__(self, model_name: str = DEFAULT_ANTHROPIC_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.retriever = get_retriever()

    def process(self, customer_text: str, top_k: int = RETRIEVAL_TOP_K) -> Dict[str, Any]:
        """
        Executes the end-to-end customer support pipeline:
        1. Classify Intent & Calibrated Confidence
        2. Retrieve Historical Customer Inquiries & Amazon Responses
        3. Deterministic Decision Engine (Auto-handle vs Escalate + Reason)
        4. Generate Grounded Draft Response (via Anthropic API or Grounded Template)
        5. Return Clean Structured JSON
        """
        # Step 1: Classification
        intent, confidence = classify_intent(customer_text)

        # Step 2: Retrieval
        evidence = self.retriever.retrieve(customer_text, top_k=top_k)

        # Step 3: Decision Engine
        outcome = evaluate_decision(
            customer_text=customer_text,
            intent=intent,
            confidence=confidence,
            evidence=evidence
        )

        # Step 4: Response Generation
        draft_reply = self._generate_response(
            customer_text=customer_text,
            intent=intent,
            decision=outcome.decision,
            reason=outcome.reason,
            evidence=evidence
        )

        # Format Evidence output
        clean_evidence = [
            {
                "customer_message": e.get("customer_message", ""),
                "historical_response": e.get("historical_response", ""),
                "similarity_score": e.get("similarity_score", 0.0)
            }
            for e in evidence
        ]

        response_obj = AgentResponse(
            intent=intent,
            confidence=confidence,
            decision=outcome.decision,
            reason=outcome.reason,
            draft_reply=draft_reply,
            evidence=clean_evidence
        )

        return response_obj.model_dump()

    def _generate_response(
        self,
        customer_text: str,
        intent: str,
        decision: str,
        reason: str,
        evidence: list
    ) -> str:
        """
        Generates grounded response using Anthropic API when available,
        or deterministic grounded template fallback when offline.
        """
        if intent == "INVALID":
            return "Thank you for contacting Amazon Help. Please let us know how we can assist you with your order or account."

        if decision == "escalate":
            # Escalation response directs customer to human/secure support
            return (
                "I want to make sure this gets resolved for you as quickly as possible. "
                "Because your request requires specific review, please reach out to our specialist team directly: https://amazon.com/help"
            )

        # Auto-handle with Anthropic API if key is available
        if self.api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)
                prompt = build_response_prompt(customer_text, intent, evidence)
                
                message = client.messages.create(
                    model=self.model_name,
                    max_tokens=150,
                    system=GROUNDED_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                if message.content:
                    return message.content[0].text.strip()
            except Exception as e:
                print(f"[Warning] Anthropic API call failed: {e}. Using grounded template.")

        # Grounded fallback draft derived from top retrieved evidence
        if evidence:
            top_resp = evidence[0].get("historical_response", "")
            if top_resp:
                # Clean @handles from historical response to make fresh draft
                import re
                cleaned_draft = re.sub(r"@[A-Za-z0-9_]+\s*", "", top_resp).strip()
                return cleaned_draft

        return (
            "We're sorry for the inconvenience. Please check the latest status in Your Orders "
            "or contact our support team at https://amazon.com/help so we can assist."
        )


_agent_instance: Optional[AIAgent] = None


def handle_customer_message(text: str) -> Dict[str, Any]:
    """Convenience functional interface for processing messages."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AIAgent()
    return _agent_instance.process(text)
