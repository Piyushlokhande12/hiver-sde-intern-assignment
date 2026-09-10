"""
Strict Grounded Prompt Templates for Amazon Support Agent.
"""
from typing import List, Dict, Any

GROUNDED_SYSTEM_PROMPT = """You are an AI Customer Support Agent representing @AmazonHelp on Twitter/X.
Your task is to draft a helpful, empathetic, and strictly grounded customer support response.

CRITICAL POLICY & GROUNDING RULES:
1. Grounding: Base your advice ONLY on the historical Amazon support responses provided as evidence. DO NOT invent delivery timelines, refund guarantees, or Amazon policies that are not grounded in the evidence.
2. Twitter Tone & Length: Keep replies concise (under 280 characters if possible, max 2-3 sentences), polite, and professional in AmazonHelp's authentic social media voice.
3. Language: Match the customer's language (e.g. English, French, Spanish, German, Japanese).
4. Escalation: If the customer's issue cannot be safely resolved from the evidence, provide standard Amazon secure support guidance to connect securely via https://amazon.com/help without making ungrounded commitments.
5. No Placeholders: Do not output "[Insert Name Here]" or unfilled bracket placeholders.
"""

def format_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    """Formats retrieved historical examples into structured prompt context."""
    if not evidence:
        return "No historical evidence available."
    
    formatted = []
    for i, e in enumerate(evidence, 1):
        cust = e.get("customer_message", "").strip()
        resp = e.get("historical_response", "").strip()
        sim = e.get("similarity_score", 0.0)
        formatted.append(
            f"--- Historical Case {i} (Similarity: {sim:.2f}) ---\n"
            f"Customer Problem: {cust}\n"
            f"Amazon Resolution: {resp}"
        )
    return "\n\n".join(formatted)


def build_response_prompt(customer_text: str, intent: str, evidence: List[Dict[str, Any]]) -> str:
    """Builds user prompt for grounded response generation."""
    evidence_text = format_evidence_block(evidence)
    return f"""Customer Message:
"{customer_text}"

Classified Intent: {intent}

Historical Grounding Evidence:
{evidence_text}

Draft the appropriate AmazonHelp response for this customer:"""
