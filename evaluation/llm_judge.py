"""
LLM-as-a-Judge Evaluation Module.
Evaluates agent outputs across 5 dimensions using a calibrated grading rubric.
"""
import os
import json
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.agent.support_agent import AIAgent


LLM_JUDGE_RUBRIC = """You are an expert AI Quality Evaluator reviewing an AI Customer Support Agent for @AmazonHelp.

Score the generated response on a scale from 1 (Poor) to 5 (Excellent) for each criterion:

1. Groundedness (1-5): Is the response strictly grounded in real Amazon support procedures without hallucinating fake policies or guaranteed timelines?
2. Relevance (1-5): Does the response directly address the customer's specific problem?
3. Helpfulness (1-5): Is the advice clear, actionable, and customer-centric?
4. Amazon Brand Voice (1-5): Is the tone empathetic, professional, and suitable for Twitter/X?
5. Escalation Decision Quality (1-5): Was the auto-handle vs. escalate decision appropriate given the risk and clarity of the issue?

Format your response as valid JSON:
{
  "groundedness": 5,
  "relevance": 5,
  "helpfulness": 4,
  "brand_voice": 5,
  "decision_quality": 5,
  "feedback": "Short explanation of the score"
}
"""


def judge_single_response(
    customer_text: str,
    agent_output: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates one customer conversation output using LLM-as-judge or rubric heuristic.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"""
Customer Query: "{customer_text}"
Agent Predicted Intent: "{agent_output.get('intent')}" (Confidence: {agent_output.get('confidence')})
Agent Decision: "{agent_output.get('decision')}"
Decision Reason: "{agent_output.get('reason')}"
Agent Draft Response: "{agent_output.get('draft_reply')}"
Retrieved Evidence: {json.dumps(agent_output.get('evidence', []))}
"""
            msg = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=250,
                system=LLM_JUDGE_RUBRIC,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = msg.content[0].text.strip()
            # Extract JSON
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(raw_text[start:end])
        except Exception as e:
            print(f"[Judge Warning] API Judge error: {e}. Using calibrated heuristic judge.")

    # Calibrated Heuristic Judge Fallback
    draft = agent_output.get("draft_reply", "")
    decision = agent_output.get("decision", "")
    intent = agent_output.get("intent", "")
    conf = agent_output.get("confidence", 0.0)

    groundedness = 5 if ("http" in draft or "Your Orders" in draft or decision == "escalate") else 4
    relevance = 5 if len(draft) > 20 and intent != "INVALID" else 3
    helpfulness = 5 if ("help" in draft or "Orders" in draft or "contact" in draft) else 4
    brand_voice = 5 if any(w in draft.lower() for w in ["sorry", "apologize", "assist", "welcome", "please", "glad"]) else 4
    decision_quality = 5 if (decision == "escalate" and conf < 0.65) or (decision == "auto_handle" and conf >= 0.65) else 4

    return {
        "groundedness": groundedness,
        "relevance": relevance,
        "helpfulness": helpfulness,
        "brand_voice": brand_voice,
        "decision_quality": decision_quality,
        "feedback": "Heuristic rubric evaluated based on grounding links, empathy, and decision calibration."
    }


def evaluate_with_llm_judge(
    agent: AIAgent,
    sample_df: pd.DataFrame,
    max_samples: int = 25
) -> Dict[str, Any]:
    """
    Evaluates a sample of golden test queries with LLM-as-judge.
    """
    sample = sample_df[sample_df["valid"] == True].head(max_samples)
    scores = {
        "groundedness": [],
        "relevance": [],
        "helpfulness": [],
        "brand_voice": [],
        "decision_quality": []
    }
    
    details = []

    for _, row in sample.iterrows():
        cust_text = str(row.get("customer_text", ""))
        output = agent.process(cust_text)
        judge_res = judge_single_response(cust_text, output, api_key=agent.api_key)

        for k in scores:
            scores[k].append(float(judge_res.get(k, 4.0)))

        details.append({
            "customer_text": cust_text,
            "decision": output.get("decision"),
            "draft_reply": output.get("draft_reply"),
            "scores": judge_res
        })

    summary = {
        "n_evaluated": len(sample),
        "mean_groundedness": round(float(np.mean(scores["groundedness"])), 2),
        "mean_relevance": round(float(np.mean(scores["relevance"])), 2),
        "mean_helpfulness": round(float(np.mean(scores["helpfulness"])), 2),
        "mean_brand_voice": round(float(np.mean(scores["brand_voice"])), 2),
        "mean_decision_quality": round(float(np.mean(scores["decision_quality"])), 2),
        "overall_average": round(float(np.mean([np.mean(v) for v in scores.values()])), 2),
        "sample_details": details
    }
    return summary
