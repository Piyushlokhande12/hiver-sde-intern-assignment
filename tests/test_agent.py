"""
Integration tests for End-to-End AIAgent.
"""
from src.agent.support_agent import AIAgent


def test_agent_output_schema():
    agent = AIAgent()
    query = "My package was supposed to arrive yesterday but tracking hasn't updated"
    res = agent.process(query)

    assert "intent" in res
    assert "confidence" in res
    assert "decision" in res
    assert "reason" in res
    assert "draft_reply" in res
    assert "evidence" in res
    assert isinstance(res["confidence"], float)
    assert res["decision"] in ["auto_handle", "escalate"]
    assert isinstance(res["evidence"], list)
