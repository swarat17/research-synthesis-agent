import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.hypothesis import hypothesis_node


# ── helpers ────────────────────────────────────────────────────────────────

def _mock_response(content: str):
    resp = MagicMock()
    resp.content = content
    resp.usage_metadata = {"input_tokens": 200, "output_tokens": 300}
    return resp


def _hypotheses_json(n=3, novelty="high", confidence=0.8):
    items = [
        {
            "hypothesis": f"Hypothesis {i}: Novel approach to the problem",
            "rationale": f"Rationale for hypothesis {i} based on synthesis",
            "confidence": confidence,
            "novelty": novelty,
            "suggested_method": "Randomized controlled trial",
            "supporting_papers": ["Paper A", "Paper B"],
        }
        for i in range(n)
    ]
    return json.dumps({"hypotheses": items})


@pytest.fixture(autouse=True)
def mock_cost():
    with patch("src.agents.hypothesis.cost_tracker.track_call"):
        yield


def _state(synthesis="Research synthesis text.", contradictions=None):
    return {"synthesis": synthesis, "contradictions": contradictions or []}


# ── tests ───────────────────────────────────────────────────────────────────

def test_generates_three_hypotheses():
    with patch("src.agents.hypothesis.ChatAnthropic") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(_hypotheses_json(3))
        result = hypothesis_node(_state())
    assert len(result["hypotheses"]) == 3


def test_confidence_is_float_in_range():
    with patch("src.agents.hypothesis.ChatAnthropic") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(_hypotheses_json(3, confidence=0.75))
        result = hypothesis_node(_state())
    for h in result["hypotheses"]:
        assert isinstance(h["confidence"], float)
        assert 0.0 <= h["confidence"] <= 1.0


def test_contradictions_appear_in_prompt():
    contradictions = [
        {
            "claim_a": "X outperforms Y on benchmarks",
            "claim_b": "Y outperforms X on benchmarks",
            "severity": "high",
            "topic": "performance",
            "paper_a_title": "Paper A",
            "paper_b_title": "Paper B",
        }
    ]
    with patch("src.agents.hypothesis.ChatAnthropic") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(_hypotheses_json())
        hypothesis_node(_state(contradictions=contradictions))

    messages = mock_cls.return_value.invoke.call_args[0][0]
    prompt_text = messages[1].content  # HumanMessage
    assert "X outperforms Y on benchmarks" in prompt_text


def test_handles_parse_failure_gracefully():
    with patch("src.agents.hypothesis.ChatAnthropic") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response("}{bad json")
        result = hypothesis_node(_state())
    assert result["hypotheses"] == []
    assert len(result.get("errors", [])) == 1
    assert "hypothesis_generator" in result["errors"][0]


def test_novelty_values_are_constrained():
    content = json.dumps({"hypotheses": [{
        "hypothesis": "Test",
        "rationale": "Test rationale",
        "confidence": 0.8,
        "novelty": "EXTREME",          # invalid
        "suggested_method": "Method",
        "supporting_papers": [],
    }]})
    with patch("src.agents.hypothesis.ChatAnthropic") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(content)
        result = hypothesis_node(_state())
    assert result["hypotheses"][0]["novelty"] in {"high", "medium", "low"}
