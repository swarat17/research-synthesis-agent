import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.contradiction import contradiction_node

# ── helpers ────────────────────────────────────────────────────────────────


def _papers(n):
    return [
        {
            "id": f"p{i}",
            "title": f"Paper {i}",
            "abstract": "A" * 100,
            "authors": ["Author"],
            "year": 2023,
            "source": "arxiv",
            "citation_count": 0,
        }
        for i in range(n)
    ]


def _mock_response(content: str):
    resp = MagicMock()
    resp.content = content
    resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    return resp


_VALID_ITEM = {
    "claim_a": "Approach X is superior",
    "claim_b": "Approach Y is superior",
    "paper_a_title": "Paper 0",
    "paper_b_title": "Paper 1",
    "severity": "high",
    "topic": "methodology",
}


@pytest.fixture(autouse=True)
def mock_cost():
    with patch("src.agents.contradiction.cost_tracker.track_call"):
        yield


# ── tests ───────────────────────────────────────────────────────────────────


def test_returns_empty_for_fewer_than_two_papers():
    with patch("src.agents.contradiction.ChatOpenAI") as mock_cls:
        result = contradiction_node({"all_papers": _papers(1)})
    assert result["contradictions"] == []
    mock_cls.assert_not_called()


def test_parses_valid_contradiction_response():
    content = json.dumps({"contradictions": [_VALID_ITEM]})
    with patch("src.agents.contradiction.ChatOpenAI") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(content)
        result = contradiction_node({"all_papers": _papers(2)})
    assert len(result["contradictions"]) == 1
    c = result["contradictions"][0]
    assert c["claim_a"] == "Approach X is superior"
    assert c["severity"] == "high"
    assert c["topic"] == "methodology"


def test_empty_contradictions_response_is_valid():
    content = json.dumps({"contradictions": []})
    with patch("src.agents.contradiction.ChatOpenAI") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(content)
        result = contradiction_node({"all_papers": _papers(3)})
    assert result["contradictions"] == []
    assert not result.get("errors")


def test_handles_json_parse_failure_gracefully():
    with patch("src.agents.contradiction.ChatOpenAI") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response("not valid json {{")
        result = contradiction_node({"all_papers": _papers(2)})
    assert result["contradictions"] == []
    assert len(result.get("errors", [])) == 1
    assert "contradiction_detector" in result["errors"][0]


def test_severity_values_are_constrained():
    bad_item = {**_VALID_ITEM, "severity": "critical"}  # invalid value
    content = json.dumps({"contradictions": [bad_item]})
    with patch("src.agents.contradiction.ChatOpenAI") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(content)
        result = contradiction_node({"all_papers": _papers(2)})
    assert result["contradictions"][0]["severity"] in {"high", "medium", "low"}


def test_caps_paper_input_at_eight():
    content = json.dumps({"contradictions": []})
    with patch("src.agents.contradiction.ChatOpenAI") as mock_cls:
        mock_cls.return_value.invoke.return_value = _mock_response(content)
        contradiction_node({"all_papers": _papers(20)})

    # Extract the HumanMessage content from the invoke call
    messages = mock_cls.return_value.invoke.call_args[0][0]
    prompt_text = messages[1].content  # index 1 = HumanMessage

    # Papers 0-7 may appear; papers 8-19 must NOT
    for i in range(8, 20):
        assert f"Paper {i}" not in prompt_text
