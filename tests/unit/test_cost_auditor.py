from unittest.mock import patch

import pytest

from src.agents.cost_auditor import cost_auditor_node

_REPORT = {
    "query_id": "test-q",
    "total_cost_usd": 0.05,
    "total_latency_ms": 1500.0,
    "breakdown": [{"node_name": "router", "cost_usd": 0.05}],
}

_STATE = {
    "query_id": "test-q",
    "query": "test query about transformers",
    "all_papers": [{}, {}, {}],
    "contradictions": [{}],
    "hypotheses": [{}, {}],
}


@pytest.fixture(autouse=True)
def mock_finish_query():
    with patch("src.agents.cost_auditor.cost_tracker.finish_query", return_value=_REPORT):
        yield


def test_cost_report_populated_in_state():
    with patch("src.agents.cost_auditor.log_query"):
        result = cost_auditor_node(_STATE)

    assert "cost_report" in result
    assert result["cost_report"]["query_id"] == "test-q"
    assert result["cost_report"]["total_cost_usd"] == 0.05
    assert result["cost_report"]["total_latency_ms"] == 1500.0


def test_supabase_failure_does_not_propagate():
    with patch("src.agents.cost_auditor.log_query", side_effect=Exception("Supabase down")):
        # must not raise
        result = cost_auditor_node(_STATE)

    assert "cost_report" in result
    assert result["cost_report"]["total_cost_usd"] == 0.05
