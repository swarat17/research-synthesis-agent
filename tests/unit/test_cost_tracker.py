import pytest

from src.utils.cost_tracker import CostTracker, CostLimitExceededError


@pytest.fixture(autouse=True)
def fresh_tracker():
    """Return a fresh CostTracker instance for every test (avoids singleton bleed)."""
    tracker = CostTracker()
    yield tracker


def test_basic_cost_calculation(fresh_tracker):
    tracker = fresh_tracker
    tracker.start_query("q1")
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    # 1000 input + 500 output = (1000*0.15 + 500*0.60) / 1_000_000
    #                         = (0.15 + 0.30) / 1_000_000 * 1_000_000 ... let's be precise:
    # cost = (1000 * 0.15 + 500 * 0.60) / 1_000_000 = (150 + 300) / 1_000_000 = 0.00045
    cost = tracker.track_call("router", "gpt-4o-mini", 1000, 500, 100)
    assert abs(cost - 0.00045) < 1e-9
    report = tracker.finish_query()
    assert abs(report["total_cost_usd"] - 0.00045) < 1e-9


def test_multi_node_accumulation(fresh_tracker):
    tracker = fresh_tracker
    tracker.start_query("q2")
    cost1 = tracker.track_call("router", "gpt-4o-mini", 500, 200, 80)
    cost2 = tracker.track_call("synthesizer", "gpt-4o-mini", 800, 400, 120)
    report = tracker.finish_query()
    assert abs(report["total_cost_usd"] - (cost1 + cost2)) < 1e-9
    assert abs(report["total_latency_ms"] - 200) < 1e-6
    assert len(report["breakdown"]) == 2


def test_cost_limit_raises_custom_error(fresh_tracker, monkeypatch):
    monkeypatch.setenv("MAX_COST_PER_QUERY", "0.000001")
    tracker = fresh_tracker
    tracker.start_query("q3")
    with pytest.raises(CostLimitExceededError):
        tracker.track_call("router", "gpt-4o-mini", 10000, 5000, 50)


def test_finish_query_resets_state(fresh_tracker):
    tracker = fresh_tracker
    tracker.start_query("q4")
    tracker.track_call("router", "gpt-4o-mini", 100, 50, 10)
    tracker.finish_query()
    # After finish, track_call without start_query must raise
    with pytest.raises(RuntimeError):
        tracker.track_call("router", "gpt-4o-mini", 100, 50, 10)


def test_report_structure_has_all_fields(fresh_tracker):
    tracker = fresh_tracker
    tracker.start_query("q5")
    tracker.track_call("router", "gpt-4o-mini", 200, 100, 30)
    report = tracker.finish_query()
    assert "query_id" in report
    assert "total_cost_usd" in report
    assert "total_latency_ms" in report
    assert "breakdown" in report
    assert report["query_id"] == "q5"


def test_claude_costs_more_than_gpt_same_tokens(fresh_tracker):
    tracker = fresh_tracker

    tracker.start_query("gpt_query")
    gpt_cost = tracker.track_call("node", "gpt-4o-mini", 1000, 1000, 10)
    tracker.finish_query()

    tracker.start_query("claude_query")
    claude_cost = tracker.track_call("node", "claude-sonnet-4", 1000, 1000, 10)
    tracker.finish_query()

    assert claude_cost > gpt_cost
