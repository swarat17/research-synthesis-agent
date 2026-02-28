import pytest

from frontend.helpers import (
    format_confidence,
    format_cost,
    identify_expensive_nodes,
    severity_badge,
)


def test_format_cost_includes_both_units():
    result = format_cost(0.004)
    assert "¢" in result
    assert "$" in result


def test_format_cost_zero():
    result = format_cost(0.0)
    assert "¢" in result
    assert "$" in result


def test_severity_badge_high():
    assert severity_badge("high") == "🔴"


def test_severity_badge_medium():
    assert severity_badge("medium") == "🟡"


def test_severity_badge_low():
    assert severity_badge("low") == "🟢"


def test_severity_badge_unknown_raises():
    with pytest.raises(ValueError):
        severity_badge("critical")


def test_identify_expensive_nodes_over_threshold():
    breakdown = [
        {"node_name": "synthesizer", "cost_usd": 0.006},
        {"node_name": "router", "cost_usd": 0.004},
    ]
    # synthesizer = 60% of total → above 40% threshold
    result = identify_expensive_nodes(breakdown, threshold=0.4)
    assert "synthesizer" in result
    assert "router" not in result


def test_identify_expensive_nodes_none_over_threshold():
    breakdown = [
        {"node_name": "node_a", "cost_usd": 0.005},
        {"node_name": "node_b", "cost_usd": 0.005},
        {"node_name": "node_c", "cost_usd": 0.005},
    ]
    # each is 33% — all under 40% threshold
    result = identify_expensive_nodes(breakdown, threshold=0.4)
    assert result == []


def test_format_confidence_rounds_correctly():
    assert format_confidence(0.856) == "86%"
    assert format_confidence(0.5) == "50%"
    assert format_confidence(1.0) == "100%"
