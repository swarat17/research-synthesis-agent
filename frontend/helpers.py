def format_cost(usd: float) -> str:
    """Format a USD cost as '0.40¢ ($0.0040)'."""
    cents = usd * 100
    return f"{cents:.2f}¢ (${usd:.4f})"


def severity_badge(severity: str) -> str:
    """Map severity to emoji. Raises ValueError for unknown values."""
    badges = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    if severity not in badges:
        raise ValueError(
            f"Unknown severity: {severity!r}. Expected one of {set(badges)}"
        )
    return badges[severity]


def identify_expensive_nodes(
    breakdown: list[dict], threshold: float = 0.4
) -> list[str]:
    """Return node names whose cost fraction exceeds threshold."""
    total = sum(n.get("cost_usd", 0.0) for n in breakdown)
    if total == 0:
        return []
    return [
        n["node_name"]
        for n in breakdown
        if (n.get("cost_usd", 0.0) / total) > threshold
    ]


def format_confidence(confidence: float) -> str:
    """Format a 0.0–1.0 confidence as a percentage string, e.g. '86%'."""
    return f"{round(confidence * 100)}%"
