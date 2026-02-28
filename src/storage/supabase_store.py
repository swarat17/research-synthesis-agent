import os
from datetime import datetime, timezone

from supabase import Client, create_client

from src.utils.logger import logger


def _client() -> Client:
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )


def log_query(
    query_id: str,
    query: str,
    cost_report: dict,
    num_papers: int,
    num_contradictions: int,
    num_hypotheses: int,
) -> None:
    _client().table("query_logs").insert({
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query[:500],
        "total_cost_usd": cost_report.get("total_cost_usd", 0.0),
        "total_latency_ms": cost_report.get("total_latency_ms", 0.0),
        "num_papers": num_papers,
        "num_contradictions": num_contradictions,
        "num_hypotheses": num_hypotheses,
        "node_breakdown": cost_report.get("breakdown", []),
    }).execute()
    logger.info(f"[supabase_store] Logged query {query_id}")


def get_recent_queries(n: int = 10) -> list[dict]:
    result = (
        _client()
        .table("query_logs")
        .select("*")
        .order("timestamp", desc=True)
        .limit(n)
        .execute()
    )
    return result.data or []
