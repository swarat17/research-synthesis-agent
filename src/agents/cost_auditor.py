from src.graph.state import ResearchState
from src.storage.supabase_store import log_query
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger


def cost_auditor_node(state: ResearchState) -> dict:
    # Finalise cost tracking
    try:
        report = cost_tracker.finish_query()
    except RuntimeError as e:
        logger.warning(f"[cost_auditor] finish_query failed: {e}")
        report = {
            "query_id": state.get("query_id", "unknown"),
            "total_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "breakdown": [],
        }

    # Log to Supabase — failure is non-fatal
    try:
        log_query(
            query_id=state.get("query_id", "unknown"),
            query=state.get("query", ""),
            cost_report=report,
            num_papers=len(state.get("all_papers") or []),
            num_contradictions=len(state.get("contradictions") or []),
            num_hypotheses=len(state.get("hypotheses") or []),
        )
    except Exception as e:
        logger.warning(f"[cost_auditor] Supabase log failed (non-fatal): {e}")

    return {"cost_report": report}
