import secrets
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.models import HealthResponse, QueryRequest, QueryResponse, StatsResponse
from src.graph.pipeline import graph
from src.storage.supabase_store import get_recent_queries
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger

app = FastAPI(title="Research Synthesis Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=QueryResponse)
async def analyze(request: QueryRequest):
    query_id = secrets.token_hex(4)
    logger.info(f"[/analyze] query_id={query_id} query={request.query!r}")

    try:
        cost_tracker.start_query(query_id)

        result = graph.invoke({
            "query": request.query,
            "original_query": request.query,
            "query_id": query_id,
            "max_papers": request.max_papers,
            "errors": [],
        })

        return QueryResponse(
            query_id=query_id,
            papers=result.get("all_papers") or [],
            synthesis=result.get("synthesis") or "",
            contradictions=result.get("contradictions") or [],
            hypotheses=result.get("hypotheses") or [],
            cost_report=result.get("cost_report") or {},
            errors=result.get("errors") or [],
        )

    except Exception as e:
        logger.error(f"[/analyze] Unhandled error: {e}")
        try:
            cost_tracker.finish_query()
        except RuntimeError:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    try:
        rows = get_recent_queries(10)
        if not rows:
            return StatsResponse(
                total_queries=0,
                avg_cost_usd=0.0,
                avg_latency_ms=0.0,
                avg_papers=0.0,
                recent_queries=[],
            )
        return StatsResponse(
            total_queries=len(rows),
            avg_cost_usd=sum(r["total_cost_usd"] for r in rows) / len(rows),
            avg_latency_ms=sum(r["total_latency_ms"] for r in rows) / len(rows),
            avg_papers=sum(r["num_papers"] for r in rows) / len(rows),
            recent_queries=rows,
        )
    except Exception as e:
        logger.warning(f"[/stats] Failed: {e}")
        return StatsResponse(
            total_queries=0,
            avg_cost_usd=0.0,
            avg_latency_ms=0.0,
            avg_papers=0.0,
            recent_queries=[],
            error=str(e),
        )


# AWS Lambda handler
handler = Mangum(app)
