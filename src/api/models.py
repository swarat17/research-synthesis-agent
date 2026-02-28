from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=1000)
    max_papers: int = Field(default=10, ge=4, le=20)


class QueryResponse(BaseModel):
    query_id: str
    papers: list[dict]
    synthesis: str
    contradictions: list[dict]
    hypotheses: list[dict]
    cost_report: dict
    errors: list[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class StatsResponse(BaseModel):
    total_queries: int
    avg_cost_usd: float
    avg_latency_ms: float
    avg_papers: float
    recent_queries: list[dict]
    error: Optional[str] = None
