import os
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import logger


class CostLimitExceededError(Exception):
    pass


# Pricing per 1M tokens (input, output) in USD
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.02},
}


def _get_model_pricing(model: str) -> dict:
    """Match model name against pricing table (supports wildcard prefix like claude-sonnet-4-*)."""
    if model in PRICING:
        return PRICING[model]
    for key in PRICING:
        if model.startswith(key):
            return PRICING[key]
    raise ValueError(f"Unknown model for pricing: {model}")


@dataclass
class NodeCost:
    node_name: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass
class QueryCostReport:
    query_id: str
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    node_costs: list = field(default_factory=list)


class CostTracker:
    def __init__(self):
        self._report: Optional[QueryCostReport] = None

    def start_query(self, query_id: str) -> None:
        if self._report is not None:
            raise RuntimeError(
                f"Query '{self._report.query_id}' is already active. Call finish_query() first."
            )
        self._report = QueryCostReport(query_id=query_id)
        logger.info(f"[CostTracker] Started query: {query_id}")

    def track_call(
        self,
        node_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> float:
        if self._report is None:
            raise RuntimeError("No active query. Call start_query() first.")

        pricing = _get_model_pricing(model)
        cost = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000

        self._report.total_cost_usd += cost
        self._report.total_latency_ms += latency_ms
        self._report.node_costs.append(
            NodeCost(
                node_name=node_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )
        )

        logger.info(
            f"[CostTracker] {node_name} | {model} | "
            f"in={input_tokens} out={output_tokens} | "
            f"${cost:.6f} | {latency_ms:.0f}ms | "
            f"running total=${self._report.total_cost_usd:.6f}"
        )

        max_cost = os.getenv("MAX_COST_PER_QUERY")
        if max_cost is not None and self._report.total_cost_usd > float(max_cost):
            raise CostLimitExceededError(
                f"Cost limit ${max_cost} exceeded: current total ${self._report.total_cost_usd:.6f}"
            )

        warning_threshold = os.getenv("COST_WARNING_THRESHOLD")
        if warning_threshold is not None and self._report.total_cost_usd > float(
            warning_threshold
        ):
            logger.warning(
                f"[CostTracker] Warning threshold ${warning_threshold} exceeded: "
                f"${self._report.total_cost_usd:.6f}"
            )

        return cost

    def finish_query(self) -> dict:
        if self._report is None:
            raise RuntimeError("No active query to finish.")

        report = self._report
        self._report = None

        breakdown = [
            {
                "node_name": nc.node_name,
                "model": nc.model,
                "input_tokens": nc.input_tokens,
                "output_tokens": nc.output_tokens,
                "latency_ms": nc.latency_ms,
                "cost_usd": nc.cost_usd,
            }
            for nc in report.node_costs
        ]

        result = {
            "query_id": report.query_id,
            "total_cost_usd": report.total_cost_usd,
            "total_latency_ms": report.total_latency_ms,
            "breakdown": breakdown,
        }

        logger.info(
            f"[CostTracker] Finished query: {report.query_id} | "
            f"total=${report.total_cost_usd:.6f} | {report.total_latency_ms:.0f}ms"
        )
        return result


cost_tracker = CostTracker()
