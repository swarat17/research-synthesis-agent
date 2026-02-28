import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    query: str
    original_query: str
    query_id: str
    max_papers: int
    routing_decision: str
    arxiv_papers: list
    all_papers: list
    synthesis: str
    contradictions: list
    hypotheses: list
    cost_report: dict
    errors: Annotated[list, operator.add]
