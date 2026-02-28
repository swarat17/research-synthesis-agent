import pytest

from src.graph.pipeline import graph
from src.utils.cost_tracker import cost_tracker


@pytest.mark.integration
def test_full_pipeline_produces_synthesis_and_hypotheses():
    """Real call — needs OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY."""
    query_id = "integration-phase4-001"
    cost_tracker.start_query(query_id)

    result = graph.invoke({
        "query": "deep learning image segmentation",
        "query_id": query_id,
        "max_papers": 6,
        "errors": [],
    })

    try:
        cost_tracker.finish_query()
    except RuntimeError:
        pass

    assert result.get("synthesis"), f"Synthesis empty. Errors: {result.get('errors')}"
    assert len(result.get("hypotheses", [])) >= 1, (
        f"Expected ≥1 hypothesis. Errors: {result.get('errors')}"
    )
