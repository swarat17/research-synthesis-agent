import pytest

from src.graph.pipeline import graph
from src.utils.cost_tracker import cost_tracker


@pytest.mark.integration
def test_graph_fetches_papers_for_known_query():
    """Real network call — needs OPENAI_API_KEY set."""
    query_id = "integration-phase2-001"
    cost_tracker.start_query(query_id)

    result = graph.invoke(
        {
            "query": "BERT pretraining language models",
            "query_id": query_id,
            "max_papers": 4,
            "errors": [],
        }
    )

    try:
        cost_tracker.finish_query()
    except RuntimeError:
        pass

    arxiv_papers = result.get("arxiv_papers", [])
    assert (
        len(arxiv_papers) >= 1
    ), f"Expected at least 1 paper, got 0. Errors: {result.get('errors')}"
