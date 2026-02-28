import pytest

from src.agents.synthesizer import synthesizer_node
from src.utils.cost_tracker import cost_tracker


@pytest.mark.integration
def test_synthesis_contains_citation_bracket():
    """Real Claude call — needs ANTHROPIC_API_KEY and PINECONE_API_KEY."""
    cost_tracker.start_query("integration-synthesis-001")

    state = {
        "query_id": "integration-synthesis-001",
        "all_papers": [
            {
                "id": "p1",
                "title": "Attention Is All You Need",
                "abstract": (
                    "The dominant sequence transduction models are based on complex recurrent "
                    "or convolutional neural networks. We propose the Transformer, a model "
                    "architecture based solely on attention mechanisms."
                ),
                "authors": ["Vaswani", "Shazeer"],
                "year": 2017,
                "url": "https://arxiv.org/abs/1706.03762",
                "source": "arxiv",
                "citation_count": 50000,
            },
            {
                "id": "p2",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": (
                    "We introduce BERT, which stands for Bidirectional Encoder Representations "
                    "from Transformers. Unlike recent language representation models, BERT is "
                    "designed to pre-train deep bidirectional representations."
                ),
                "authors": ["Devlin", "Chang"],
                "year": 2019,
                "url": "https://arxiv.org/abs/1810.04805",
                "source": "arxiv",
                "citation_count": 40000,
            },
        ],
    }

    result = synthesizer_node(state)

    try:
        cost_tracker.finish_query()
    except RuntimeError:
        pass

    synthesis = result.get("synthesis", "")
    assert len(synthesis) > 100, "Synthesis should not be empty"
    assert (
        "[" in synthesis
    ), "Synthesis should contain inline citations like [Author et al., Year]"
