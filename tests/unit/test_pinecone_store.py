from unittest.mock import MagicMock, patch

import pytest

from src.utils.cost_tracker import cost_tracker


def _make_papers(n=3):
    return [
        {
            "id": f"paper-{i}",
            "title": f"Paper Title {i}",
            "abstract": "A" * 100,
            "authors": ["Author One"],
            "year": 2023,
            "url": f"https://example.com/{i}",
            "source": "arxiv",
            "citation_count": i * 10,
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def active_query():
    cost_tracker.start_query("test-pinecone-q")
    yield
    try:
        cost_tracker.finish_query()
    except RuntimeError:
        pass


def _mock_embed_response(n_texts, total_tokens=500):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n_texts)]
    mock_response.usage.total_tokens = total_tokens
    return mock_response


def test_upsert_count_matches_input():
    papers = _make_papers(3)
    mock_index = MagicMock()

    with patch("src.storage.pinecone_store.OpenAI") as mock_openai_cls, patch(
        "src.storage.pinecone_store._get_index", return_value=mock_index
    ):
        mock_openai_cls.return_value.embeddings.create.return_value = (
            _mock_embed_response(3)
        )

        from src.storage.pinecone_store import embed_and_upsert

        count = embed_and_upsert(papers, "qid-001")

    assert count == 3
    mock_index.upsert.assert_called_once()
    vectors = mock_index.upsert.call_args.kwargs["vectors"]
    assert len(vectors) == 3


def test_vector_metadata_has_required_fields():
    papers = _make_papers(1)
    mock_index = MagicMock()

    with patch("src.storage.pinecone_store.OpenAI") as mock_openai_cls, patch(
        "src.storage.pinecone_store._get_index", return_value=mock_index
    ):
        mock_openai_cls.return_value.embeddings.create.return_value = (
            _mock_embed_response(1)
        )

        from src.storage.pinecone_store import embed_and_upsert

        embed_and_upsert(papers, "qid-002")

    vectors = mock_index.upsert.call_args.kwargs["vectors"]
    metadata = vectors[0]["metadata"]
    for field in ("title", "year", "source", "url", "abstract"):
        assert field in metadata, f"Missing metadata field: {field}"


def test_namespace_uses_query_id():
    papers = _make_papers(2)
    mock_index = MagicMock()
    query_id = "my-namespace-123"

    with patch("src.storage.pinecone_store.OpenAI") as mock_openai_cls, patch(
        "src.storage.pinecone_store._get_index", return_value=mock_index
    ):
        mock_openai_cls.return_value.embeddings.create.return_value = (
            _mock_embed_response(2)
        )

        from src.storage.pinecone_store import embed_and_upsert

        embed_and_upsert(papers, query_id)

    assert mock_index.upsert.call_args.kwargs["namespace"] == query_id
