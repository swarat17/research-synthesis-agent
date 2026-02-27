import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.fetchers import arxiv_fetcher, semantic_fetcher

REQUIRED_PAPER_FIELDS = {"id", "title", "abstract", "authors", "year", "url", "source", "citation_count"}


# ── helpers ────────────────────────────────────────────────────────────────

def _mock_arxiv_result(title="Test Paper", abstract=None):
    result = MagicMock()
    result.entry_id = "https://arxiv.org/abs/2301.00001"
    result.title = title
    result.summary = abstract or (
        "This is a sufficiently long abstract for testing purposes, definitely over fifty characters."
    )
    author = MagicMock()
    author.__str__ = MagicMock(return_value="Author One")
    result.authors = [author]
    result.published = datetime.datetime(2023, 1, 1)
    return result


def _base_state(routing="both", max_papers=4):
    return {"routing_decision": routing, "max_papers": max_papers, "query": "test query"}


# ── arxiv tests ─────────────────────────────────────────────────────────────

def test_arxiv_skipped_when_semantic_only():
    with patch("src.agents.fetchers.arxiv.Client") as mock_client_cls:
        result = arxiv_fetcher(_base_state(routing="semantic_only"))
    assert result == {"arxiv_papers": []}
    mock_client_cls.assert_not_called()


def test_normalized_paper_has_all_required_fields():
    mock_result = _mock_arxiv_result()
    with patch("src.agents.fetchers.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_cls.return_value = mock_client

        result = arxiv_fetcher(_base_state())

    papers = result["arxiv_papers"]
    assert len(papers) == 1
    assert REQUIRED_PAPER_FIELDS.issubset(papers[0].keys())
    assert papers[0]["source"] == "arxiv"


def test_short_abstract_filtered_out():
    short_result = _mock_arxiv_result(abstract="Too short.")
    long_result = _mock_arxiv_result(abstract="A" * 60)

    with patch("src.agents.fetchers.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter([short_result, long_result])
        mock_client_cls.return_value = mock_client

        result = arxiv_fetcher(_base_state())

    assert len(result["arxiv_papers"]) == 1


def test_network_error_returns_empty_with_error_entry():
    with patch("src.agents.fetchers.arxiv.Client") as mock_client_cls:
        mock_client_cls.side_effect = ConnectionError("network down")
        result = arxiv_fetcher(_base_state())

    assert result["arxiv_papers"] == []
    assert len(result.get("errors", [])) == 1
    assert "arxiv_fetcher" in result["errors"][0]


# ── semantic scholar tests ──────────────────────────────────────────────────

def test_semantic_skipped_when_arxiv_only():
    with patch("src.agents.fetchers.SemanticScholar") as mock_sch_cls:
        result = semantic_fetcher(_base_state(routing="arxiv_only"))
    assert result == {"semantic_papers": []}
    mock_sch_cls.assert_not_called()


def test_semantic_network_error_returns_empty_with_error_entry():
    with patch("src.agents.fetchers.SemanticScholar") as mock_sch_cls:
        mock_sch_cls.side_effect = ConnectionError("network down")
        result = semantic_fetcher(_base_state())

    assert result["semantic_papers"] == []
    assert len(result.get("errors", [])) == 1
    assert "semantic_fetcher" in result["errors"][0]
