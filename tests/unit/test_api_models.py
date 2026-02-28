import pytest
from pydantic import ValidationError

from src.api.models import QueryRequest


def test_query_too_short_fails_validation():
    with pytest.raises(ValidationError):
        QueryRequest(query="short", max_papers=10)


def test_max_papers_out_of_range_fails():
    with pytest.raises(ValidationError):
        QueryRequest(query="a valid long enough query string here", max_papers=25)
    with pytest.raises(ValidationError):
        QueryRequest(query="a valid long enough query string here", max_papers=2)


def test_valid_request_passes():
    req = QueryRequest(query="transformer attention mechanisms in NLP", max_papers=8)
    assert req.max_papers == 8
    assert len(req.query) >= 10


def test_default_max_papers_is_ten():
    req = QueryRequest(query="deep learning image segmentation methods")
    assert req.max_papers == 10
