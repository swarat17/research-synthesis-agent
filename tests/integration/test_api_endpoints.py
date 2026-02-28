from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_short_query_returns_422():
    """FastAPI returns 422 Unprocessable Entity for Pydantic validation failures."""
    response = client.post("/analyze", json={"query": "short", "max_papers": 10})
    assert response.status_code == 422


def test_stats_returns_200_on_supabase_failure():
    """Stats endpoint should return empty stats (not 500) when Supabase is unavailable."""
    with patch("src.api.main.get_recent_queries", side_effect=Exception("no db")):
        response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_queries"] == 0
    assert data["error"] is not None


@pytest.mark.integration
def test_analyze_returns_full_response():
    """Real call — needs OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY, SUPABASE_URL/KEY."""
    response = client.post("/analyze", json={
        "query": "transformer attention mechanisms in NLP",
        "max_papers": 4,
    })
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "synthesis" in data
    assert "contradictions" in data
    assert "hypotheses" in data
    assert "cost_report" in data
    assert "errors" in data
