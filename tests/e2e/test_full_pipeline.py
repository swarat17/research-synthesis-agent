"""
End-to-end test against a live Lambda deployment.
Requires env var: E2E_API_URL=https://<id>.execute-api.<region>.amazonaws.com
All API keys must also be set in the Lambda environment.
"""
import os
import time

import httpx
import pytest

E2E_API_URL = os.getenv("E2E_API_URL", "").rstrip("/")


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_API_URL, reason="E2E_API_URL not set")
def test_full_pipeline():
    t0 = time.time()

    with httpx.Client(timeout=90.0) as client:
        resp = client.post(
            f"{E2E_API_URL}/analyze",
            json={"query": "deep learning image segmentation neural networks", "max_papers": 6},
        )

    elapsed = time.time() - t0

    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    assert elapsed < 60, f"Response took {elapsed:.1f}s — exceeded 60s limit"

    data = resp.json()
    query_id = data.get("query_id")

    assert data.get("synthesis"), "synthesis must not be empty"
    assert len(data.get("hypotheses", [])) >= 1, "must have at least 1 hypothesis"

    cost = data.get("cost_report", {}).get("total_cost_usd", 999)
    assert cost < 0.50, f"Cost ${cost:.4f} exceeded $0.50 limit"

    # Verify Supabase record was written
    from src.storage.supabase_store import get_recent_queries
    rows = get_recent_queries(10)
    assert any(r.get("query_id") == query_id for r in rows), (
        f"query_id {query_id!r} not found in Supabase logs"
    )
