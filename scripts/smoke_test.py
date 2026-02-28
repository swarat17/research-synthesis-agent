"""
Standalone smoke test for Phase 2.
Run with: .venv/Scripts/python scripts/smoke_test.py
Requires OPENAI_API_KEY in environment / .env file.
"""
import os
import sys

# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.graph.pipeline import graph
from src.utils.cost_tracker import cost_tracker

QUERY = "transformer attention mechanisms in NLP"
MAX_PAPERS = 6
QUERY_ID = "smoke-001"

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Smoke test — query: {QUERY!r}")
    print(f"{'='*60}\n")

    cost_tracker.start_query(QUERY_ID)

    initial_state = {
        "query": QUERY,
        "query_id": QUERY_ID,
        "max_papers": MAX_PAPERS,
        "errors": [],
    }

    result = graph.invoke(initial_state)

    arxiv_count = len(result.get("arxiv_papers", []))
    errors = result.get("errors", [])

    print(f"Routing decision : {result.get('routing_decision')}")
    print(f"ArXiv papers     : {arxiv_count}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    try:
        report = cost_tracker.finish_query()
        print(f"\nCost report:")
        print(f"  Total cost    : ${report['total_cost_usd']:.6f}")
        print(f"  Total latency : {report['total_latency_ms']:.0f}ms")
        print(f"  Breakdown:")
        for node in report["breakdown"]:
            print(f"    {node['node_name']}: ${node['cost_usd']:.6f} ({node['latency_ms']:.0f}ms)")
    except RuntimeError:
        print("\n[No cost report — cost_tracker may not have been active]")

    print(f"\n{'='*60}\n")
