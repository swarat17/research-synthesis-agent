"""
Integration test configuration:
- Loads .env so real API keys are available
- Resets CostTracker singleton between tests to prevent state leakage
"""

import pytest
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root before any test runs

from src.utils.cost_tracker import cost_tracker  # noqa: E402


@pytest.fixture(autouse=True)
def reset_cost_tracker():
    """Ensure CostTracker has no active query before and after each test."""
    if cost_tracker._report is not None:
        cost_tracker.finish_query()
    yield
    if cost_tracker._report is not None:
        cost_tracker.finish_query()
