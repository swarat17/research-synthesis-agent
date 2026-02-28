import json
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import ResearchState
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger

_SYSTEM = (
    "You are a research query router. Extract keywords from the query. "
    "Respond only with valid JSON, no markdown."
)

_USER_TEMPLATE = """Query: {query}

Return exactly this JSON:
{{"keywords": ["kw1", "kw2", "kw3"]}}

keywords: 3-5 specific technical terms from the query"""


def router_node(state: ResearchState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    t0 = time.time()
    try:
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=_USER_TEMPLATE.format(query=state["query"])),
            ]
        )
        latency_ms = (time.time() - t0) * 1000

        data = json.loads(response.content)
        keywords = data.get("keywords", [])

        usage = response.usage_metadata or {}
        cost_tracker.track_call(
            node_name="router",
            model="gpt-4o-mini",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
        )

        enriched = state["query"]
        if keywords:
            enriched = f"{state['query']} {' '.join(keywords)}"

        logger.info(f"[router] keywords={keywords}")
        return {"routing_decision": "arxiv", "query": enriched}

    except Exception as e:
        logger.warning(f"[router] Failed: {e}")
        return {"routing_decision": "arxiv", "errors": [f"router: {str(e)}"]}
