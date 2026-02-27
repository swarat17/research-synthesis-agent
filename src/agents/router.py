import json
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import ResearchState
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger

_SYSTEM = (
    "You are a research query router. Classify the query and extract keywords. "
    "Respond only with valid JSON, no markdown."
)

_USER_TEMPLATE = """Query: {query}

Return exactly this JSON:
{{"routing": "<decision>", "keywords": ["kw1", "kw2", "kw3"]}}

routing values:
- "arxiv_only"   : pure math, physics, CS theory, preprints
- "semantic_only": medicine, clinical, biology, social science
- "both"         : general ML/NLP, interdisciplinary, or uncertain

keywords: 3-5 specific technical terms from the query"""


def router_node(state: ResearchState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    t0 = time.time()
    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_USER_TEMPLATE.format(query=state["query"])),
        ])
        latency_ms = (time.time() - t0) * 1000

        data = json.loads(response.content)
        routing = data.get("routing", "both")
        keywords = data.get("keywords", [])

        if routing not in ("arxiv_only", "semantic_only", "both"):
            routing = "both"

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

        logger.info(f"[router] routing={routing} keywords={keywords}")
        return {"routing_decision": routing, "query": enriched}

    except Exception as e:
        logger.warning(f"[router] Failed, defaulting to 'both': {e}")
        return {"routing_decision": "both", "errors": [f"router: {str(e)}"]}
