import json
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import ResearchState
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger

_MODEL = "gpt-4o-mini"
_VALID_SEVERITIES = {"high", "medium", "low"}

_SYSTEM = """You are a scientific contradiction detector. Given research paper abstracts, identify claims that directly contradict each other.
Return ONLY valid JSON — no markdown, no explanation:
{"contradictions": [{"claim_a": "...", "claim_b": "...", "paper_a_title": "...", "paper_b_title": "...", "severity": "high|medium|low", "topic": "..."}]}
If no contradictions found, return {"contradictions": []}."""


def _build_prompt(papers: list[dict]) -> str:
    lines = ["Identify contradictions between claims in these papers:\n"]
    for i, p in enumerate(papers[:8], 1):
        lines.append(f"{i}. **{p.get('title', 'Untitled')}** ({p.get('year', 'n.d.')})")
        lines.append(f"   {p.get('abstract', '')[:300]}\n")
    return "\n".join(lines)


def contradiction_node(state: ResearchState) -> dict:
    papers = state.get("all_papers") or []

    if len(papers) < 2:
        logger.info("[contradiction_detector] Fewer than 2 papers — skipping")
        return {"contradictions": []}

    llm = ChatOpenAI(model=_MODEL, temperature=0)
    prompt = _build_prompt(papers)

    t0 = time.time()
    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        latency_ms = (time.time() - t0) * 1000

        usage = response.usage_metadata or {}
        cost_tracker.track_call(
            node_name="contradiction_detector",
            model=_MODEL,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
        )

        data = json.loads(response.content)
        contradictions = data.get("contradictions", [])

        for c in contradictions:
            if c.get("severity") not in _VALID_SEVERITIES:
                c["severity"] = "low"

        logger.info(f"[contradiction_detector] Found {len(contradictions)} contradictions")
        return {"contradictions": contradictions}

    except json.JSONDecodeError as e:
        logger.warning(f"[contradiction_detector] JSON parse failed: {e}")
        return {"contradictions": [], "errors": [f"contradiction_detector: JSON parse failed: {e}"]}
    except Exception as e:
        logger.error(f"[contradiction_detector] Failed: {e}")
        return {"contradictions": [], "errors": [f"contradiction_detector: {e}"]}
