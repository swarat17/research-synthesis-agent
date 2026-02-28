import json
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import ResearchState
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger

_MODEL = "claude-sonnet-4-6"
_VALID_NOVELTY = {"high", "medium", "low"}

_SYSTEM = """You are a scientific hypothesis generator. Generate exactly 3 novel, testable research hypotheses.
Return ONLY valid JSON — no markdown, no explanation:
{"hypotheses": [
  {
    "hypothesis": "...",
    "rationale": "...",
    "confidence": 0.0,
    "novelty": "high|medium|low",
    "suggested_method": "...",
    "supporting_papers": ["title1", "title2"]
  }
]}
confidence must be a float between 0.0 and 1.0."""


def _build_prompt(synthesis: str, contradictions: list[dict]) -> str:
    lines = [f"## Research Synthesis\n{synthesis}\n"]
    if contradictions:
        lines.append("## Identified Contradictions")
        for c in contradictions:
            lines.append(
                f"- [{c.get('severity', '').upper()}] {c.get('claim_a', '')} "
                f"vs {c.get('claim_b', '')} (Topic: {c.get('topic', '')})"
            )
        lines.append("")
    lines.append(
        "Based on the synthesis and contradictions above, generate exactly 3 "
        "novel research hypotheses that could meaningfully advance this field."
    )
    return "\n".join(lines)


def hypothesis_node(state: ResearchState) -> dict:
    synthesis = state.get("synthesis") or ""
    contradictions = state.get("contradictions") or []

    llm = ChatAnthropic(model=_MODEL, temperature=0.7)
    prompt = _build_prompt(synthesis, contradictions)

    t0 = time.time()
    try:
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        latency_ms = (time.time() - t0) * 1000

        usage = response.usage_metadata or {}
        cost_tracker.track_call(
            node_name="hypothesis_generator",
            model=_MODEL,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
        )

        data = json.loads(response.content)
        hypotheses = data.get("hypotheses", [])

        for h in hypotheses:
            if h.get("novelty") not in _VALID_NOVELTY:
                h["novelty"] = "medium"
            h["confidence"] = max(0.0, min(1.0, float(h.get("confidence", 0.5))))

        logger.info(f"[hypothesis_generator] Generated {len(hypotheses)} hypotheses")
        return {"hypotheses": hypotheses}

    except json.JSONDecodeError as e:
        logger.warning(f"[hypothesis_generator] JSON parse failed: {e}")
        return {
            "hypotheses": [],
            "errors": [f"hypothesis_generator: JSON parse failed: {e}"],
        }
    except Exception as e:
        logger.error(f"[hypothesis_generator] Failed: {e}")
        return {"hypotheses": [], "errors": [f"hypothesis_generator: {e}"]}
