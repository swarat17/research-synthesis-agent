import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import ResearchState
from src.storage.pinecone_store import embed_and_upsert
from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger

_MODEL = "claude-sonnet-4-6"

_SYSTEM = (
    "You are a scientific research synthesizer. Write a clear, structured narrative "
    "synthesis of the provided papers. Group findings by theme. Use inline citations "
    "in the format [Author et al., Year]. Aim for 400-600 words."
)


def _build_prompt(papers: list[dict]) -> str:
    lines = ["Synthesize the following research papers:\n"]
    for i, p in enumerate(papers[:10], 1):
        authors = p.get("authors", [])
        author_str = ", ".join(authors[:2])
        if len(authors) > 2:
            author_str += " et al."
        lines.append(
            f"{i}. **{p.get('title', 'Untitled')}** "
            f"({author_str}, {p.get('year', 'n.d.')})\n"
            f"   {p.get('abstract', '')[:300]}\n"
        )
    lines.append(
        "\nWrite a 400-600 word synthesis grouping papers by theme. "
        "Use inline citations like [Author et al., Year]."
    )
    return "\n".join(lines)


def synthesizer_node(state: ResearchState) -> dict:
    papers = state.get("all_papers") or []
    query_id = state.get("query_id", "unknown")

    # Embed & store in Pinecone (best-effort)
    try:
        embed_and_upsert(papers, query_id)
    except Exception as e:
        logger.warning(f"[synthesizer] Pinecone upsert failed (continuing): {e}")

    if not papers:
        return {"synthesis": "", "errors": ["synthesizer: no papers to synthesize"]}

    prompt = _build_prompt(papers)
    llm = ChatAnthropic(model=_MODEL, temperature=0)

    t0 = time.time()
    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        latency_ms = (time.time() - t0) * 1000

        usage = response.usage_metadata or {}
        cost_tracker.track_call(
            node_name="synthesizer",
            model=_MODEL,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
        )

        synthesis = response.content
        logger.info(f"[synthesizer] Generated {len(synthesis)} char synthesis")
        return {"synthesis": synthesis}

    except Exception as e:
        logger.error(f"[synthesizer] Failed: {e}")
        return {"synthesis": "", "errors": [f"synthesizer: {str(e)}"]}
