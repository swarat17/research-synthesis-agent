import re

from src.graph.state import ResearchState
from src.utils.logger import logger


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[-_]", " ", title)      # hyphens/underscores → spaces (so Pre-Training == Pre Training)
    title = re.sub(r"[^\w\s]", "", title)    # strip remaining punctuation
    title = re.sub(r"\s+", " ", title).strip()
    return title


def deduplicator_node(state: ResearchState) -> dict:
    arxiv = state.get("arxiv_papers") or []
    semantic = state.get("semantic_papers") or []
    combined = arxiv + semantic

    # Filter short abstracts
    papers = [p for p in combined if len(p.get("abstract", "")) >= 50]

    # Deduplicate: keep higher citation_count on collision
    seen: dict[str, dict] = {}
    for paper in papers:
        key = _normalize_title(paper.get("title", ""))
        if key not in seen:
            seen[key] = paper
        elif (paper.get("citation_count") or 0) > (seen[key].get("citation_count") or 0):
            seen[key] = paper

    deduped = list(seen.values())

    # Sort: citation_count desc, year desc
    deduped.sort(key=lambda p: (-(p.get("citation_count") or 0), -(p.get("year") or 0)))

    logger.info(
        f"[deduplicator] {len(combined)} input → {len(deduped)} after dedup/filter"
    )
    return {"all_papers": deduped}
