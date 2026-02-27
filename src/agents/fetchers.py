import arxiv
from semanticscholar import SemanticScholar

from src.graph.state import ResearchState
from src.utils.logger import logger

_REQUIRED_FIELDS = [
    "title", "abstract", "authors", "year",
    "citationCount", "url", "externalIds",
]


def arxiv_fetcher(state: ResearchState) -> dict:
    if state.get("routing_decision") == "semantic_only":
        return {"arxiv_papers": []}

    limit = max(1, state.get("max_papers", 10) // 2)
    query = state.get("query", "")

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=limit)
        papers = []
        for result in client.results(search):
            abstract = result.summary or ""
            if len(abstract) < 50:
                continue
            papers.append({
                "id": result.entry_id,
                "title": result.title,
                "abstract": abstract,
                "authors": [str(a) for a in result.authors[:5]],
                "year": result.published.year if result.published else None,
                "url": result.entry_id,
                "source": "arxiv",
                "citation_count": 0,
            })
        logger.info(f"[arxiv_fetcher] fetched {len(papers)} papers")
        return {"arxiv_papers": papers}

    except Exception as e:
        logger.warning(f"[arxiv_fetcher] Error: {e}")
        return {"arxiv_papers": [], "errors": [f"arxiv_fetcher: {str(e)}"]}


def semantic_fetcher(state: ResearchState) -> dict:
    if state.get("routing_decision") == "arxiv_only":
        return {"semantic_papers": []}

    limit = max(1, state.get("max_papers", 10) // 2)
    query = state.get("query", "")

    try:
        sch = SemanticScholar()
        results = sch.search_paper(query, limit=limit, fields=_REQUIRED_FIELDS)
        papers = []
        for paper in results:
            abstract = paper.abstract or ""
            if len(abstract) < 50:
                continue
            papers.append({
                "id": paper.paperId or "",
                "title": paper.title or "",
                "abstract": abstract,
                "authors": [a["name"] for a in (paper.authors or [])[:5]],
                "year": paper.year,
                "url": (
                    paper.url
                    or f"https://www.semanticscholar.org/paper/{paper.paperId}"
                ),
                "source": "semantic_scholar",
                "citation_count": paper.citationCount or 0,
            })
        logger.info(f"[semantic_fetcher] fetched {len(papers)} papers")
        return {"semantic_papers": papers}

    except Exception as e:
        logger.warning(f"[semantic_fetcher] Error: {e}")
        return {"semantic_papers": [], "errors": [f"semantic_fetcher: {str(e)}"]}
