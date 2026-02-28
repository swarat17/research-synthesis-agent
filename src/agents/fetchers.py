import arxiv

from src.graph.state import ResearchState
from src.utils.logger import logger


def arxiv_fetcher(state: ResearchState) -> dict:
    limit = max(1, state.get("max_papers", 10))
    query = state.get("query", "")

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=limit)
        papers = []
        for result in client.results(search):
            abstract = result.summary or ""
            if len(abstract) < 50:
                continue
            papers.append(
                {
                    "id": result.entry_id,
                    "title": result.title,
                    "abstract": abstract,
                    "authors": [str(a) for a in result.authors[:5]],
                    "year": result.published.year if result.published else None,
                    "url": result.entry_id,
                    "source": "arxiv",
                    "citation_count": 0,
                }
            )
        logger.info(f"[arxiv_fetcher] fetched {len(papers)} papers")
        return {"arxiv_papers": papers}

    except Exception as e:
        logger.warning(f"[arxiv_fetcher] Error: {e}")
        return {"arxiv_papers": [], "errors": [f"arxiv_fetcher: {str(e)}"]}
