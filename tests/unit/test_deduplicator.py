from src.agents.deduplicator import deduplicator_node


def _paper(title, source="arxiv", citation_count=0, year=2023, abstract=None):
    return {
        "id": f"id-{title[:8]}",
        "title": title,
        "abstract": abstract or ("A" * 60),
        "authors": ["Author One"],
        "year": year,
        "url": "https://example.com",
        "source": source,
        "citation_count": citation_count,
    }


def _run(arxiv_papers=None):
    state = {"arxiv_papers": arxiv_papers or []}
    return deduplicator_node(state)["all_papers"]


def test_deduplicates_same_title_different_id():
    papers = _run(arxiv_papers=[
        _paper("Deep Learning Survey"),
        _paper("Deep Learning Survey"),
    ])
    assert len(papers) == 1


def test_deduplicates_with_punctuation_differences():
    papers = _run(arxiv_papers=[
        _paper("BERT: Pre-Training of Deep Bidirectional Transformers"),
        _paper("BERT Pre Training of Deep Bidirectional Transformers"),
    ])
    assert len(papers) == 1


def test_keeps_higher_citation_count_on_dedup():
    papers = _run(arxiv_papers=[
        _paper("Attention Is All You Need", citation_count=5),
        _paper("Attention Is All You Need", citation_count=100),
    ])
    assert len(papers) == 1
    assert papers[0]["citation_count"] == 100


def test_all_distinct_papers_kept():
    titles = ["Paper Alpha", "Paper Beta", "Paper Gamma", "Paper Delta", "Paper Epsilon"]
    papers = _run(arxiv_papers=[_paper(t) for t in titles])
    assert len(papers) == 5


def test_filters_short_abstracts():
    papers = _run(arxiv_papers=[
        _paper("Short Abstract Paper", abstract="Too short."),
        _paper("Long Abstract Paper", abstract="B" * 60),
    ])
    assert len(papers) == 1
    assert papers[0]["title"] == "Long Abstract Paper"


def test_sorted_by_citation_descending():
    papers = _run(arxiv_papers=[
        _paper("Paper Low", citation_count=10),
        _paper("Paper High", citation_count=500),
        _paper("Paper Mid", citation_count=50),
    ])
    counts = [p["citation_count"] for p in papers]
    assert counts == sorted(counts, reverse=True)
