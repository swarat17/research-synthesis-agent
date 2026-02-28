import os
import time

from openai import OpenAI
from pinecone import Pinecone

from src.utils.cost_tracker import cost_tracker
from src.utils.logger import logger


def _get_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX"])


def _embed_texts(texts: list[str]) -> tuple[list[list[float]], int]:
    """Returns (embeddings, total_tokens)."""
    client = OpenAI()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    embeddings = [item.embedding for item in response.data]
    total_tokens = response.usage.total_tokens
    return embeddings, total_tokens


def embed_and_upsert(papers: list[dict], query_id: str) -> int:
    """Embed papers and upsert into Pinecone. Returns number of vectors upserted."""
    if not papers:
        return 0

    texts = [f"{p['title']}. {p['abstract'][:500]}" for p in papers]

    t0 = time.time()
    embeddings, total_tokens = _embed_texts(texts)
    latency_ms = (time.time() - t0) * 1000

    cost_tracker.track_call(
        node_name="pinecone_embed",
        model="text-embedding-3-small",
        input_tokens=total_tokens,
        output_tokens=0,
        latency_ms=latency_ms,
    )

    vectors = [
        {
            "id": f"{query_id}-{i}",
            "values": embeddings[i],
            "metadata": {
                "title": papers[i].get("title", ""),
                "year": papers[i].get("year"),
                "source": papers[i].get("source", ""),
                "url": papers[i].get("url", ""),
                "abstract": papers[i].get("abstract", "")[:300],
            },
        }
        for i in range(len(papers))
    ]

    index = _get_index()
    index.upsert(vectors=vectors, namespace=query_id)

    logger.info(f"[pinecone_store] Upserted {len(vectors)} vectors (namespace={query_id})")
    return len(vectors)


def query_similar(query_text: str, query_id: str, top_k: int = 10) -> list[dict]:
    """Query Pinecone for similar papers. Returns list of metadata dicts."""
    embeddings, _ = _embed_texts([query_text])
    index = _get_index()
    result = index.query(
        vector=embeddings[0],
        top_k=top_k,
        include_metadata=True,
        namespace=query_id,
    )
    return [match.metadata for match in result.matches]
