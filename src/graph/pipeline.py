from langgraph.graph import END, START, StateGraph

from src.agents.deduplicator import deduplicator_node
from src.agents.fetchers import arxiv_fetcher, semantic_fetcher
from src.agents.router import router_node
from src.agents.synthesizer import synthesizer_node
from src.graph.state import ResearchState


def build_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("router", router_node)
    builder.add_node("arxiv_fetcher", arxiv_fetcher)
    builder.add_node("semantic_fetcher", semantic_fetcher)
    builder.add_node("deduplicator", deduplicator_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "router")
    # parallel fan-out
    builder.add_edge("router", "arxiv_fetcher")
    builder.add_edge("router", "semantic_fetcher")
    # fan-in → deduplicator waits for both branches
    builder.add_edge("arxiv_fetcher", "deduplicator")
    builder.add_edge("semantic_fetcher", "deduplicator")
    builder.add_edge("deduplicator", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


graph = build_graph()
