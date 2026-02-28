from src.graph.pipeline import build_graph, graph

EXPECTED_NODES = {
    "router",
    "arxiv_fetcher",
    "deduplicator",
    "synthesizer",
    "contradiction_detector",
    "hypothesis_generator",
    "cost_auditor",
}


def test_graph_has_exactly_seven_nodes():
    drawable = graph.get_graph()
    user_nodes = {n for n in drawable.nodes if not n.startswith("__")}
    assert len(user_nodes) == 7, f"Expected 7 nodes, got {len(user_nodes)}: {user_nodes}"


def test_all_node_names_correct():
    drawable = graph.get_graph()
    user_nodes = {n for n in drawable.nodes if not n.startswith("__")}
    assert user_nodes == EXPECTED_NODES, (
        f"Missing: {EXPECTED_NODES - user_nodes}  |  Extra: {user_nodes - EXPECTED_NODES}"
    )


def test_graph_compiles_without_error():
    g = build_graph()
    assert g is not None
