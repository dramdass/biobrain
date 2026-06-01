# tests/test_search_graph.py
from biobrain.planner.search_graph import SearchGraph, NodeMetadata


def test_empty_graph():
    g = SearchGraph()
    assert len(g) == 0
    assert g.has_node(123) is False
    assert g.child(123, ("noop",)) is None


def test_add_edge_creates_nodes():
    g = SearchGraph()
    g.add_edge(parent_hash=100, action_key=("click", 5, 0),
               child_hash=200, attempt_id=0)
    assert g.has_node(100)
    assert g.has_node(200)
    assert g.child(100, ("click", 5, 0)) == 200
    assert len(g) == 2


def test_add_edge_idempotent():
    g = SearchGraph()
    g.add_edge(parent_hash=100, action_key=("click", 5, 0),
               child_hash=200, attempt_id=0)
    g.add_edge(parent_hash=100, action_key=("click", 5, 0),
               child_hash=200, attempt_id=1)
    assert len(g) == 2
    assert g.node_metadata(100).visit_count == 2


def test_mark_terminal():
    g = SearchGraph()
    g.add_edge(100, ("noop",), 200, attempt_id=0)
    g.mark_terminal(200)
    assert g.node_metadata(200).is_terminal


def test_mark_scoring():
    g = SearchGraph()
    g.add_edge(100, ("click", 5, 0), 200, attempt_id=0)
    g.mark_scoring(200, attempt_id=0)
    assert g.node_metadata(200).is_scoring
    assert 200 in g.scoring_nodes()


def test_unexpanded_actions():
    g = SearchGraph()
    g.add_edge(100, ("key", 1, 0), 200, attempt_id=0)
    g.add_edge(100, ("key", 2, 0), 300, attempt_id=0)
    candidate_keys = [("key", 1, 0), ("key", 2, 0), ("key", 3, 0),
                      ("click", 5, 0)]
    unexpanded = g.unexpanded_actions(100, candidate_keys)
    assert ("key", 1, 0) not in unexpanded
    assert ("key", 2, 0) not in unexpanded
    assert ("key", 3, 0) in unexpanded
    assert ("click", 5, 0) in unexpanded


def test_path_from_root_traces_actions():
    g = SearchGraph()
    g.set_root(1)
    g.add_edge(1, ("key", 1, 0), 2, attempt_id=0)
    g.add_edge(2, ("key", 2, 0), 3, attempt_id=0)
    g.add_edge(3, ("click", 5, 0), 4, attempt_id=0)
    path = g.path_from_root(4)
    assert path == [("key", 1, 0), ("key", 2, 0), ("click", 5, 0)]


def test_path_from_root_missing_returns_none():
    g = SearchGraph()
    g.set_root(1)
    g.add_edge(99, ("noop",), 100, attempt_id=0)
    assert g.path_from_root(100) is None


def test_reachable_count_depth_1():
    g = SearchGraph()
    g.add_edge(1, ("a",), 2, attempt_id=0)
    g.add_edge(1, ("b",), 3, attempt_id=0)
    g.add_edge(1, ("c",), 2, attempt_id=0)  # same child via different action
    assert g.reachable_count(1, depth=1) == 2


def test_reachable_count_depth_2():
    g = SearchGraph()
    g.add_edge(1, ("a",), 2, attempt_id=0)
    g.add_edge(2, ("b",), 3, attempt_id=0)
    g.add_edge(2, ("c",), 4, attempt_id=0)
    # From node 1, depth 1 reaches {2}; depth 2 reaches {2, 3, 4}
    assert g.reachable_count(1, depth=2) == 3


def test_lru_eviction_at_cap():
    g = SearchGraph(max_nodes=3)
    g.add_edge(1, ("a",), 2, attempt_id=0)
    g.add_edge(2, ("b",), 3, attempt_id=0)
    g.add_edge(3, ("c",), 4, attempt_id=0)
    # 4 nodes (1, 2, 3, 4); cap is 3 — eviction should bring it down
    # After eviction, oldest (1) is gone
    g._evict_lru()
    assert len(g) == 3
    assert not g.has_node(1)


def test_reset_game_wipes():
    g = SearchGraph()
    g.add_edge(1, ("a",), 2, attempt_id=0)
    assert len(g) > 0
    g.reset_game()
    assert len(g) == 0
