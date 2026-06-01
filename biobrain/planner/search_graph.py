"""biobrain.planner.search_graph — within-game reachable-state graph.

Nodes are grid_hash values; edges are (action_key)-labeled transitions.
The graph accumulates across attempts within a game, providing:
  - Frontier expansion priority (for curiosity-guided search)
  - Empowerment computation over the REAL reachable graph
  - Replay of action sequences from root to discovered scoring nodes

Lifecycle: wipes on reset_game; persists across reset_attempt.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional


# RL-TODO: derive memory bound from per-game state-space estimates.
# 10,000 nodes covers any plausible reachable graph for ARC-AGI-3 games.
DEFAULT_MAX_NODES = 10_000


@dataclass
class NodeMetadata:
    """Per-node bookkeeping."""
    grid_hash: int
    visit_count: int = 0
    first_attempt: int = 0
    last_attempt: int = 0
    is_terminal: bool = False
    is_scoring: bool = False


class SearchGraph:
    """Within-game reachable-state graph.

    nodes: grid_hash -> NodeMetadata (OrderedDict for LRU ordering)
    edges: (parent_hash, action_key) -> child_hash
    parents: child_hash -> set of (parent_hash, action_key) (for path tracing)

    Persists across reset_attempt; wipes on reset_game.
    """

    def __init__(self, max_nodes: int = DEFAULT_MAX_NODES) -> None:
        self._nodes: OrderedDict[int, NodeMetadata] = OrderedDict()
        self._edges: dict[tuple[int, Any], int] = {}
        self._parents: dict[int, set[tuple[int, Any]]] = {}
        self._root: Optional[int] = None
        self._scoring: set[int] = set()
        self.max_nodes = max_nodes

    # ----------------------------------------------------------- lifecycle

    def reset_game(self) -> None:
        self._nodes = OrderedDict()
        self._edges = {}
        self._parents = {}
        self._root = None
        self._scoring = set()

    def set_root(self, grid_hash: int) -> None:
        """Mark a node as root (used for path_from_root)."""
        self._touch_node(grid_hash, attempt_id=0)
        self._root = grid_hash

    # ----------------------------------------------------------- mutation

    def add_edge(self, parent_hash: int, action_key: Any,
                 child_hash: int, attempt_id: int) -> None:
        """Add an edge from parent through action_key to child.

        If the edge already exists, increments visit counts (idempotent
        for graph structure).
        """
        self._touch_node(parent_hash, attempt_id)
        self._touch_node(child_hash, attempt_id)
        if self._root is None:
            self._root = parent_hash
        edge_key = (parent_hash, action_key)
        self._edges[edge_key] = child_hash
        self._parents.setdefault(child_hash, set()).add(edge_key)
        if len(self._nodes) > self.max_nodes:
            self._evict_lru()

    def mark_terminal(self, grid_hash: int) -> None:
        if grid_hash in self._nodes:
            self._nodes[grid_hash].is_terminal = True

    def mark_scoring(self, grid_hash: int, attempt_id: int) -> None:
        if grid_hash in self._nodes:
            self._nodes[grid_hash].is_scoring = True
            self._scoring.add(grid_hash)

    # ----------------------------------------------------------- queries

    def has_node(self, grid_hash: int) -> bool:
        return grid_hash in self._nodes

    def node_metadata(self, grid_hash: int) -> Optional[NodeMetadata]:
        return self._nodes.get(grid_hash)

    def child(self, parent_hash: int, action_key: Any) -> Optional[int]:
        return self._edges.get((parent_hash, action_key))

    def unexpanded_actions(self, parent_hash: int,
                            candidates: Iterable) -> list:
        """Among candidate actions, which have NOT yet been tried from
        parent_hash? Returned in input order.
        """
        out = []
        for a in candidates:
            if (parent_hash, a) not in self._edges:
                out.append(a)
        return out

    def scoring_nodes(self) -> set[int]:
        return set(self._scoring)

    def path_from_root(self, target_hash: int) -> Optional[list]:
        """Return action sequence from root to target_hash, or None if not
        reachable. BFS over parents — picks the shortest path.
        """
        if self._root is None or target_hash not in self._nodes:
            return None
        if target_hash == self._root:
            return []
        visited = {target_hash}
        queue = deque([(target_hash, [])])
        while queue:
            node, action_seq = queue.popleft()
            for (parent, action_key) in self._parents.get(node, set()):
                if parent in visited:
                    continue
                new_seq = [action_key] + action_seq
                if parent == self._root:
                    return new_seq
                visited.add(parent)
                queue.append((parent, new_seq))
        return None

    def reachable_count(self, source_hash: int, depth: int) -> int:
        """BFS from source up to `depth` hops; return |distinct nodes
        reached| (excluding source itself).
        """
        if source_hash not in self._nodes or depth <= 0:
            return 0
        seen = {source_hash}
        frontier = {source_hash}
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                # Find all children of `node` via edges
                for (parent, _action), child in self._edges.items():
                    if parent != node:
                        continue
                    if child not in seen:
                        seen.add(child)
                        new_frontier.add(child)
            frontier = new_frontier
            if not frontier:
                break
        return len(seen) - 1  # exclude source

    # ----------------------------------------------------------- internals

    def _touch_node(self, grid_hash: int, attempt_id: int) -> None:
        if grid_hash in self._nodes:
            m = self._nodes[grid_hash]
            m.visit_count += 1
            m.last_attempt = attempt_id
            self._nodes.move_to_end(grid_hash)  # LRU touch
        else:
            self._nodes[grid_hash] = NodeMetadata(
                grid_hash=grid_hash,
                visit_count=1,
                first_attempt=attempt_id,
                last_attempt=attempt_id,
            )

    def _evict_lru(self) -> None:
        """Drop the least-recently-touched node and its incident edges.

        No-op if the graph is already within the size cap; this lets the
        method serve both as the auto-trigger inside add_edge and as an
        explicit hook callable by tests/clients without double-evicting.
        """
        if not self._nodes or len(self._nodes) <= self.max_nodes:
            return
        oldest_hash, _ = next(iter(self._nodes.items()))
        # Remove edges where oldest is parent
        to_remove = [k for k in self._edges if k[0] == oldest_hash]
        for k in to_remove:
            child = self._edges.pop(k)
            if child in self._parents:
                self._parents[child].discard(k)
        # Remove edges where oldest is child
        for child, parents in list(self._parents.items()):
            self._parents[child] = {
                (p, a) for (p, a) in parents if p != oldest_hash
            }
        self._parents.pop(oldest_hash, None)
        self._nodes.pop(oldest_hash, None)
        self._scoring.discard(oldest_hash)

    def __len__(self) -> int:
        return len(self._nodes)


__all__ = ["SearchGraph", "NodeMetadata", "DEFAULT_MAX_NODES"]
