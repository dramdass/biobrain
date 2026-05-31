"""biobrain.critic.base — types and protocols for L3 goal extraction.

The L3 layer (Goal-setting capability, third of the four ARC-AGI-3
capabilities) is built as a collection of pluggable GoalExtractors,
each consuming one signal source and emitting ProtoGoals.

A ProtoGoal is a state-distance function the action policy can take
ΔDL with respect to. All extractors output the same ProtoGoal type
regardless of how the goal was derived; the action layer is agnostic
to which extractor produced which goal.

The unifying abstraction: every proto-goal is a discrepancy between
features_current and features_preferred. Different extractors choose
different feature_extractors and preferred_state_operators, but they
all produce ProtoGoals that share the same downstream interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from biobrain.types import State, Transition


# ---------------------------------------------------------------------------
# ProtoGoal — the universal output type
# ---------------------------------------------------------------------------

@dataclass
class ProtoGoal:
    """A soft state-distance function derivable from compression structure.

    `distance(state) ∈ [0, 1]`: 0 = goal satisfied, 1 = furthest possible.
    `weight ∈ [0, 1]`: how strongly this proto-goal contributes; derived
        from the compression progress satisfying the goal would yield.
    `relevant_cells`: (row, col) cells where the gradient lives. Used by
        the action layer to bias candidate selection and identify
        gradient-bearing actions.
    `source`: name of the extractor that produced this goal. For
        diagnostics and per-extractor ablation.
    `region_a_bbox`, `region_b_bbox`: when a goal pairs two regions,
        the two bboxes are stored directly (avoids lossy reconstruction
        from relevant_cells). Empty (0,0,0,0) when not applicable.
    """
    goal_id: str
    description: str
    distance_fn: Callable[[State], float]
    weight: float
    relevant_cells: frozenset = field(default_factory=frozenset)
    source: str = "unknown"
    region_a_bbox: tuple = (0, 0, 0, 0)
    region_b_bbox: tuple = (0, 0, 0, 0)

    def distance(self, state: State) -> float:
        return self.distance_fn(state)


# ---------------------------------------------------------------------------
# TransitionHistory — running summary of observed transitions
# ---------------------------------------------------------------------------

class TransitionHistory:
    """Per-game running summary of observed transitions.

    Tracks per-cell change frequency so extractors can classify cells
    as STATIC (rarely change) vs DYNAMIC (frequently change). This is
    the substrate for the ChangeDynamics extractor.

    Brains call `update(transition)` on every observe(). Extractors
    call `change_rate(row, col)` to query.

    Resets per-game (changes between games are not comparable).
    """

    def __init__(self, grid_dim: int = 64) -> None:
        self._grid_dim = grid_dim
        # Per-cell counters
        self._change_count = np.zeros((grid_dim, grid_dim), dtype=np.int32)
        self._n_transitions = 0

    def reset(self) -> None:
        self._change_count.fill(0)
        self._n_transitions = 0

    def update(self, transition: Transition) -> None:
        if transition.before is None or transition.after is None:
            return
        before = transition.before.raw_grid
        after = transition.after.raw_grid
        if before is None or after is None:
            return
        before_arr = np.asarray(before)
        after_arr = np.asarray(after)
        if before_arr.shape != after_arr.shape:
            return
        if before_arr.shape != self._change_count.shape:
            # Resize (rare; safer than failing)
            self._change_count = np.zeros(before_arr.shape, dtype=np.int32)
            self._n_transitions = 0
        diff = (before_arr != after_arr).astype(np.int32)
        self._change_count += diff
        self._n_transitions += 1

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    def change_rate_grid(self) -> np.ndarray:
        """Per-cell change rate ∈ [0, 1]. Shape = grid.

        Returns all-zeros until at least one transition is recorded.
        """
        if self._n_transitions == 0:
            return np.zeros_like(self._change_count, dtype=np.float32)
        return self._change_count.astype(np.float32) / self._n_transitions


# ---------------------------------------------------------------------------
# GoalExtractor protocol
# ---------------------------------------------------------------------------

class GoalExtractor(Protocol):
    """Pluggable goal-extraction module.

    Each extractor consumes one signal source (static structure, change
    dynamics, predictability, etc.) and emits ProtoGoals. The L3
    orchestrator runs all registered extractors and combines outputs.

    Implementations MUST be deterministic in (state, history) — no
    internal randomness — so the brain's seed schedule is reproducible.
    """
    name: str

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        ...


def state_distance_to_goals(state, goals: list) -> float:
    """Weighted-average distance over goals. [0, 1]. 0.5 when no goals.

    Accepts a State (used by brains for current-state distance) or a
    fact set (used by simulator for predicted-state distance). Each goal's
    distance_fn handles both inputs.
    """
    if not goals:
        return 0.5
    total_w = sum(g.weight for g in goals) or 1.0
    return sum(g.weight * g.distance_fn(state) for g in goals) / total_w
