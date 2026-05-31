"""biobrain.critic — the heuristic reward function.

Brain region: Limbic critic (orbitofrontal cortex / ventral striatum).
ML/RL term: Q-function / value function / intrinsic reward function.

The Critic evaluates state for "win-state aesthetics" — properties humans
recognize as the likely target of an ARC-AGI-3 puzzle without being told
the goal. All aesthetics unify under compression-progress (the
Chollet/Schmidhuber objective): a state is preferred if its predicate
representation is more compressible.

Aesthetic primitives (each implemented as a GoalExtractor):
    Compression       — fewer unique predicates / lower fact count
    Symmetry          — mirror-pair alignment of object-identity predicates
    Cohesion          — fewer connected components per color
    Noise             — fewer tiny isolated entities
    PatternRecurrence — paired regions with partial cell match (static)
    ChangeDynamics    — paired dynamic-canvas vs static-target regions

The Critic's output is a list of ProtoGoals — soft state-distance
functions the planner uses for action selection.

Currently wraps `biobrain.critic` directly. The next build phase adds the
remaining four aesthetic primitives (Compression/Symmetry/Cohesion/Noise)
as additional GoalExtractors registered with the L3 orchestrator.
"""

from __future__ import annotations

from biobrain.types import State
from biobrain.critic import (
    L3, ProtoGoal, GoalExtractor, TransitionHistory,
    StaticPatternRecurrence, ChangeDynamics, default_extractors,
)


class Critic:
    """Composes goal-extractors into a single value-function surface.

    Interface:
        observe_transition(transition) — update internal history
        evaluate(state) → list[ProtoGoal] — current goal stream
        state_value(state) → float — aggregate goal-distance (lower=better)
    """

    def __init__(self, extractors: list[GoalExtractor] | None = None) -> None:
        self._l3 = L3(extractors=extractors if extractors is not None
                       else default_extractors())
        self._history = TransitionHistory()

    def reset_game(self) -> None:
        self._history = TransitionHistory()

    def observe_transition(self, transition) -> None:
        self._history.update(transition)

    def evaluate(self, state: State) -> list[ProtoGoal]:
        return self._l3.detect(state, self._history)

    def state_value(self, state: State) -> float:
        """Aggregate value [0, 1]: weighted mean goal-distance.

        Lower = closer to goal-satisfaction = higher "win-state likelihood."
        Returns 0.5 (neutral) when no goals are detected.
        """
        goals = self.evaluate(state)
        if not goals:
            return 0.5
        total_w = sum(g.weight for g in goals) or 1.0
        return sum(g.weight * g.distance(state) for g in goals) / total_w


__all__ = ["Critic", "ProtoGoal", "GoalExtractor"]
