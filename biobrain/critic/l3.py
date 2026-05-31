"""biobrain.critic.l3 — L3 orchestrator.

Runs registered GoalExtractors and combines their outputs into a single
ProtoGoal list. The brain consumes the combined list; it's agnostic to
which extractor produced which goal.

Combining strategy:
  - Union all extractors' outputs.
  - Dedupe by (source, region_a_bbox, region_b_bbox).
  - Sort by weight (descending) so the highest-DL-saving goals dominate
    downstream consumers that take the top-K.

Per the user's spec: compression (DL saving) IS the arbiter when
extractors disagree. The weight field on each ProtoGoal IS the bid.
"""

from __future__ import annotations

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory
from biobrain.critic.pattern_recurrence import StaticPatternRecurrence
from biobrain.critic.change_dynamics import ChangeDynamics
from biobrain.critic.compression import Compression
from biobrain.critic.noise import Noise
from biobrain.critic.symmetry import Symmetry
from biobrain.critic.change_dynamics_facts import ChangeDynamicsFactSpace


def default_extractors() -> list[GoalExtractor]:
    """The set of extractors used when none is explicitly passed.

    Order is arbitrary (L3 sorts by weight at output time); listed here
    in the order they were introduced.
    """
    return [
        StaticPatternRecurrence(),
        ChangeDynamics(),
        ChangeDynamicsFactSpace(),
        Compression(),
        Noise(),
        Symmetry(),
    ]


class L3:
    """Run all registered extractors; return combined proto-goal list."""

    def __init__(self,
                 extractors: list[GoalExtractor] | None = None) -> None:
        self._extractors = extractors if extractors is not None \
            else default_extractors()

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        goals: list[ProtoGoal] = []
        seen: set = set()
        for ext in self._extractors:
            try:
                produced = ext.detect(state, history)
            except Exception:
                continue
            for g in produced:
                key = (g.source, g.region_a_bbox, g.region_b_bbox)
                if key in seen:
                    continue
                seen.add(key)
                goals.append(g)
        # Sort by weight, descending — highest-DL-saving first
        goals.sort(key=lambda g: -g.weight)
        return goals

    @property
    def extractors(self) -> list[GoalExtractor]:
        return list(self._extractors)
