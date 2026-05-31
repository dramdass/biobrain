"""biobrain.critic.symmetry — Symmetry / mirror-pair extractor.

Aesthetic primitive: states with mirror symmetry (left↔right or top↔bottom)
compress better — one half encodes the other. The extractor uses the
joint predicate `entity_color_quadrant(c, q)` (added in Phase 0) to pair
quadrants by content for mirror checks.

Spatial layout (4×4 quadrants, indices 0..15):
    Q0  Q1  Q2  Q3
    Q4  Q5  Q6  Q7
    Q8  Q9  Q10 Q11
    Q12 Q13 Q14 Q15

Mirror pairings:
    Left↔Right (mirror_x):
        0↔3, 1↔2, 4↔7, 5↔6, 8↔11, 9↔10, 12↔15, 13↔14
    Top↔Bottom (mirror_y):
        0↔12, 1↔13, 2↔14, 3↔15, 4↔8, 5↔9, 6↔10, 7↔11

Distance metric: fraction of quadrant pairs whose color sets disagree.
Lower = more symmetric.
"""

from __future__ import annotations

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory
from biobrain.curiosity.predicates import emit_atomic_facts


MIRROR_X_PAIRS = [
    (0, 3), (1, 2), (4, 7), (5, 6),
    (8, 11), (9, 10), (12, 15), (13, 14),
]
MIRROR_Y_PAIRS = [
    (0, 12), (1, 13), (2, 14), (3, 15),
    (4, 8), (5, 9), (6, 10), (7, 11),
]


def _color_quadrant_map(facts: set) -> dict[int, set[int]]:
    """Build {quadrant → set of colors present in that quadrant}."""
    m: dict[int, set[int]] = {}
    for f in facts:
        if isinstance(f, tuple) and len(f) == 3 and f[0] == "entity_color_quadrant":
            c, q = int(f[1]), int(f[2])
            m.setdefault(q, set()).add(c)
    return m


def _symmetry_distance(state_or_facts, axis_pairs: list) -> float:
    """Fraction of quadrant pairs whose color sets disagree.

    Accepts either a State (emits facts internally) or a fact set
    (for lookahead's predicted-state evaluation).
    """
    if isinstance(state_or_facts, (set, frozenset)):
        facts = state_or_facts
    else:
        facts = emit_atomic_facts(None, state_or_facts)
    by_quad = _color_quadrant_map(facts)
    # Only count pairs where at least one side has content
    active_pairs = [
        (a, b) for a, b in axis_pairs
        if by_quad.get(a) or by_quad.get(b)
    ]
    if not active_pairs:
        return 0.0
    n_disagree = sum(
        1 for a, b in active_pairs
        if by_quad.get(a, set()) != by_quad.get(b, set())
    )
    return n_disagree / len(active_pairs)


class Symmetry:
    """Goal: align mirror-paired quadrants by entity-color content.

    Emits up to two goals: mirror_x (left↔right) and mirror_y (top↔bottom).
    Only emitted when there's enough content to be meaningful (≥2 colored
    quadrants on at least one mirror axis).
    """
    name = "symmetry"

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        if not state.entities or len(state.entities) < 2:
            return []
        facts = emit_atomic_facts(None, state)
        by_quad = _color_quadrant_map(facts)
        n_active = sum(1 for s in by_quad.values() if s)
        if n_active < 2:
            return []

        goals: list[ProtoGoal] = []
        for name, pairs in (("mirror_x", MIRROR_X_PAIRS),
                            ("mirror_y", MIRROR_Y_PAIRS)):
            d = _symmetry_distance(state, pairs)
            if d < 0.05 or d > 0.99:
                # Already nearly-symmetric or entirely asymmetric → no
                # actionable gradient
                continue

            def distance_fn(s, pairs=pairs):
                return _symmetry_distance(s, pairs)

            goals.append(ProtoGoal(
                goal_id=f"symmetry:{name}",
                description=f"symmetry: align {name} quadrants by color",
                distance_fn=distance_fn,
                weight=d,  # bigger asymmetry = more DL to save
                source=self.name,
            ))
        return goals
