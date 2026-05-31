"""biobrain.critic.change_dynamics_facts — fact-space ChangeDynamics extractor.

Phase 0 completion (alignment): the original ChangeDynamics operates on
raw grid cells, which means the lookahead simulator can't consume its
distance because the simulator outputs predicted facts, not predicted
grids. This extractor reformulates the same canvas-vs-target principle
entirely in fact-space, using the entity_color_quadrant joint predicate.

Mechanism:
  1. From TransitionHistory: count per-quadrant change rate. Use the
     SAME per-cell change rates already tracked, aggregated by quadrant
     (i.e., per-quadrant total cells that changed / total transitions).
  2. Classify quadrants:
       DYNAMIC if quadrant's per-transition cell-change count > threshold
       STATIC  if it's below the static threshold and has content
  3. For each (DYNAMIC quadrant Q_d, STATIC quadrant Q_s) pair:
       - Goal distance = symmetric difference of their entity-color sets,
         normalized to [0, 1].
       - Goal-satisfied when DYNAMIC Q_d's color set matches STATIC Q_s's.
  4. Critically: distance_fn operates on a FACT SET (the state's emitted
     facts), so the lookahead simulator can evaluate it on predicted facts.

This is the cd82-shaped goal in fact-space: cc_1 (the active shape area,
DYNAMIC) should come to resemble cc_0 (the target pattern, STATIC). Both
are now represented by their entity_color_quadrant joints.
"""

from __future__ import annotations

import numpy as np

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory
from biobrain.curiosity.predicates import emit_atomic_facts


MIN_TRANSITIONS = 5
DYNAMIC_THRESHOLD_PER_QUADRANT = 0.5  # cells per quadrant per transition
STATIC_THRESHOLD_PER_QUADRANT = 0.05  # very-low change rate


def _quadrant_change_rate(history: TransitionHistory) -> dict[int, float]:
    """Per-quadrant: cells that changed / N transitions / cells per quadrant."""
    if history.n_transitions == 0:
        return {}
    rate = history.change_rate_grid()
    if rate.size == 0:
        return {}
    h, w = rate.shape
    qh = h // 4
    qw = w // 4
    out: dict[int, float] = {}
    for qy in range(4):
        for qx in range(4):
            q = qy * 4 + qx
            r0 = qy * qh
            r1 = (qy + 1) * qh if qy < 3 else h
            c0 = qx * qw
            c1 = (qx + 1) * qw if qx < 3 else w
            # Mean per-cell change rate within the quadrant
            out[q] = float(rate[r0:r1, c0:c1].mean())
    return out


def _quadrant_to_colors(facts: set) -> dict[int, set[int]]:
    """Build {quadrant → set of entity-colors present in that quadrant}."""
    m: dict[int, set[int]] = {}
    for f in facts:
        if isinstance(f, tuple) and len(f) == 3 and f[0] == "entity_color_quadrant":
            c, q = int(f[1]), int(f[2])
            m.setdefault(q, set()).add(c)
    return m


def _color_set_distance(a: set, b: set) -> float:
    """Symmetric difference / union ∈ [0, 1]. Jaccard-style."""
    if not a and not b:
        return 0.0
    sym_diff = a.symmetric_difference(b)
    union = a | b
    return len(sym_diff) / len(union)


class ChangeDynamicsFactSpace:
    """L3-Change-FactSpace extractor.

    Pairs dynamic quadrants with static quadrants by their entity-color
    set distance. Critically, distance_fn consumes a fact set — so the
    lookahead simulator can compute it on predicted facts without
    materializing a State.
    """
    name = "change_dynamics_facts"

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        if history.n_transitions < MIN_TRANSITIONS:
            return []
        rates = _quadrant_change_rate(history)
        if not rates:
            return []
        # Classify quadrants
        dyn_quads = [q for q, r in rates.items() if r >= DYNAMIC_THRESHOLD_PER_QUADRANT]
        stat_quads = [q for q, r in rates.items() if r < STATIC_THRESHOLD_PER_QUADRANT]
        if not dyn_quads or not stat_quads:
            return []

        # Current entity-color sets per quadrant from facts
        facts = emit_atomic_facts(None, state)
        by_quad = _quadrant_to_colors(facts)
        # Only emit goals for quadrants that have SOME content (else
        # the pairing is degenerate)
        active_dyn = [q for q in dyn_quads if by_quad.get(q)]
        active_stat = [q for q in stat_quads if by_quad.get(q)]
        if not active_dyn or not active_stat:
            return []

        goals: list[ProtoGoal] = []
        for q_d in active_dyn:
            d_colors = by_quad.get(q_d, set())
            # Pair dynamic q_d with each static quadrant by color-set
            # similarity (closer = more promising goal pair)
            scored: list[tuple[float, int, set]] = []
            for q_s in active_stat:
                s_colors = by_quad.get(q_s, set())
                dist = _color_set_distance(d_colors, s_colors)
                scored.append((dist, q_s, s_colors))
            scored.sort()
            # Take top-2 best matches per dynamic quadrant
            for dist, q_s, s_colors_at_detect in scored[:2]:
                # Goal-satisfied when dynamic quadrant's content = static's
                # at detection time (s_colors_at_detect is captured in
                # closure — STATIC quadrant doesn't change much by
                # construction)

                def distance_fn(s_or_facts,
                                q_d=q_d, target=s_colors_at_detect):
                    # Accept either a State or a raw fact set (for lookahead)
                    if isinstance(s_or_facts, set):
                        f = s_or_facts
                    else:
                        if not hasattr(s_or_facts, "raw_grid"):
                            return 1.0
                        f = emit_atomic_facts(None, s_or_facts)
                    bq = _quadrant_to_colors(f)
                    current = bq.get(q_d, set())
                    return _color_set_distance(current, target)

                weight = max(0.1, 1.0 - dist)  # closer initial match → higher weight
                goals.append(ProtoGoal(
                    goal_id=f"change_facts:dyn_{q_d}_to_stat_{q_s}",
                    description=(
                        f"change-facts: dyn quad {q_d} colors "
                        f"→ match stat quad {q_s} colors "
                        f"(initial dist {dist:.2f})"
                    ),
                    distance_fn=distance_fn,
                    weight=weight,
                    source=self.name,
                ))
        return goals


__all__ = ["ChangeDynamicsFactSpace"]
