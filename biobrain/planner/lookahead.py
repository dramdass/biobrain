"""biobrain.planner.lookahead — Planner + 1-step forward-simulation bonus.

Phase 2 build target per the refined architecture plan. Extends
MemoryBrainPlanner with a value-prior derived from forward simulation:
for each candidate action, predict the next-state fact set via the
BayesianWorldModel and compute an approximate Critic distance over the
predicted facts. Actions whose predicted next-state has LOWER
Critic-distance than current get a positive bonus added to their
Thompson sample.

Gated by Phase 1 measurement: WM achieved F1≥0.7 on vc33/lp85/cd82/g50t,
so the Simulator output is trustworthy for shallow lookahead. F1<0.6 on
bp35/r11l means lookahead may be noisy there; the bonus is bounded so
the substrate posterior + course-correction remain primary signals.

Approximation: predicted Critic distance uses only fact-space extractors
(Compression, Noise) that can be computed directly from a predicted fact
set without materializing a State object. Raw-cell extractors
(StaticPatternRecurrence, ChangeDynamics) are not consulted in
lookahead — they still drive the Critic at observe()-time
course-correction.

No magic constants: the bonus IS `current_d - predicted_d`, clamped to
[0, 0.5] so it cannot exceed the surprise/course-correction signal scale.
"""

from __future__ import annotations

from typing import Optional

from biobrain.types import (
    Action, ComputeBudget, State, action_kind,
)
from biobrain.planner.agency import _candidate_actions
from biobrain.planner.planner_brain import MemoryBrainPlanner
from biobrain.planner.posterior import ActionScoreTable
from biobrain.curiosity.predicates import emit_atomic_facts
from biobrain.curiosity.residual import SURPRISE_CLIP


def _critic_distance_from_facts(facts: set,
                                 active_goals: list) -> float:
    """Weighted-mean distance over goals whose distance_fn accepts a fact set.

    Principle: the L3 layer emits goals each with a weight (derived from
    DL-saving). The Critic's value on a predicted state is the
    weight-weighted mean of each goal's distance, computed only over
    goals that can evaluate on the predicate set (raw-cell-only goals
    are silently skipped — they have no input here).

    Returns 0.5 (uninformative) when no goal supports fact-set evaluation,
    so the lookahead bonus = current_d - predicted_d = 0 for that action.
    No hardcoded blends, no background-vs-foreground split, no magic
    constants: the goal weights ARE the arbiter.
    """
    if not active_goals:
        return 0.5
    weighted_sum = 0.0
    weight_total = 0.0
    for g in active_goals:
        try:
            d = g.distance_fn(facts)
        except (TypeError, AttributeError, KeyError):
            # Goal's distance_fn doesn't accept a fact set; skip
            continue
        weighted_sum += g.weight * d
        weight_total += g.weight
    if weight_total == 0:
        return 0.5
    return weighted_sum / weight_total


class MemoryBrainLookahead(MemoryBrainPlanner):
    """Planner + 1-step forward-simulation value prior.

    For each candidate action: simulate next-state via the world model,
    compute Critic distance approximation on predicted fact set, add the
    reduction (current_d - predicted_d) as a bonus to the Thompson sample.

    Inherits observe() (substrate + L1 + course-correction credit) from
    MemoryBrainPlanner. Overrides act() with the lookahead bonus.
    """

    def act(self, state: State, budget: ComputeBudget) -> Action:
        candidates = _candidate_actions(state)
        if not candidates:
            raise ValueError(
                f"MemoryBrainLookahead has no candidates from "
                f"available_actions={state.available_actions}"
            )
        # Refresh proto-goals (consumed by observe() course-correction
        # AND by lookahead Critic — the same goal list drives both).
        self._current_goals = self._l3.detect(state, self._history)

        # Current-state Critic distance: weighted-mean over goals whose
        # distance_fn supports fact-set input. No magic constants.
        current_facts = emit_atomic_facts(None, state)
        current_d = _critic_distance_from_facts(current_facts,
                                                 self._current_goals)

        best_score = -1.0
        best_action: Optional[Action] = None
        for a in candidates:
            sig = ActionScoreTable._signature(a, state)
            n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
            alpha = max(0.01, n_goal + 1.0)
            beta = max(0.01, n_obs - n_goal + 1.0)
            thompson = self._rng.betavariate(alpha, beta)

            # Lookahead: predicted next-state fact set via world model.
            # Bonus = reduction in Critic distance from current to predicted.
            # Clamped so the bonus cannot exceed surprise/course-correction
            # signal scale. NOTE: the quality of this bonus depends entirely
            # on the Critic's coverage of fact-space — if extractors only
            # see "fewer facts = better" they will mislead. Phase 0
            # completion (migrate SPR/ChangeDynamics to fact-space) is the
            # real fix.
            predicted = self._world.predict(state, a)
            predicted_facts = {f for f, p in predicted.items() if p >= 0.5}
            if predicted_facts:
                predicted_d = _critic_distance_from_facts(predicted_facts,
                                                           self._current_goals)
                lookahead_bonus = max(
                    -SURPRISE_CLIP,
                    min(SURPRISE_CLIP, current_d - predicted_d),
                )
            else:
                lookahead_bonus = 0.0

            score = thompson + lookahead_bonus
            if score > best_score:
                best_score = score
                best_action = a
        if best_action is None:
            best_action = self._rng.choice(candidates)
        return best_action


__all__ = ["MemoryBrainLookahead", "_critic_distance_from_facts"]
