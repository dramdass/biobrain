"""biobrain.planner.ledger_planner — Lookahead + Ledger (scientific method).

Phase 3 build. Extends MemoryBrainLookahead with the Ledger working
memory: trajectory abstraction on score events, cross-level program
promotion, hierarchical Beta per (program, level).

Mechanics at act():
  1. If a Ledger-promoted program is mid-execution (continuation in flight),
     step it (just like MemoryBrainDream).
  2. Else, on entry to a new level, pull promoted programs from the Ledger
     and pick the highest-confidence one as the current candidate. Try it.
  3. Else, fall back to MemoryBrainLookahead's standard policy
     (substrate Thompson + lookahead bonus).

Failure tracking: when a promoted program completes its sequence without
producing a score event, register_failure on it so future trials at this
level reflect the negative evidence.

The Ledger object is wiped on reset_game (House model: inter-game amnesia)
but persists across reset_attempt (intra-game memory).
"""

from __future__ import annotations

from typing import Optional

from biobrain.types import (
    Action, ComputeBudget, State, Transition,
    EVENT_LEVEL_INCREASED, EVENT_SCORE_INCREASED,
)
from biobrain.planner.agency import _candidate_actions
from biobrain.motor_cortex.brain import _resolve_sig_to_action
from biobrain.motor_cortex.core import Program
from biobrain.planner.lookahead import MemoryBrainLookahead
from biobrain.planner.posterior import ActionScoreTable
from biobrain.curiosity.predicates import emit_atomic_facts
from biobrain.curiosity.residual import SURPRISE_CLIP
from biobrain.planner.lookahead import _critic_distance_from_facts
from biobrain.ledger.ledger import Ledger


class MemoryBrainLedger(MemoryBrainLookahead):
    """Lookahead planner + Ledger-promoted DSL programs.

    Inherits observe() from MemoryBrainPlanner (substrate + L1 +
    course-correction credit). Adds Ledger updating at observe() and
    Ledger-promoted execution at act().
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._ledger = Ledger()
        # Track in-flight Ledger program (continuation + id)
        self._current_program: Optional[Program] = None
        self._current_program_id: Optional[str] = None
        self._current_program_score_baseline: int = 0
        # Track last-seen level so we can detect level transitions
        # (to fetch fresh promoted programs)
        self._last_level: int = -1
        self._scores_seen: int = 0

    def reset_game(self, game_id: str) -> None:
        super().reset_game(game_id)
        self._ledger.reset_game()
        self._current_program = None
        self._current_program_id = None
        self._current_program_score_baseline = 0
        self._last_level = -1
        self._scores_seen = 0

    def reset_attempt(self) -> None:
        super().reset_attempt()
        # Abandon any in-flight program at attempt boundary
        self._current_program = None
        self._current_program_id = None
        self._last_level = -1

    def observe(self, transition: Transition) -> None:
        super().observe(transition)
        self._ledger.observe(transition)
        # Count score events for failure detection on Ledger-promoted runs
        for e in transition.events:
            if e.kind in (EVENT_SCORE_INCREASED, EVENT_LEVEL_INCREASED):
                self._scores_seen += 1
                break

    def _maybe_start_promoted(self, state: State) -> None:
        """On level transition, fetch promoted programs and start one."""
        if state.level == self._last_level:
            return
        self._last_level = state.level
        # Don't override an in-flight program
        if self._current_program is not None:
            return
        promoted = self._ledger.promote_at_level(state.level)
        if not promoted:
            return
        # Pick the highest-confidence promoted program
        program, conf, pid = promoted[0]
        self._current_program = program
        self._current_program_id = pid
        self._current_program_score_baseline = self._scores_seen

    def _end_program_if_done(self, ended_naturally: bool) -> None:
        """Called when an in-flight program has no continuation (completed)
        or had to be abandoned. If the program completed without producing
        a score event since it started, register_failure.
        """
        if self._current_program_id is None:
            return
        # Did a score event happen since this program started?
        if (ended_naturally
                and self._scores_seen <= self._current_program_score_baseline):
            self._ledger.register_failure(
                self._current_program_id, self._last_level
            )
        self._current_program = None
        self._current_program_id = None

    def act(self, state: State, budget: ComputeBudget) -> Action:
        candidates = _candidate_actions(state)
        if not candidates:
            raise ValueError(
                f"MemoryBrainLedger has no candidates from "
                f"available_actions={state.available_actions}"
            )
        # Refresh goals (for observe-time course-correction)
        self._current_goals = self._l3.detect(state, self._history)
        # Maybe start a promoted program on level entry
        self._maybe_start_promoted(state)

        # 1. If in-flight Ledger program: step it, resolve sig
        if self._current_program is not None:
            sig, next_prog = self._current_program.step(state)
            action = _resolve_sig_to_action(sig, state, candidates)
            self._current_program = next_prog
            if action is None:
                # Can't resolve — abandon
                self._end_program_if_done(ended_naturally=False)
            elif next_prog is None:
                # Program done after this step; mark failure later if no
                # score in subsequent transitions
                self._end_program_if_done(ended_naturally=True)
                return action
            else:
                return action

        # 2. Fall back to Lookahead behavior (substrate Thompson + lookahead)
        # — replicate parent logic explicitly here (we already have
        # current_goals refreshed; just do the candidate loop):
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

    # Diagnostics
    @property
    def ledger(self) -> Ledger:
        return self._ledger


__all__ = ["MemoryBrainLedger"]
