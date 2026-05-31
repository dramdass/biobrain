"""biobrain.motor_cortex.goal_dream_brain — MemoryBrainDream + L3 tracking.

Extends MemoryBrainDream with two things:
  - The expanded library that includes SEQ(click(c1), click(c2)) modal-
    paint templates (relevant for click-paint games where two-step
    interactions are needed).
  - TransitionHistory + L3 orchestrator wiring so subclasses or
    diagnostics can read what proto-goals were detected.

This brain does NOT apply program-level goal bias at action selection
time. A prior version added a `_program_goal_bonus` of up to
PROGRAM_GOAL_BIAS_MAX=0.4 to each library program's Thompson sample
based on whether the program's target cells overlapped goal regions.
That was a hack: hand-tuned constant, hand-rolled overlap heuristic,
and program-level (not action-effect) bias. Removed for the same
reason MemoryBrainGoal lost its click-only bias — the principled
mechanism is effect-based learning via observe() credit, not hand-
rolled bias at decision time.
"""

from __future__ import annotations

import random
from typing import Optional

from biobrain.types import (
    Action, ComputeBudget, State, Transition,
)
from biobrain.planner.agency import _candidate_actions
from biobrain.planner.posterior import ActionScoreTable
from biobrain.motor_cortex.brain import (
    MemoryBrainDream, _resolve_sig_to_action, seed_library,
)
from biobrain.motor_cortex.core import (
    Program, SEQ, click_on_color,
)
from biobrain.critic import L3, ProtoGoal, TransitionHistory


def seed_library_with_click_pairs(
        observed_colors: list, observed_keys: list
) -> list[tuple[str, Program]]:
    """Extend base library with SEQ(click(c1), click(c2)) modal-paint
    templates. Bounds combinatorial expansion to c1 ≠ c2 pairs.
    """
    base = seed_library(observed_colors, observed_keys)
    extras: list[tuple[str, Program]] = []
    for c1 in observed_colors:
        for c2 in observed_colors:
            if c1 == c2:
                continue
            extras.append((
                f"click({c1}) → click({c2})",
                SEQ(click_on_color(c1), click_on_color(c2)),
            ))
    return base + extras


class MemoryBrainGoalDream(MemoryBrainDream):
    """MemoryBrainDream + click-pair library + L3 tracking.

    Same act() flow as MemoryBrainDream (program Thompson vs substrate
    Thompson). The L3 layer detects proto-goals each step for
    diagnostics; it does not bias program selection (removed hack).
    """

    def __init__(self, *, seed: int = 0,
                 observed_colors: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15),
                 observed_keys: tuple = (0, 1, 2, 3, 4)) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        from biobrain.planner.representation import RepLoop
        self._rep = RepLoop()
        self._action_table = ActionScoreTable()
        self._library = seed_library_with_click_pairs(
            list(observed_colors), list(observed_keys)
        )
        self._posterior: dict[str, tuple[float, float]] = {
            pid: (0, 0) for pid, _ in self._library
        }
        self._current_program: Optional[Program] = None
        self._current_program_id: Optional[str] = None
        self._observed_count: int = 0
        self._current_goals: list[ProtoGoal] = []
        self._l3 = L3()
        self._history = TransitionHistory()

    def reset_game(self, game_id: str) -> None:
        super().reset_game(game_id)
        self._current_goals = []
        self._history = TransitionHistory()

    def observe(self, transition: Transition) -> None:
        super().observe(transition)
        self._history.update(transition)

    def act(self, state: State, budget: ComputeBudget) -> Action:
        candidates = _candidate_actions(state)
        if not candidates:
            raise ValueError(
                f"MemoryBrainGoalDream has no candidates from "
                f"available_actions={state.available_actions}"
            )

        # Refresh proto-goals (diagnostic; not used for biasing here).
        self._current_goals = self._l3.detect(state, self._history)

        # 1. Continue a multi-step program in flight?
        if self._current_program is not None:
            sig, next_program = self._current_program.step(state)
            self._current_program = next_program
            action = _resolve_sig_to_action(sig, state, candidates)
            if action is not None:
                return action
            self._current_program = None
            self._current_program_id = None

        # 2. Thompson-sample programs (no goal bias)
        best_score = -1.0
        best_id = None
        best_program = None
        for pid, prog in self._library:
            α, β = self._posterior.get(pid, (0, 0))
            v = self._rng.betavariate(α + 1, β + 1)
            if v > best_score:
                best_score = v
                best_id = pid
                best_program = prog

        # 3. Substrate fallback
        substrate_score = -1.0
        substrate_action = None
        for a in candidates:
            sig = ActionScoreTable._signature(a, state)
            n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
            v = self._rng.betavariate(n_goal + 1, n_obs - n_goal + 1)
            if v > substrate_score:
                substrate_score = v
                substrate_action = a

        if best_score > substrate_score and best_program is not None:
            sig, next_program = best_program.step(state)
            action = _resolve_sig_to_action(sig, state, candidates)
            if action is not None:
                self._current_program = next_program
                self._current_program_id = best_id
                return action

        return substrate_action if substrate_action is not None \
            else self._rng.choice(candidates)
