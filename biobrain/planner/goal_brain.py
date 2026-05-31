"""biobrain.planner.goal_brain — Residual + L3 goal tracking (no action bias).

Extends MemoryBrainResidual to:
  - Maintain a TransitionHistory in observe().
  - Run the full L3 orchestrator (incl. dynamics-aware extractors) at act()
    and cache the detected proto-goals.
  - Expose current_goal_distance() so external diagnostics can read it.

Important: this brain does NOT bias action selection. A previous version
applied a click-only "goal cell" bonus (GOAL_BIAS_MAX = 0.3) on the
Thompson sample. That was a hack — hand-tuned, action-class-specific,
and empirically counterproductive on key-driven games like cd82 because
it starved key actions of exploration budget. The principled replacement
is per-action-class effect learning, which lives in MemoryBrainPlanner
via observe()-time credit injection (course-correction). This brain is
the wiring layer (L3 + history) that Planner builds on.
"""

from __future__ import annotations

from typing import Optional

from biobrain.types import (
    Action, ComputeBudget, State, Transition,
)
from biobrain.planner.agency import _candidate_actions
from biobrain.planner.posterior import ActionScoreTable
from biobrain.curiosity.residual import MemoryBrainResidual
from biobrain.critic.base import state_distance_to_goals
from biobrain.critic import L3, ProtoGoal, TransitionHistory


class MemoryBrainGoal(MemoryBrainResidual):
    """Residual brain + L3 goal tracking. No action bias.

    Inherits observe() semantics from MemoryBrainResidual (substrate
    update + world model update + signed-surprise injection). Adds:
      - TransitionHistory updated on each observe()
      - L3 orchestrator invoked at each act() to refresh proto-goals
      - current_goal_distance() for diagnostics

    Subclasses (MemoryBrainPlanner) USE the goals via observe-time credit.
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
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
                f"MemoryBrainGoal has no candidates from "
                f"available_actions={state.available_actions}"
            )
        # Refresh proto-goals (consumed by subclasses' observe()).
        self._current_goals = self._l3.detect(state, self._history)
        # Action selection: pure Thompson over the substrate posterior.
        # Surprise-injection (in observe) shapes the posterior; that IS
        # the brain's "use" of the world model for action selection here.
        best_score = -1.0
        best_action: Optional[Action] = None
        for a in candidates:
            sig = ActionScoreTable._signature(a, state)
            n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
            alpha = max(0.01, n_goal + 1.0)
            beta = max(0.01, n_obs - n_goal + 1.0)
            thompson = self._rng.betavariate(alpha, beta)
            if thompson > best_score:
                best_score = thompson
                best_action = a
        if best_action is None:
            best_action = self._rng.choice(candidates)
        return best_action

    def current_goal_distance(self, state: State) -> float:
        """Aggregate state distance under current goals. Diagnostic only."""
        return state_distance_to_goals(state, self._current_goals)
