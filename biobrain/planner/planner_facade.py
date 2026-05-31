"""biobrain.planner — the actor / policy.

Brain region: Prefrontal cortex (executive) + basal ganglia (selection).
ML/RL term: Actor / policy.

The Planner picks the next real action. It combines four signals:
    1. Curiosity (intrinsic) — surprise injection into action posterior
    2. Critic (extrinsic-mimicking) — goal-distance reduction
    3. Ledger (Strategic / scientific method) — known-good scripts
    4. Simulator (lookahead) — predicted Critic on imagined next-states

Decision rule (unified Bayesian):
    For each candidate action a:
        score(a) = Thompson(substrate posterior[a])
                 + optional value_prior(simulator_lookahead(a))
                 + optional value_prior(ledger.confidence(a))
    Choose argmax score(a).

The Ledger/Simulator components contribute as additive value priors,
not as competing policies. Hierarchical Beta inside the Ledger and
substrate posterior decay both regulate exploration/exploitation
naturally — no hardcoded threshold for "trust Ledger over Curiosity."

STATUS: currently delegates to `biobrain.planner.planner_brain.MemoryBrainPlanner`,
which implements Curiosity + Critic + substrate Thompson. Ledger and
Simulator wiring is pending Phases 2-3.
"""

from __future__ import annotations

from biobrain.types import Action, ComputeBudget, State, Transition
from biobrain.planner.planner_brain import MemoryBrainPlanner


class Planner:
    """Composes signals into action selection.

    Current implementation: thin wrapper around MemoryBrainPlanner.
    The planner brain already integrates:
      - substrate Beta+Thompson (Curiosity inputs via surprise injection)
      - L3 goal tracking (Critic inputs via TransitionHistory)
      - course-correction credit (ΔDistance over Critic state-value)

    Pending additions (Phases 2-3):
      - Simulator lookahead value prior
      - Ledger program promotion
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._brain = MemoryBrainPlanner(seed=seed)

    def reset_game(self, game_id: str) -> None:
        self._brain.reset_game(game_id)

    def reset_attempt(self) -> None:
        self._brain.reset_attempt()

    def observe(self, transition: Transition) -> None:
        self._brain.observe(transition)

    def act(self, state: State, budget: ComputeBudget) -> Action:
        return self._brain.act(state, budget)

    def end_of_attempt(self) -> None:
        self._brain.end_of_attempt()

    @property
    def underlying_brain(self) -> MemoryBrainPlanner:
        """For diagnostics. Don't use to inject behavior."""
        return self._brain


__all__ = ["Planner"]
