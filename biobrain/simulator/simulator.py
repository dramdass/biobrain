"""biobrain.simulator — mental sandbox / forward dynamics model.

Brain region: Prefrontal cortex (deliberative reasoning).
ML/RL term: Forward dynamics model / mental simulation / model-based planning.

STATUS: Interface stub. Implementation pending Phase 2 (after Phase 1
World Model accuracy validation).

The Simulator wraps the BayesianWorldModel for forward queries: given
(state, action), return predicted next-state facts. The Critic can then
score predicted states without executing actions in the real environment,
enabling lookahead and (eventually) MCTS.

Key design choices:

  Fact-space, not pixel-space: predicted "next state" is a set of predicates,
  not a raw grid. This aligns with the Critic's abstraction level
  (post-Phase-0 migration to fact-space) and avoids the lossy pixel
  prediction trap that hurts neural ICM.

  Sampled vs deterministic: each fact has a Beta posterior P(fact_next).
  We sample a fact set (Bernoulli over each posterior) for a stochastic
  rollout, or take the mode for a deterministic projection.

  Bounded rollout depth: errors compound across multi-step rollouts. For
  early MCTS depth ≤ 2-3 is realistic; deeper rollouts require better
  world model coverage.

The Phase 1 measurement (validate_world_model.py) gates this work:
  - If WM hits >70% per-step fact-prediction accuracy → Simulator viable.
  - If accuracy < 50% → World Model needs richer predictions first.
"""

from __future__ import annotations

from typing import Optional

from biobrain.types import Action, State
from biobrain.curiosity.world_model import BayesianWorldModel


class Simulator:
    """Forward-dynamics queries over BayesianWorldModel.

    Interface (designed; partial implementation):
        simulate_one(state, action) → predicted_facts
            Returns the set of facts the world model predicts will hold
            in the next state given this action.
        rollout(state, action_sequence, depth) → list[predicted_facts]
            Multi-step rollout. NOT IMPLEMENTED — needs Phase 2.
    """

    def __init__(self, world_model: BayesianWorldModel) -> None:
        self._wm = world_model

    def simulate_one(self, state: State, action: Action,
                     threshold: float = 0.5) -> set:
        """Deterministic projection: predicted facts whose P > threshold.

        Returns: set of facts. Empty when world model has no evidence at
        this (action_kind, target_color, level) context.
        """
        predicted = self._wm.predict(state, action)
        return {fact for fact, p in predicted.items() if p >= threshold}

    def simulate_one_sampled(self, state: State, action: Action,
                              rng) -> set:
        """Stochastic Bernoulli sampling from per-fact posteriors.

        Each predicted fact is included with probability P(fact_next | context).
        Useful for MCTS-style rollouts where stochasticity broadens search.
        """
        predicted = self._wm.predict(state, action)
        return {fact for fact, p in predicted.items() if rng.random() < p}

    def rollout(self, state: State, action_sequence: list,
                depth: int = 3) -> list[set]:
        """Multi-step rollout. NOT IMPLEMENTED — Phase 2.

        Open question: how to materialize an intermediate "predicted state"
        usable as input to the next prediction step. Two approaches:
          (a) carry forward only the predicted fact set (no state object) —
              cheap but loses entity-level context the WM relies on.
          (b) reconstruct a synthetic State from predicted facts — expensive,
              error-prone, requires inverse φ.

        Phase 2 design decision: probably (a) for K=1 lookahead, defer (b)
        until MCTS phase.
        """
        raise NotImplementedError("Phase 2 build target")


__all__ = ["Simulator"]
