"""biobrain.curiosity — the dopamine system / intrinsic motivation.

Brain region: Dopaminergic system (ventral tegmental area / substantia nigra).
ML/RL term: ICM (Intrinsic Curiosity Module) / RPE (Reward Prediction Error).

The Curiosity module produces intrinsic reward when the world model is
surprised by an observation. As the world model learns a context, surprise
drops to zero — the agent gets "bored" of resolved dynamics and naturally
shifts attention to unresolved ones.

Implementation:
    1. World model = BayesianWorldModel: per-(fact, context) Beta posterior
       of P(fact in next-state | context).
    2. Observe transition: predict fact probabilities under current model,
       observe actual fact set, compute signed surprise.
    3. Inject surprise into the agent's action posterior as fractional credit.
       Positive surprise → α (encourages revisits to discover the pattern).
       Negative surprise → β (this context is predictable; less interesting).

The "encoder φ" of the ICM is `emit_atomic_facts` — symbolic, hand-engineered,
derived from Spelke primitives. No neural net, no training.

Cons of the symbolic encoder: features we don't emit, we don't see. The
predicate vocabulary determines the curiosity surface. The vocabulary
itself is derived from Spelke primitives (see predicate_pool.py).

Pros: fast, interpretable, no training, doesn't get stuck on chaotic
pixel-level noise.
"""

from __future__ import annotations

from typing import Optional

from biobrain.types import Action, State, Transition
from biobrain.curiosity.world_model import BayesianWorldModel
from biobrain.curiosity.residual import (
    MemoryBrainResidual, SURPRISE_CLIP,
)


class Curiosity:
    """ICM-style intrinsic reward via Bayesian world model.

    Single-source wrapper: the surprise computation lives in
    `MemoryBrainResidual._compute_signed_surprise`. We borrow a Residual
    brain instance internally to avoid duplicating the algorithm.

    Interface:
        observe(transition) — update world model + return signed surprise
        predict(state, action) → dict[fact, P(fact)]
        compute_surprise(before, action, after) → signed surprise ∈ [-1, 1]
    """

    def __init__(self) -> None:
        # Borrow the residual brain's WM + surprise computation as our
        # canonical implementation. We don't use its action policy.
        self._residual = MemoryBrainResidual(seed=0)

    def reset_game(self) -> None:
        self._residual = MemoryBrainResidual(seed=0)

    def reset_attempt(self) -> None:
        self._residual.reset_attempt()

    @property
    def world_model(self) -> BayesianWorldModel:
        return self._residual._world

    def predict(self, state: State, action: Action) -> dict:
        return self._residual._world.predict(state, action)

    def compute_surprise(self, before: State, action: Action,
                          after: State) -> float:
        """Signed surprise ∈ [-1, +1]. Delegates to MemoryBrainResidual."""
        return self._residual._compute_signed_surprise(before, action, after)

    def observe(self, transition: Transition) -> float:
        """Update world model from transition; return signed surprise clipped.

        Note: this only updates the world model — it does NOT inject
        credit into an action posterior. Curiosity is the signal; brains
        that consume it (like the Residual/Planner stack) handle the
        substrate-posterior shaping.
        """
        self._residual._world.observe(transition.before, transition.action,
                                       transition.after)
        if transition.before is None:
            return 0.0
        surprise = self.compute_surprise(transition.before, transition.action,
                                          transition.after)
        return max(-SURPRISE_CLIP, min(SURPRISE_CLIP, surprise))


__all__ = ["Curiosity"]
