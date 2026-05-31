"""biobrain.brain_v2 — the v0.2 composer (8 components + commit-and-monitor).

This is the canonical BioBrain class for v0.2. It composes:
  1. Motor Cortex (via DSL programs)
  2. Perception (Encoder)
  3. Salience (central organ)
  4. Curiosity (Bayesian WM + signed surprise)
  5. Critic (multi-extractor L3)
  6. Simulator (forward queries via WM)
  7. Ledger (per-game program memory)
  8. Planner (commit-and-monitor)

The composer is the ONLY thing that knows the inter-component wiring.
Each component is pure (no other-component references). Data flow is
explicit in observe() and act().
"""

from __future__ import annotations

from typing import Optional

from biobrain.adapters.arc.adapter import ArcAdapter
from biobrain.critic import Critic
from biobrain.curiosity.world_model import BayesianWorldModel
from biobrain.curiosity.residual import (
    MemoryBrainResidual, SURPRISE_CLIP,
)
from biobrain.curiosity.predicates import emit_atomic_facts
from biobrain.ledger.ledger import Ledger
from biobrain.planner.commit_monitor import CommitMonitorPlanner
from biobrain.protocols import (
    Adapter, ActionLike, Encoder, StateLike, TransitionLike,
)
from biobrain.salience.central import CentralSalience
from biobrain.simulator.simulator import Simulator
from biobrain.types import ComputeBudget


class BioBrainV2:
    """v0.2 composer. 8 components. Commit-and-monitor control loop.

    Public interface (matches the simpler BrainEngine):
      reset_game(game_id)
      reset_attempt()
      observe(transition)
      act(state, budget) → Action
      end_of_attempt()
    """

    def __init__(self, *,
                 seed: int = 0,
                 adapter: Optional[Adapter] = None,
                 ) -> None:
        self.adapter: Adapter = adapter if adapter is not None else ArcAdapter()
        self.encoder: Encoder = self.adapter.encoder
        # 4. Curiosity (Bayesian WM + signed surprise + substrate Beta)
        # We use the existing MemoryBrainResidual as the source of the WM,
        # the signed-surprise computation, and the substrate posterior
        # (which it shares with the Planner via observe()).
        self._residual = MemoryBrainResidual(seed=seed)
        # 3. Salience — central organ
        self.salience = CentralSalience()
        # Seed affordance from adapter (default: uniform)
        self.salience.affordance.seed(self.adapter.initial_affordance_priors())
        # 5. Critic — multi-extractor L3
        self.critic = Critic()
        # 6. Simulator — forward queries via WM
        self.simulator = Simulator(self._residual._world)
        # 7. Ledger — per-game program memory
        self.ledger = Ledger()
        # 8. Planner — commit-and-monitor
        self.planner = CommitMonitorPlanner(seed=seed)
        # Track last-seen level for on_level_change detection
        self._last_level: int = -1

    # ============================================================ lifecycle

    def reset_game(self, game_id: str) -> None:
        """Inter-game amnesia: wipe everything except adapter and encoder."""
        self._residual.reset_game(game_id)
        self.salience.reset_game()
        self.salience.affordance.seed(self.adapter.initial_affordance_priors())
        self.critic.reset_game()
        # Rebuild simulator with fresh WM
        self.simulator = Simulator(self._residual._world)
        self.ledger.reset_game()
        self.planner.reset_game(game_id)
        self._last_level = -1

    def reset_attempt(self) -> None:
        """Intra-game memory preserved; only per-attempt transients clear."""
        self._residual.reset_attempt()
        self.salience.reset_attempt()
        self.planner.reset_attempt()
        # Ledger and Critic preserve across attempts (no reset_attempt needed)

    def _on_level_change(self, prev_level: int, new_level: int) -> None:
        """Explicit control-flow event at level transitions."""
        self.ledger.on_level_change(prev_level, new_level) if hasattr(
            self.ledger, "on_level_change") else None
        self.salience.on_level_change(prev_level, new_level)
        self.planner.on_level_change(prev_level, new_level)

    # ============================================================ observe

    def observe(self, transition: TransitionLike) -> None:
        """Process one transition through all components, in order.

        Order matters: Curiosity computes surprise → Salience consumes it
        → Critic updates history → Ledger tracks → Planner observes.
        """
        # 1. Substrate + WM update (also computes signed surprise internally)
        self._residual.observe(transition)
        # 2. Critic history update (for ChangeDynamics-family extractors)
        self.critic.observe_transition(transition)
        # 3. Ledger trajectory tracking + score-event abstraction
        self.ledger.observe(transition)
        # 4. Salience: compute surprise here (from current WM) and update
        if transition.before is not None and transition.action is not None:
            from biobrain.types import action_kind
            predicted = self._residual._world.predict(
                transition.before, transition.action)
            predicted_facts = frozenset(
                f for f, p in predicted.items() if p >= 0.5)
            actual_facts = self.encoder.encode(transition.after,
                                                before=transition.before)
            # Recompute signed surprise (Salience consumes this)
            surprise = self._residual._compute_signed_surprise(
                transition.before, transition.action, transition.after,
            )
            context = (
                action_kind(transition.action),
                None,  # target_color extraction happens in residual; here we
                       # use the action_kind for Salience's affordance update
                transition.after.level,
            )
            self.salience.observe(
                surprise=surprise,
                context=context,
                action=transition.action,
                predicted_facts=predicted_facts,
                actual_facts=actual_facts,
            )
            # 5. Planner observation: surprise for violation trigger
            scored = any(
                e.kind in ("ScoreIncreased", "LevelIncreased")
                for e in transition.events
            )
            from biobrain.planner.commit_monitor import CommitMonitorPlanner
            sig = CommitMonitorPlanner._signature(
                transition.action, transition.before)
            self.planner.observe(surprise=surprise, action_sig=sig,
                                  scored=scored)
        # 6. Level change detection — composer-level event
        if (transition.before is not None
                and transition.after.level > transition.before.level):
            self._on_level_change(transition.before.level,
                                   transition.after.level)
        self._last_level = transition.after.level

    # ============================================================ act

    def act(self, state: StateLike, budget: ComputeBudget) -> ActionLike:
        """Hot path → cold path as needed. Returns concrete Action."""
        candidates = self.encoder.candidate_actions(state)
        if not candidates:
            raise ValueError("no candidate actions available")
        # Salience may have queued fine-attention cells; the Encoder can
        # consume them on the next encode() call. For this act, use coarse.
        # Get current Critic goals
        critic_goals = self.critic.evaluate(state)
        # Get promoted programs from Ledger
        promoted = []
        try:
            ledger_promotions = self.ledger.promote_at_level(state.level)
            promoted = [p for p, _conf, _id in ledger_promotions]
        except Exception:
            pass
        # Affordance lookup function
        affordance_fn = self.salience.get_affordance
        # Route through Planner (which handles hot/cold paths internally)
        return self.planner.act(
            state=state,
            candidates=candidates,
            encoder=self.encoder,
            critic_goals=critic_goals,
            promoted_programs=promoted,
            simulator=self.simulator,
            affordance_fn=affordance_fn,
        )

    def end_of_attempt(self) -> None:
        pass

    # ============================================================ diagnostics

    @property
    def n_hot_calls(self) -> int:
        return self.planner.hot_call_count

    @property
    def n_cold_calls(self) -> int:
        return self.planner.cold_call_count

    @property
    def world_model(self) -> BayesianWorldModel:
        return self._residual._world


__all__ = ["BioBrainV2"]
