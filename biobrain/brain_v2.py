"""biobrain.brain_v2 — the v0.2 composer (8 components + commit-and-monitor).

Audit-bug fixes integrated:
  #1 substrate single-source: Planner shares Residual's ActionScoreTable
     (no separate _substrate dict).
  #2 surprise single-compute: composer computes surprise ONCE, passes it
     to Residual.observe (no recomputation).
  #3 register_failure: Planner tracks in-flight program scoring; on
     program completion without score, registers failure with Ledger.
  #4 cold-path encode delta: cold path passes `last_state` to encoder
     so delta facts (any_change, count_up_color, etc.) are emitted.
  #6 observe order: surprise computed first; routed to all consumers.
  #7 initial transition: Salience/Planner still run with surprise=0.0
     when before is None (no skip).
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
    """v0.2 composer. 8 components. Commit-and-monitor control loop."""

    def __init__(self, *,
                 seed: int = 0,
                 adapter: Optional[Adapter] = None,
                 ) -> None:
        self.adapter: Adapter = adapter if adapter is not None else ArcAdapter()
        self.encoder: Encoder = self.adapter.encoder
        # Curiosity — owns the WM and the substrate ActionScoreTable
        self._residual = MemoryBrainResidual(seed=seed)
        # Salience — central organ
        self.salience = CentralSalience()
        self.salience.affordance.seed(self.adapter.initial_affordance_priors())
        # Critic
        self.critic = Critic()
        # Simulator
        self.simulator = Simulator(self._residual._world)
        # Ledger
        self.ledger = Ledger()
        # Planner — SHARES the substrate ActionScoreTable with Curiosity
        # (bug #1 fix)
        self.planner = CommitMonitorPlanner(
            seed=seed,
            action_table=self._residual._action_table,
        )
        # Track last seen state for cold-path encoder delta facts (bug #4)
        self._last_state: Optional[StateLike] = None
        # v0.3 — flag to set SearchGraph root once per game
        self._search_root_set: bool = False

    @staticmethod
    def _quadrant_of_entity(entity) -> int:
        """4x4 grid quadrant of entity centroid → 0..15. Place primitive.

        Mirrors prism.predicate_pool._quadrant_of but takes an entity
        object rather than raw cells. v0.3 — needed for fingerprint
        computation in observe() and act().
        """
        cells = list(entity.region.cells) if entity.region.cells else []
        if not cells:
            return 0
        cy = sum(c[0] for c in cells) / len(cells)
        cx = sum(c[1] for c in cells) / len(cells)
        by = min(3, max(0, int(cy // 16)))
        bx = min(3, max(0, int(cx // 16)))
        return by * 4 + bx

    # ============================================================ lifecycle

    def reset_game(self, game_id: str) -> None:
        self._residual.reset_game(game_id)
        self.salience.reset_game()
        self.salience.affordance.seed(self.adapter.initial_affordance_priors())
        self.critic.reset_game()
        self.simulator = Simulator(self._residual._world)
        self.ledger.reset_game()
        # Re-create planner with the FRESH action_table from residual
        self.planner = CommitMonitorPlanner(
            seed=self.planner._seed,
            action_table=self._residual._action_table,
        )
        self._last_state = None
        self._search_root_set = False

    def reset_attempt(self) -> None:
        self._residual.reset_attempt()
        self.salience.reset_attempt()
        self.planner.reset_attempt()
        self._last_state = None

    def _on_level_change(self, prev_level: int, new_level: int) -> None:
        if hasattr(self.ledger, "on_level_change"):
            self.ledger.on_level_change(prev_level, new_level)
        self.salience.on_level_change(prev_level, new_level)
        self.planner.on_level_change(prev_level, new_level)

    # ============================================================ observe

    def observe(self, transition: TransitionLike) -> None:
        """Process one transition through all components.

        Order (per audit bug #6 fix):
          1. Detect level change (defer firing the hook until after observe
             so per-component observe sees the new level naturally).
          2. Compute surprise ONCE (before any WM update — audit bug #2).
          3. Update Residual (substrate + WM) with the precomputed surprise.
          4. Salience.observe with the same surprise.
          5. Critic history update.
          6. Ledger.observe (with before-level score_level — fixed).
          7. Planner.observe with the same surprise (no double-write).
          8. Fire on_level_change hook if applicable.
        """
        # v0.3 — set the SearchGraph root once per game (first valid observation)
        if (transition.before is not None
                and not self._search_root_set):
            self.planner.search_graph.set_root(
                int(transition.before.grid_hash))
            self._search_root_set = True

        # v0.3 — compute n_cells_changed_elsewhere for role-counter updates.
        # Used by Salience to identify "selector"-class actions.
        n_cells_changed_elsewhere = 0
        if transition.before is not None and transition.action is not None:
            try:
                import numpy as np
                g_before = np.asarray(transition.before.raw_grid)
                g_after = np.asarray(transition.after.raw_grid)
                if g_before.shape == g_after.shape:
                    diff_mask = g_before != g_after
                    # For clicks, subtract cells near the click position so
                    # only "elsewhere" changes count.
                    if (transition.action[0] == "click"
                            and len(transition.action) >= 3):
                        x, y = int(transition.action[1]), int(transition.action[2])
                        for r in range(max(0, y - 1),
                                        min(g_before.shape[0], y + 2)):
                            for c in range(max(0, x - 1),
                                            min(g_before.shape[1], x + 2)):
                                diff_mask[r, c] = False
                    n_cells_changed_elsewhere = int(diff_mask.sum())
            except Exception:
                pass

        # Compute surprise ONCE (before WM update)
        precomputed_surprise = 0.0
        if transition.before is not None and transition.action is not None:
            try:
                precomputed_surprise = self._residual._compute_signed_surprise(
                    transition.before, transition.action, transition.after,
                )
            except Exception:
                precomputed_surprise = 0.0

        # Residual: substrate + WM updates (passes back the clipped value
        # actually injected; we ignore that — we use the precomputed)
        self._residual.observe(transition,
                                precomputed_surprise=precomputed_surprise)

        # Salience: bank surprise, update affordance, request fine attention
        if transition.before is not None and transition.action is not None:
            try:
                predicted = self._residual._world.predict(
                    transition.before, transition.action)
                predicted_facts = frozenset(
                    f for f, p in predicted.items() if p >= 0.5)
                actual_facts = self.encoder.encode(
                    transition.after, before=transition.before)
            except Exception:
                predicted_facts = frozenset()
                actual_facts = frozenset()
            from biobrain.types import action_kind
            context = (
                action_kind(transition.action),
                None,
                transition.after.level,
            )
            self.salience.observe(
                surprise=precomputed_surprise,
                context=context,
                action=transition.action,
                predicted_facts=predicted_facts,
                actual_facts=actual_facts,
            )

        # v0.3 — role-counter and fingerprint machinery
        self.salience.update_causal_counters(
            transition, n_cells_changed_elsewhere=n_cells_changed_elsewhere)
        self.salience.refresh_role_assignments()
        if transition.before is not None and transition.action is not None:
            fp_before = self.salience.current_fingerprint(
                transition.before,
                quadrant_of=self._quadrant_of_entity)
            fp_after = self.salience.current_fingerprint(
                transition.after,
                quadrant_of=self._quadrant_of_entity)
            # Critic-distance drop as validation channel
            try:
                from biobrain.critic.base import state_distance_to_goals
                d_before = state_distance_to_goals(
                    transition.before,
                    self.critic.evaluate(transition.before))
                d_after = state_distance_to_goals(
                    transition.after,
                    self.critic.evaluate(transition.after))
                critic_dropped = d_after < d_before
            except Exception:
                critic_dropped = False
            self.salience.detect_subgoal(
                fingerprint_before=fp_before,
                fingerprint_after=fp_after,
                action=transition.action,
                critic_distance_dropped=critic_dropped,
                source_level=transition.after.level,
                source_attempt_id=0,  # v0.3 — composer doesn't track yet
            )

        # Critic history update
        self.critic.observe_transition(transition)

        # Ledger (with before-level score_level)
        self.ledger.observe(transition)

        # Planner: track in-flight scoring + violation flag
        if transition.action is not None:
            scored = any(
                e.kind in ("ScoreIncreased", "LevelIncreased")
                for e in transition.events
            )
            action_sig = CommitMonitorPlanner._signature(
                transition.action,
                transition.before if transition.before is not None else
                transition.after,
            )
            self.planner.observe(
                surprise=precomputed_surprise,
                action_sig=action_sig,
                scored=scored,
                current_level=transition.after.level,
                ledger=self.ledger,
                transition=transition,  # v0.3 — for SearchGraph edge recording
                attempt_id=0,            # v0.3 — composer doesn't track yet
            )

        # Level change hook
        if (transition.before is not None
                and transition.after.level > transition.before.level):
            self._on_level_change(transition.before.level,
                                   transition.after.level)

        self._last_state = transition.after

    # ============================================================ act

    def act(self, state: StateLike, budget: ComputeBudget) -> ActionLike:
        candidates = self.encoder.candidate_actions(state)
        if not candidates:
            raise ValueError("no candidate actions available")
        critic_goals = self.critic.evaluate(state)
        promoted = []
        try:
            ledger_promotions = self.ledger.promote_at_level(state.level)
            promoted = [p for p, _conf, _id in ledger_promotions]
        except Exception:
            pass
        affordance_fn = self.salience.get_affordance
        # v0.3 — look up transferred subgoals via fingerprint index
        transferred_subgoals: list = []
        try:
            fp_current = self.salience.current_fingerprint(
                state, quadrant_of=self._quadrant_of_entity)
            transferred_subgoals = self.salience.fingerprint_index.lookup(
                fp_current)
        except Exception:
            transferred_subgoals = []
        return self.planner.act(
            state=state,
            candidates=candidates,
            encoder=self.encoder,
            critic_goals=critic_goals,
            promoted_programs=promoted,
            simulator=self.simulator,
            affordance_fn=affordance_fn,
            last_state=self._last_state,
            ledger=self.ledger,
            transferred_subgoals=transferred_subgoals,  # v0.3
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
