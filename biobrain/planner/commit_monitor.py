"""biobrain.planner.commit_monitor — the commit-and-monitor Planner.

Per the phenomenology correction: humans don't reason every step. They
commit a hypothesis (a Program), execute it open-loop predicting each
step, and re-engage expensive reasoning only on violations.

Three paths:
  HOT:  default — runs every step. Step the in-flight Program continuation,
        compute prediction error, return Action. Cost: one WM lookup + one
        Program step.

  WARM: implicit in observe() — runs every transition. WM update + Salience
        banking + Critic history + Ledger trajectory tracking. Learning
        never stops.

  COLD: decision time — runs on violation OR program-end. Full reasoning
        stack: Thompson over substrate, Critic-distance evaluation,
        Simulator-mediated lookahead, Ledger-promoted programs, affordance
        prior. Picks a Program, commits.

The violation trigger is the surprise signal: when surprise exceeds
`VIOLATION_SURPRISE_THRESHOLD`, the brain abandons the in-flight Program
and re-engages cold path.
"""

from __future__ import annotations

import random
from typing import Optional

from biobrain.curiosity.residual import SURPRISE_CLIP
from biobrain.protocols import ActionLike, StateLike
from biobrain.motor_cortex.core import Program


# RL-TODO: derive from observed surprise distribution
VIOLATION_SURPRISE_THRESHOLD = 0.35


class CommitMonitorPlanner:
    """Event-driven Planner. Commits programs; monitors for violations.

    Three-path control:
      - hot_step(state, candidates, encoder): step in-flight program; if
        none or violation pending, escalate to cold.
      - cold_step(state, ...): full reasoning; commit a Program; step it.
      - observe(surprise, transition): set violation flag if surprise
        exceeds threshold; update substrate.

    State:
      - in-flight Program continuation
      - violation pending flag
      - substrate posterior (action-signature → Beta)
      - random generator (Thompson)
    """

    def __init__(self, *,
                 seed: int = 0,
                 violation_threshold: float = VIOLATION_SURPRISE_THRESHOLD,
                 ) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._in_flight: Optional[Program] = None
        self._in_flight_id: Optional[str] = None
        self._violation_pending: bool = False
        self._violation_threshold = violation_threshold
        # Substrate posterior: (action_sig) → (n_obs, n_goal)
        self._substrate: dict[tuple, tuple[float, float]] = {}
        # Track when we last entered cold path (for diagnostics)
        self._n_cold_calls = 0
        self._n_hot_calls = 0

    # ----------------------------------------------------------- lifecycle

    def reset_game(self, game_id: str) -> None:
        self._in_flight = None
        self._in_flight_id = None
        self._violation_pending = False
        self._substrate = {}
        self._rng = random.Random(self._seed)
        self._n_cold_calls = 0
        self._n_hot_calls = 0

    def reset_attempt(self) -> None:
        # Wipe in-flight program (can't continue across attempts); keep substrate
        self._in_flight = None
        self._in_flight_id = None
        self._violation_pending = False

    def on_level_change(self, prev_level: int, new_level: int) -> None:
        # Abandon in-flight program at level boundary
        self._in_flight = None
        self._in_flight_id = None
        self._violation_pending = True  # force cold-path on next act

    # ----------------------------------------------------------- observe

    def observe(self, surprise: float, action_sig: tuple,
                scored: bool) -> None:
        """Update substrate posterior; set violation flag if surprise high."""
        # Substrate update — same pattern as residual injection
        n_obs, n_goal = self._substrate.get(action_sig, (0.0, 0.0))
        if scored:
            self._substrate[action_sig] = (n_obs, n_goal + 1.0)
        else:
            clipped = max(-SURPRISE_CLIP, min(SURPRISE_CLIP, surprise))
            if clipped >= 0:
                self._substrate[action_sig] = (n_obs, n_goal + clipped)
            else:
                self._substrate[action_sig] = (n_obs + (-clipped), n_goal)
        # Violation detection — gate cold-path re-engagement
        if abs(surprise) >= self._violation_threshold:
            self._violation_pending = True

    # ----------------------------------------------------------- act paths

    def act(self,
            state: StateLike,
            candidates: list[ActionLike],
            encoder,
            critic_goals: Optional[list] = None,
            promoted_programs: Optional[list] = None,
            simulator=None,
            affordance_fn=None,
            ) -> ActionLike:
        """Single entry point. Routes to hot or cold path based on state.

        Returns a concrete Action.
        """
        # HOT path: continue in-flight program if available and no violation
        if self._in_flight is not None and not self._violation_pending:
            self._n_hot_calls += 1
            sig, next_prog = self._in_flight.step(state)
            self._in_flight = next_prog
            action = encoder.resolve(sig, state, candidates)
            if action is not None:
                if next_prog is None:
                    # Program completed — next act will go cold
                    self._in_flight = None
                    self._in_flight_id = None
                return action
            # Couldn't resolve → abandon program, fall through to cold
            self._in_flight = None
            self._in_flight_id = None

        # COLD path: full reasoning
        return self._cold_path(state, candidates, encoder,
                                critic_goals, promoted_programs,
                                simulator, affordance_fn)

    def _cold_path(self,
                    state: StateLike,
                    candidates: list[ActionLike],
                    encoder,
                    critic_goals: Optional[list],
                    promoted_programs: Optional[list],
                    simulator,
                    affordance_fn,
                    ) -> ActionLike:
        """Full reasoning: Thompson + Critic + Simulator + Ledger + Affordance."""
        self._n_cold_calls += 1
        self._violation_pending = False
        if not candidates:
            raise ValueError("No candidate actions")

        # 1. Try promoted programs first (Ledger)
        if promoted_programs:
            for prog in promoted_programs[:3]:  # top-3 by confidence
                sig, next_prog = prog.step(state)
                action = encoder.resolve(sig, state, candidates)
                if action is not None:
                    self._in_flight = next_prog
                    self._in_flight_id = id(prog)
                    return action

        # 2. Substrate Thompson + lookahead per atomic candidate
        best_score = -float("inf")
        best_action = None
        current_facts = encoder.encode(state)
        # Current critic distance (state-level)
        current_d = self._goals_distance(current_facts, critic_goals)

        for a in candidates:
            sig = self._signature(a, state)
            n_obs, n_goal = self._substrate.get(sig, (0.0, 0.0))
            alpha = max(0.01, n_goal + 1.0)
            beta = max(0.01, n_obs - n_goal + 1.0)
            thompson = self._rng.betavariate(alpha, beta)

            # Lookahead via simulator (if available)
            lookahead_bonus = 0.0
            if simulator is not None and critic_goals:
                try:
                    predicted = simulator.simulate_one(state, a)
                    if predicted:
                        predicted_d = self._goals_distance(predicted, critic_goals)
                        lookahead_bonus = max(
                            -SURPRISE_CLIP,
                            min(SURPRISE_CLIP, current_d - predicted_d),
                        )
                except Exception:
                    pass

            # Affordance prior
            affordance_bonus = 0.0
            if affordance_fn is not None:
                kind = a[0] if a else "unknown"
                affordance_bonus = (affordance_fn(kind) - 0.5) * 0.2

            score = thompson + lookahead_bonus + affordance_bonus
            if score > best_score:
                best_score = score
                best_action = a

        return best_action if best_action is not None else self._rng.choice(candidates)

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _signature(action: ActionLike, state: StateLike) -> tuple:
        """Compact action signature for substrate posterior.

        (action_kind, target_color, level). For clicks, target_color is the
        color of the entity at the click position (or 'empty' if none).
        For keys/spacebar/undo, target_color is None.
        """
        kind = action[0] if action else "unknown"
        level = getattr(state, "level", 0)
        target_color = None
        if kind == "click" and len(action) >= 3:
            x, y = int(action[1]), int(action[2])
            for e in state.entities:
                if (y, x) in e.region.cells:
                    target_color = int(e.color)
                    break
            if target_color is None:
                target_color = "empty"
        return (kind, target_color, level)

    @staticmethod
    def _goals_distance(facts_or_state, goals) -> float:
        """Weighted-mean distance over goals whose distance_fn accepts facts/state."""
        if not goals:
            return 0.5
        weighted_sum = 0.0
        weight_total = 0.0
        for g in goals:
            try:
                d = g.distance_fn(facts_or_state)
            except (TypeError, AttributeError, KeyError):
                continue
            weighted_sum += g.weight * d
            weight_total += g.weight
        if weight_total == 0:
            return 0.5
        return weighted_sum / weight_total

    # ----------------------------------------------------------- diagnostics

    @property
    def hot_call_count(self) -> int:
        return self._n_hot_calls

    @property
    def cold_call_count(self) -> int:
        return self._n_cold_calls

    @property
    def violation_pending(self) -> bool:
        return self._violation_pending


__all__ = ["CommitMonitorPlanner", "VIOLATION_SURPRISE_THRESHOLD"]
