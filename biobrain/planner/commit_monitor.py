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
from biobrain.planner.posterior import ActionScoreTable
from biobrain.planner.search_graph import SearchGraph
from biobrain.protocols import ActionLike, StateLike
from biobrain.motor_cortex.core import Program


# RL-TODO: derive from observed surprise distribution
VIOLATION_SURPRISE_THRESHOLD = 0.35


class CommitMonitorPlanner:
    """Event-driven Planner. Commits programs; monitors for violations.

    Sharing the substrate posterior with Curiosity (Residual): we no longer
    maintain a separate _substrate dict. The composer passes the shared
    ActionScoreTable at construction time. This fixes audit bug #1
    (split-substrate) — there is ONE source of truth for the action-
    signature Beta posterior.

    Three-path control:
      - hot path: step in-flight Program; if none or violation pending,
        escalate to cold path.
      - cold path: full reasoning; commit a Program; step it.
      - observe(surprise, action_sig, scored): set violation flag if
        surprise exceeds threshold. The substrate posterior is updated
        UPSTREAM in MemoryBrainResidual.observe (so we don't double-write).
        If a committed Program completes without scoring, register its
        failure with the Ledger.
    """

    def __init__(self, *,
                 seed: int = 0,
                 violation_threshold: float = VIOLATION_SURPRISE_THRESHOLD,
                 action_table: ActionScoreTable | None = None,
                 ) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._in_flight: Optional[Program] = None
        self._in_flight_id: Optional[str] = None
        # Track per-in-flight-program metadata for register_failure (bug #3)
        self._in_flight_start_level: int = 0
        self._in_flight_scored: bool = False
        self._violation_pending: bool = False
        self._violation_threshold = violation_threshold
        # Shared substrate posterior — passed by composer. If None,
        # we create our own (standalone usage / tests).
        self._action_table = action_table if action_table is not None \
            else ActionScoreTable()
        # v0.3 — within-game reachable-state graph (persists across attempts)
        self.search_graph = SearchGraph(max_nodes=10_000)
        # Diagnostic counters
        self._n_cold_calls = 0
        self._n_hot_calls = 0

    # ----------------------------------------------------------- lifecycle

    def reset_game(self, game_id: str) -> None:
        self._in_flight = None
        self._in_flight_id = None
        self._in_flight_start_level = 0
        self._in_flight_scored = False
        self._violation_pending = False
        self._rng = random.Random(self._seed)
        self._n_cold_calls = 0
        self._n_hot_calls = 0
        # NEW v0.3: reset the within-game search graph
        self.search_graph.reset_game()
        # NOTE: we do NOT reset the action_table here — that's owned by
        # the composer (shared with Residual). Composer's reset_game
        # handles it via residual.reset_game().

    def reset_attempt(self) -> None:
        # Wipe in-flight program (can't continue across attempts);
        # substrate carries (it lives in residual._action_table)
        self._in_flight = None
        self._in_flight_id = None
        self._in_flight_scored = False
        self._violation_pending = False

    def on_level_change(self, prev_level: int, new_level: int) -> None:
        # Abandon in-flight program at level boundary
        self._in_flight = None
        self._in_flight_id = None
        self._in_flight_scored = False
        self._violation_pending = True  # force cold-path on next act

    # ----------------------------------------------------------- observe

    def observe(self, surprise: float, action_sig: tuple,
                scored: bool, current_level: int = 0,
                ledger=None,
                transition=None,    # NEW v0.3
                attempt_id: int = 0  # NEW v0.3
                ) -> None:
        """Update violation flag; track in-flight program scoring.

        Substrate update is handled UPSTREAM in MemoryBrainResidual.observe
        (single source of truth). This method:
          1. Tracks whether the in-flight program scored.
          2. If a program completes without scoring across multiple steps,
             registers failure with the Ledger.
          3. Sets violation flag on surprise spike.

        Note: action_sig is kept for backward compat / diagnostics; the
        substrate update happens elsewhere now.
        """
        # v0.3 — record the transition's edge in the SearchGraph
        if transition is not None and transition.before is not None:
            self.search_graph.add_edge(
                parent_hash=int(transition.before.grid_hash),
                action_key=action_sig,
                child_hash=int(transition.after.grid_hash),
                attempt_id=attempt_id,
            )
            if scored:
                self.search_graph.mark_scoring(
                    int(transition.after.grid_hash), attempt_id)
        if scored:
            self._in_flight_scored = True
        # Violation detection — gate cold-path re-engagement
        if abs(surprise) >= self._violation_threshold:
            self._violation_pending = True
        # Track current level for register_failure on program completion
        self._current_level = current_level
        # Stash ledger reference for use when in-flight program completes
        self._ledger_ref = ledger

    # ----------------------------------------------------------- act paths

    def act(self,
            state: StateLike,
            candidates: list[ActionLike],
            encoder,
            critic_goals: Optional[list] = None,
            promoted_programs: Optional[list] = None,
            simulator=None,
            affordance_fn=None,
            last_state: Optional[StateLike] = None,
            ledger=None,
            transferred_subgoals: Optional[list] = None,
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
                    # Program completed. Register success or failure with Ledger
                    # depending on whether any score event fired during execution.
                    self._on_program_complete(ledger=ledger or
                                                getattr(self, "_ledger_ref", None),
                                                current_level=state.level)
                    self._in_flight = None
                    self._in_flight_id = None
                return action
            # Couldn't resolve → abandon program, fall through to cold
            self._on_program_complete(ledger=ledger or
                                        getattr(self, "_ledger_ref", None),
                                        current_level=state.level,
                                        was_aborted=True)
            self._in_flight = None
            self._in_flight_id = None

        # COLD path: full reasoning
        return self._cold_path(state, candidates, encoder,
                                critic_goals, promoted_programs,
                                simulator, affordance_fn, last_state, ledger,
                                transferred_subgoals)

    def _on_program_complete(self, ledger, current_level: int,
                              was_aborted: bool = False) -> None:
        """Bug #3 fix: register_failure when committed Program ends without score.

        Called when the in-flight Program returns next_program=None (natural
        completion) or when resolution fails (was_aborted=True). If the
        program produced ANY score event during its execution, we count
        that as success (no failure registration). Otherwise: register
        failure at the level the program was running at, so the Ledger's
        per-level Beta gets the negative evidence and promotion will be
        less confident next time.
        """
        if ledger is None or self._in_flight_id is None:
            return
        if self._in_flight_scored:
            # Program achieved a score event during execution → no failure.
            self._in_flight_scored = False
            return
        # Look up the program_id by object id (best-effort)
        try:
            for entry in ledger.all_entries():
                if id(entry.program) == self._in_flight_id:
                    ledger.register_failure(entry.program_id,
                                             self._in_flight_start_level)
                    break
        except Exception:
            pass

    def _cold_path(self,
                    state: StateLike,
                    candidates: list[ActionLike],
                    encoder,
                    critic_goals: Optional[list],
                    promoted_programs: Optional[list],
                    simulator,
                    affordance_fn,
                    last_state: Optional[StateLike] = None,
                    ledger=None,
                    transferred_subgoals: Optional[list] = None,
                    ) -> ActionLike:
        """Full reasoning: Thompson + Critic + Simulator + Ledger + Affordance."""
        self._n_cold_calls += 1
        self._violation_pending = False
        if not candidates:
            raise ValueError("No candidate actions")

        # 1. Try promoted programs via Thompson over substrate-Beta.
        # We DON'T pick first-takes-all — that over-commits to prior-level
        # programs that may not work at this level. Thompson naturally
        # explores: at a new level with no per-level evidence yet, all
        # promoted programs have similar Beta(1,1) ≈ uniform sampling.
        # As the brain tries each and they fail (register_failure on
        # observe), their per-level posteriors sharpen down.
        if promoted_programs:
            # Sample-rank promoted programs by Thompson over the substrate
            # signature of their FIRST action (we can't easily ask the
            # substrate about a multi-step program directly).
            best_promoted_score = -float("inf")
            best_promoted = None
            best_promoted_id = None
            for prog in promoted_programs:
                try:
                    sig, _ = prog.step(state)
                    # Use the program's id and current state.level to look
                    # up a per-program-per-level Beta. For v0 we use the
                    # substrate signature of the first resolved action as
                    # a proxy.
                    a = encoder.resolve(sig, state, candidates)
                    if a is None:
                        continue
                    action_sig = self._signature(a, state)
                    n_obs, n_goal = self._action_table.counts.get(
                        action_sig, (0, 0))
                    alpha = max(0.01, n_goal + 1.0)
                    beta = max(0.01, n_obs - n_goal + 1.0)
                    v = self._rng.betavariate(alpha, beta)
                    if v > best_promoted_score:
                        best_promoted_score = v
                        best_promoted = prog
                        best_promoted_id = id(prog)
                except Exception:
                    continue
            if best_promoted is not None:
                sig, next_prog = best_promoted.step(state)
                action = encoder.resolve(sig, state, candidates)
                if action is not None:
                    self._in_flight = next_prog
                    self._in_flight_id = best_promoted_id
                    self._in_flight_start_level = state.level
                    self._in_flight_scored = False
                    return action

        # 2. Substrate Thompson + lookahead per atomic candidate
        best_score = -float("inf")
        best_action = None
        # Bug #4 fix: pass last_state as `before` to get delta facts.
        # When last_state is None (start of attempt) we encode static-only.
        try:
            current_facts = encoder.encode(state, before=last_state)
        except TypeError:
            # Encoder doesn't accept before kwarg (legacy) — fall back
            current_facts = encoder.encode(state)
        # Current critic distance (state-level)
        current_d = self._goals_distance(current_facts, critic_goals)

        # v0.3 — compute first-action sets for pragmatic bonus lookup.
        # Each promoted Program's first step yields an ActionSig; resolve
        # to a concrete Action against the candidate pool for comparison.
        promoted_first_actions: set = set()
        if promoted_programs:
            for prog in promoted_programs:
                try:
                    sig, _ = prog.step(state)
                    a_first = encoder.resolve(sig, state, candidates)
                    if a_first is not None:
                        promoted_first_actions.add(tuple(a_first))
                except Exception:
                    continue
        # Transferred subgoals come from the Salience fingerprint index
        # (passed in via kwarg). Default to empty set when not provided.
        transferred_first_actions: set = set()
        if transferred_subgoals:
            for sg in transferred_subgoals:
                if sg.action_subsequence:
                    transferred_first_actions.add(tuple(sg.action_subsequence[0]))

        for a in candidates:
            sig = self._signature(a, state)
            n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
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

            # v0.3 — EV-augmented scoring per spec §4.2
            parent_hash = int(getattr(state, "grid_hash", 0))
            action_sig_for_graph = self._signature(a, state)
            child_hash = self.search_graph.child(parent_hash, action_sig_for_graph)
            # Epistemic — info gain (unexpanded edges scored high)
            epistemic = self._epistemic_score(parent_hash, action_sig_for_graph,
                                                symbolic_surprise=lookahead_bonus)
            # Pragmatic — progress toward goals + macro/subgoal first-step bonuses
            # NOTE: predicted_d here is the existing `current_d - lookahead_bonus`
            # — we invert the cached delta to feed _pragmatic_score
            pragmatic = self._pragmatic_score(
                action=a,
                current_d=current_d,
                predicted_d=(current_d - lookahead_bonus),
                promoted_first_actions=promoted_first_actions,
                transferred_first_actions=transferred_first_actions,
            )
            # Empowerment — control over reachable future from child
            empowerment = self._empowerment_score(child_hash, depth=2)

            # Equal weights for v0; RL-TODO: learn the weights
            ev = (epistemic + pragmatic + empowerment) / 3.0
            score = thompson + ev + affordance_bonus
            if score > best_score:
                best_score = score
                best_action = a

        return best_action if best_action is not None else self._rng.choice(candidates)

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _signature(action: ActionLike, state: StateLike) -> tuple:
        """Compact action signature — DELEGATES to ActionScoreTable._signature.

        Bug #1 fix: keep signature identical to the one used by
        MemoryBrainResidual / ActionScoreTable so the substrate posterior
        is keyed consistently across Curiosity and Planner. Single source
        of truth: ActionScoreTable._signature.
        """
        return ActionScoreTable._signature(action, state)

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

    # ----------------------------------------------------------- EV scoring

    def _epistemic_score(self, parent_hash: int, action_sig: tuple,
                          symbolic_surprise: float) -> float:
        """Epistemic = expected information gain.

        Unexpanded edges get a high prior (1.0); expanded edges get the
        WM's expected surprise. Bounded [0, 1].
        """
        if self.search_graph.child(parent_hash, action_sig) is None:
            # Unexpanded — high epistemic value
            return 1.0
        # Expanded — use the symbolic WM's surprise expectation, clipped
        return max(0.0, min(1.0, abs(symbolic_surprise)))

    def _pragmatic_score(self, action, current_d: float,
                          predicted_d: float,
                          promoted_first_actions: set,
                          transferred_first_actions: set) -> float:
        """Pragmatic = progress toward known goals.

        Combines:
          (a) Critic-distance reduction via 1-step lookahead;
          (b) bonus if action is the first step of a promoted macro;
          (c) bonus if action is the first step of a transferred subgoal.
        """
        d_reduction = max(0.0, current_d - predicted_d)
        a_first = tuple(action)
        # RL-TODO: 0.3 macro/subgoal first-step bonuses are hand-set.
        # Could be derived from per-game macro success rates.
        macro_bonus = 0.3 if a_first in promoted_first_actions else 0.0
        subgoal_bonus = 0.3 if a_first in transferred_first_actions else 0.0
        return d_reduction + macro_bonus + subgoal_bonus

    def _empowerment_score(self, child_hash, depth: int = 2) -> float:
        """Empowerment = |reachable states from child within depth K|.

        Normalized by a coarse upper bound. Returns ∈ [0, 1].
        Returns 0 if child is None (unexpanded edge) or is terminal.
        """
        if child_hash is None:
            return 0.0
        if child_hash not in self.search_graph._nodes:
            return 0.0
        node = self.search_graph.node_metadata(child_hash)
        if node and node.is_terminal:
            return 0.0
        n_reachable = self.search_graph.reachable_count(child_hash, depth)
        # RL-TODO: normalization constant 50.0 is a coarse bound. Could
        # be derived from observed reachable-count distribution per game.
        return min(1.0, n_reachable / 50.0)

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
