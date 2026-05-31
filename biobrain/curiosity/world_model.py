"""biobrain.curiosity.world_model — per-variable Bayesian world model.

Per the user's deep reframe: every forward pass we should predict every
state variable conditioned on action and history. Score becomes one
variable among many. Curiosity (information gain) emerges as the
exploration driver — no score events needed to bootstrap.

For each (fact, context) pair we maintain a Beta(α, β) posterior of
P(fact in next-state | context). The context is small and game-agnostic:

    context = (action_kind, action_target_color_or_None, level)

The atomic facts (already defined in predicate_pool.emit_atomic_facts)
form the state variable space. Every transition gives us a supervised
example for every observed fact — dense signal vs the sparse score
events that drive ActionScoreTable.

Key properties:
  - Closed-form Bayesian (Beta-Bernoulli conjugate). No optimization.
  - CPU-trivial: ~1000 ops per step, <20 KB memory.
  - Cross-level transfer is automatic: context tuples are small and
    recur. Level partition is via the level component of context, not
    a separate machine.
  - Information gain (sum of Beta variances) gives curiosity-driven
    exploration. With no evidence everything is variance-1/12 (uniform
    Beta(1,1)); as evidence accumulates, variance drops, exploration
    focuses.

Separate from prism/world_model.py (which predicts whole derived-states
for empowerment). This module predicts per-fact, which is what's needed
for curiosity-driven cold-start bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from biobrain.types import Action, State, action_kind

from biobrain.curiosity.predicates import Fact, emit_atomic_facts


# Context size matters. Small ⇒ contexts recur, model learns faster.
# Large ⇒ fewer false generalizations.
# We use 3-tuples: (action_kind, action_target_color, level)


def _action_context(action: Action, state: Optional[State]) -> tuple:
    """Game-agnostic small context. Cross-level by construction."""
    kind = action_kind(action)
    target_color: Optional[int] = None
    if kind == "click" and len(action) >= 3 and state is not None:
        x, y = int(action[1]), int(action[2])
        for e in state.entities:
            if (y, x) in e.region.cells:
                target_color = int(e.color)
                break
    level = int(state.level) if state is not None else 0
    if kind == "key" and len(action) >= 2:
        return (kind, int(action[1]), target_color, level)
    return (kind, None, target_color, level)


@dataclass
class BayesianWorldModel:
    """Per-(fact, context) Beta posterior of P(fact_next | context).

    Public interface:
        observe(before, action, after)   — train on one transition
        predict(state, action)            — dict[Fact, P(fact next)]
        information_gain(state, action)   — sum of Beta variances (curiosity)
        reset_game() / reset_attempt()

    The Beta-Bernoulli conjugate model gives closed-form updates and
    closed-form information gain. No optimization. CPU-trivial.
    """

    # (fact, context) → (α, β) where α counts transitions where fact appears
    # in after_facts and β counts transitions where it doesn't (given context).
    predictors: dict[tuple, tuple[float, float]] = field(default_factory=dict)
    # For computing information gain efficiently, track per-context predictor keys.
    _context_to_facts: dict[tuple, set] = field(default_factory=dict)
    decay_rate: float = 0.99
    DECAY_INTERVAL: int = 100
    _step_count: int = 0

    def reset_game(self) -> None:
        self.predictors = {}
        self._context_to_facts = {}
        self._step_count = 0

    def reset_attempt(self) -> None:
        # Predictors persist across attempts (cross-attempt memory).
        pass

    def observe(self, before: Optional[State], action: Action,
                after: State) -> None:
        """Train predictors on one transition."""
        if before is None:
            return  # need both states for supervised signal
        self._step_count += 1

        before_facts = emit_atomic_facts(None, before)
        after_facts = emit_atomic_facts(None, after)
        ctx = _action_context(action, before)

        # For every fact observed in either state, update its predictor
        # at this context. Label = whether fact appears in after.
        # This gives us dense supervision per transition.
        union_facts = before_facts | after_facts
        for fact in union_facts:
            key = (fact, ctx)
            alpha, beta = self.predictors.get(key, (0.0, 0.0))
            if fact in after_facts:
                self.predictors[key] = (alpha + 1, beta)
            else:
                self.predictors[key] = (alpha, beta + 1)
            self._context_to_facts.setdefault(ctx, set()).add(fact)

        # Periodic decay (handles non-stationarity at level transitions).
        if self._step_count % self.DECAY_INTERVAL == 0:
            for k in self.predictors:
                a, b = self.predictors[k]
                self.predictors[k] = (a * self.decay_rate, b * self.decay_rate)

    def predict(self, state: State, action: Action) -> dict[Fact, float]:
        """For each known fact in this context, P(fact in next state)."""
        ctx = _action_context(action, state)
        out: dict[Fact, float] = {}
        facts = self._context_to_facts.get(ctx, set())
        for fact in facts:
            key = (fact, ctx)
            alpha, beta = self.predictors.get(key, (0.0, 0.0))
            out[fact] = (alpha + 1) / (alpha + beta + 2)
        return out

    def sample_facts(self, state: State, action: Action, rng) -> set:
        """Thompson-sample WHICH facts will be in next state.

        For each known fact at this context, sample its posterior Beta to
        get P(fact present). Then Bernoulli-sample by that probability.
        Returns a sampled subset of predicted facts.

        Why this matters for natural explore/exploit balance:
        - When world model is uncertain (wide Beta), samples vary widely
          across calls. Resulting fact sets differ. Action scoring varies.
          Exploration emerges naturally.
        - When world model is sharp (concentrated Beta), samples are
          consistent. Resulting fact sets stable. Action scoring stable.
          Exploitation emerges naturally.
        - No lambda. No if-switch. Same posterior arithmetic does both.
        """
        ctx = _action_context(action, state)
        facts = self._context_to_facts.get(ctx, set())
        sampled = set()
        for fact in facts:
            key = (fact, ctx)
            alpha, beta = self.predictors.get(key, (0.0, 0.0))
            a, b = max(0.01, alpha + 1), max(0.01, beta + 1)
            p = rng.betavariate(a, b)  # Thompson sample of P(fact present)
            if rng.random() < p:
                sampled.add(fact)
        return sampled

    def information_gain(self, state: State, action: Action) -> float:
        """Expected entropy reduction from observing this transition.

        For Beta(α+1, β+1), variance = ab / ((a+b)²(a+b+1)).
        Total information gain = sum of variances across facts at this context.

        High variance ⇒ this (fact, context) is uncertain ⇒ trying this
        action would be informative. Curiosity signal, no reward needed.
        """
        ctx = _action_context(action, state)
        facts = self._context_to_facts.get(ctx, set())
        if not facts:
            # No history at this context — maximally uncertain.
            # Beta(1,1) variance = 1*1 / (2² * 3) = 1/12.
            # Return a uniform exploration bonus that grows with action novelty.
            return 1.0 / 12.0
        total_variance = 0.0
        for fact in facts:
            key = (fact, ctx)
            alpha, beta = self.predictors.get(key, (0.0, 0.0))
            a, b = alpha + 1, beta + 1
            total_variance += (a * b) / ((a + b) ** 2 * (a + b + 1))
        # Normalize by number of facts so context-size doesn't dominate.
        return total_variance / max(1, len(facts))

    def _observe_delta_mode(self, before: Optional[State], action: Action,
                            after: State) -> None:
        """Variant F: predict DELTA facts (transitions) rather than static
        facts in next state.

        Difference from observe(): uses emit_atomic_facts(before, after)
        which emits delta facts like count_up_color, spawn_color, etc.
        The world model learns 'given this context, which delta events
        fire on this transition?' Action selection's IG then quantifies
        uncertainty about WHAT CHANGES, not uncertainty about NEXT STATE.
        """
        if before is None:
            return
        self._step_count += 1
        # Full fact set including deltas
        full_facts = emit_atomic_facts(before, after)
        # Only the delta-kind facts: variant F focuses on transitions
        delta_kinds = {"any_change", "any_motion", "any_spawn", "any_despawn",
                       "count_up_color", "count_down_color",
                       "count_reached_zero_color", "count_first_appeared_color",
                       "spawn_color", "despawn_color"}
        ctx = _action_context(action, before)
        # For tracked context's delta facts, mark which fired
        observed_deltas = {f for f in full_facts if f[0] in delta_kinds}
        existing_deltas = {f for f in self._context_to_facts.get(ctx, set())
                           if f[0] in delta_kinds}
        all_known = observed_deltas | existing_deltas
        for fact in all_known:
            key = (fact, ctx)
            alpha, beta = self.predictors.get(key, (0.0, 0.0))
            if fact in observed_deltas:
                self.predictors[key] = (alpha + 1, beta)
            else:
                self.predictors[key] = (alpha, beta + 1)
            self._context_to_facts.setdefault(ctx, set()).add(fact)
        if self._step_count % self.DECAY_INTERVAL == 0:
            for k in self.predictors:
                a, b = self.predictors[k]
                self.predictors[k] = (a * self.decay_rate, b * self.decay_rate)

    def report(self) -> dict:
        """Diagnostic snapshot."""
        n_predictors = len(self.predictors)
        n_contexts = len(self._context_to_facts)
        # Highest-variance contexts ⇒ most-curious actions
        ig_per_ctx = []
        for ctx, facts in self._context_to_facts.items():
            total_var = 0.0
            for fact in facts:
                key = (fact, ctx)
                alpha, beta = self.predictors.get(key, (0.0, 0.0))
                a, b = alpha + 1, beta + 1
                total_var += (a * b) / ((a + b) ** 2 * (a + b + 1))
            ig_per_ctx.append((ctx, total_var / max(1, len(facts)), len(facts)))
        ig_per_ctx.sort(key=lambda x: -x[1])
        return {
            "n_predictors": n_predictors,
            "n_contexts": n_contexts,
            "n_steps": self._step_count,
            "top_curious_contexts": ig_per_ctx[:5],
        }
