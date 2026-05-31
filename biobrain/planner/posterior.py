"""biobrain.planner.posterior — the unified posterior over Function[Transition → Bool].

Stage 2 ships the minimal viable structure. Stage 5 fleshes out the
full MDL-Bayesian update across multi-entry posteriors.

The minimal version maintains:
  1. A frozen GoalPosterior seeded with Spelke archetypes.
  2. A growing MechanicPosterior tracking action → effect correlations
     under a Beta(1, 1) (Jeffreys-like) smoothed estimator.

Both are typed Function[Transition → Bool] predicates with weights.
The architecture lets us mix richer mechanic forms (kernel-program-
synthesized) into MP at later stages without changing the type.

Citations:
  - Rissanen 1978 (MDL prior over programs; weight ∝ exp(-DL)).
  - Beta(1,1) smoothing per Laplace's rule of succession.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from biobrain.types import EVENT_LEVEL_INCREASED, EVENT_SCORE_INCREASED, Action, Transition, action_kind


@dataclass(frozen=True)
class HypothesisEntry:
    """One typed predicate + posterior weight + provenance.

    Fields:
      predicate     Callable[[Transition], bool] — the Q3 unified
                    type. Returns True iff the entry's hypothesis
                    holds on this transition.

      weight        Posterior mass. Normalized within each role by
                    the update step.

      role          "goal" | "mechanic". Goals reference terminal
                    signals; mechanics describe dynamics.

      program       Canonical AST tuple for hashing + MDL.

      citation      REQUIRED for seeded archetypes; None for
                    synthesizer-derived entries.

      progress      OPTIONAL continuous progress evaluator.
                    Signature: Callable[[State, State, schemas],
                    Optional[float]] returning a log-likelihood when
                    a continuous progress signal is available, or
                    None when only the boolean predicate applies.

                    Used by decide() when the predicate evaluates
                    False but the goal admits a continuous gradient
                    (e.g., navigate-overlap → distance reduction;
                    fill-container → fraction-filled). Lets
                    decide() be GENERIC over goal classes — no
                    per-name dispatch.
    """
    predicate: Callable[[Transition], bool]
    weight: float
    role: str                       # "goal" | "mechanic"
    program: tuple
    citation: Optional[str] = None
    progress: Optional[Callable] = None


@dataclass
class Posterior:
    """The unified posterior. Goals and Mechanics differ only by role.

    Stage 2: a simple list of entries. Stage 5 adds renormalization,
    MDL prior, retention threshold.
    """
    entries: list[HypothesisEntry] = field(default_factory=list)

    def by_role(self, role: str) -> list[HypothesisEntry]:
        return [e for e in self.entries if e.role == role]


# ---------------------------------------------------------------------------
# ActionScoreTable — the minimal mechanic posterior at Stage 2
# ---------------------------------------------------------------------------
#
# Each row is structurally equivalent to a typed mechanic of the form
#   λ t. action_kind(t.action) == K → (ScoreIncreased | LevelIncreased)
# parameterized by action kind K. Under Beta(1, 1) prior, the
# posterior mean is P(score | a) ≈ (k_a + 1) / (n_a + 2).
#
# Stage 5/7 replaces this with the typed-combinator-synthesizer's
# output: mechanics whose antecedent is a richer typed predicate.
# The TYPE of the row stays the same — only the predicate grows.

@dataclass
class ActionScoreTable:
    """Observed action → score-event counts with Beta(1, 1) smoothing.

    Structurally a posterior over the typed mechanic
        `λ t. action_kind(t.action) == K → goal-event-fires`
    keyed by action signature (currently the kind tag; Stage 7
    generalizes to include click positions, undo context, etc.).
    """
    # (action_signature) → (n_observations, n_goal_events)
    counts: dict[tuple, tuple[int, int]] = field(default_factory=dict)

    @staticmethod
    def _signature(action: Action, before_state=None) -> tuple:
        """Action signature for the mechanic posterior — DERIVED, not hacked.

        - Keys: (kind, key_id, level)
        - Click: (kind, "entity", color, level) when the click lands on
                  an entity in before_state; (kind, "empty", level) else
        - Undo: (kind, level)

        Level partition: each level is a sub-game; the mechanic
        `click on color X → score` may hold on level 0 but not
        level 1. Including `level` in the signature lets the brain
        learn per-level mechanics without mixing evidence. `level`
        is a base type in the Q1 kernel (paper §3); this is a typed
        combinator application, not a hack.

        Citation chain:
          - Spelke 1990 — object-color as unit of attention.
          - paper §3 kernel — `Level` as a base type.
          - paper §4 unified predicate — mechanic predicates are
            typed predicates over Transitions; here we just include
            level in the key-axis of the signature dict.

        When before_state is None (brain has no state context), the
        level is encoded as -1 (synthetic / test fixture).
        """
        kind = action_kind(action)
        level = before_state.level if before_state is not None else -1
        if kind == "click" and len(action) >= 3 and before_state is not None:
            x, y = int(action[1]), int(action[2])
            for e in before_state.entities:
                if (y, x) in e.region.cells:
                    return (kind, "entity", e.color, level)
            return (kind, "empty", level)
        if kind == "key" and len(action) >= 2:
            return (kind, int(action[1]), level)
        return (kind, level)

    @staticmethod
    def _is_goal_event(transition: Transition) -> bool:
        """Stage-2 universal goal: any score or level event."""
        for e in transition.events:
            if e.kind in (EVENT_SCORE_INCREASED, EVENT_LEVEL_INCREASED):
                return True
        return False

    def observe(self, transition: Transition) -> None:
        """Record one observation against the action that produced it."""
        sig = self._signature(transition.action, transition.before)
        n_obs, n_goal = self.counts.get(sig, (0, 0))
        n_obs += 1
        if self._is_goal_event(transition):
            n_goal += 1
        self.counts[sig] = (n_obs, n_goal)

    def estimate(self, action: Action, state=None) -> float:
        """Beta(1,1)-smoothed P(goal event | action) in the given state.

        Returns 0.5 for never-observed signatures (Laplace prior).
        `state` is required for clicks (so we can determine which
        entity the click targets); for keys/undo it's ignored.
        """
        sig = self._signature(action, state)
        n_obs, n_goal = self.counts.get(sig, (0, 0))
        return (n_goal + 1) / (n_obs + 2)

    def observation_count(self, action: Action, state=None) -> int:
        """How many times have we tried this action signature?"""
        sig = self._signature(action, state)
        return self.counts.get(sig, (0, 0))[0]


def log_estimate(p: float, log_eps: float = -4.6) -> float:
    """log(p) clamped at log_eps to keep J finite."""
    if p <= 0.0:
        return log_eps
    return max(math.log(p), log_eps)


# ---------------------------------------------------------------------------
# Bayesian posterior update (Stage 5)
# ---------------------------------------------------------------------------

# Likelihoods derived from Beta(1,1) posterior mean after one observation:
#   correct prediction → posterior P = 2/3
#   wrong prediction   → posterior P = 1/3
# (Laplace's rule of succession; cited Beta(1,1) prior throughout.)
_LIK_CORRECT = 2.0 / 3.0
_LIK_WRONG = 1.0 / 3.0


def _is_goal_event_transition(transition: Transition) -> bool:
    for e in transition.events:
        if e.kind in (EVENT_SCORE_INCREASED, EVENT_LEVEL_INCREASED):
            return True
    return False


def _renormalize_within_roles(entries: list[HypothesisEntry]) -> list[HypothesisEntry]:
    """Renormalize weights to sum to 1.0 within each role.

    Entries with the same role have weights scaled so they sum to 1.
    If a role has total weight 0 (all entries refuted), weights are
    left unchanged — caller decides whether to re-seed.
    """
    from dataclasses import replace

    roles = {e.role for e in entries}
    out = []
    for role in roles:
        role_entries = [e for e in entries if e.role == role]
        total = sum(e.weight for e in role_entries)
        if total > 0:
            for e in role_entries:
                out.append(replace(e, weight=e.weight / total))
        else:
            out.extend(role_entries)
    # Preserve original ordering by sorting by program tuple.
    out.sort(key=lambda e: (e.role, str(e.program)))
    return out


def update_posterior(
    posterior: Posterior,
    transition: Transition,
) -> Posterior:
    """Bayesian update of the unified posterior on one transition.

    For each entry, multiply weight by likelihood:
      Beta(1,1) posterior mean after one observation:
        correct prediction → 2/3
        wrong prediction   → 1/3
      No tunable smoothing parameter — these are derived constants.

    GOALS only update on TRANSITIONS WHERE A GOAL EVENT FIRES.
    Updating on non-goal transitions would penalize goals for the
    absence of events they can't predict (their structural conditions
    don't claim to forbid non-goal transitions).

    MECHANICS update on every transition. Mechanics make claims about
    dynamics; every transition is evidence about whether a mechanic's
    consequent matches the observed events.

    After updates, weights renormalize within each role separately
    so goals and mechanics remain comparable.

    NO eviction at Stage 5 — entries can recover. Stage 8 may add
    a retention threshold under audit.
    """
    from dataclasses import replace

    actual = _is_goal_event_transition(transition)
    new_entries: list[HypothesisEntry] = []

    for e in posterior.entries:
        if e.role == "goal":
            # Skip update on transitions with no goal event.
            if not actual:
                new_entries.append(e)
                continue
            predicted = bool(e.predicate(transition))
            lik = _LIK_CORRECT if predicted else _LIK_WRONG
        else:  # mechanic (or any other role)
            predicted = bool(e.predicate(transition))
            lik = _LIK_CORRECT if predicted == actual else _LIK_WRONG
        new_entries.append(replace(e, weight=e.weight * lik))

    return Posterior(entries=_renormalize_within_roles(new_entries))
