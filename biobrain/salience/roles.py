"""biobrain.salience.roles — the 10-role Spelke-grounded catalogue.

Each role is identified by a distinctive causal signature observable
through transitions. Likelihood functions are Spelke-grounded heuristics,
not learned, not per-game tuned.

# RL-TODO: likelihood weights could be learned from per-game scoring
# correlations once we have validation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    SELECTOR = "selector"
    CURSOR = "cursor"
    PAINTER = "painter"
    TARGET = "target"
    TOGGLE = "toggle"
    COUNTER = "counter"
    BARRIER = "barrier"
    CONTAINER = "container"
    STATIC = "static"
    UNKNOWN = "unknown"


ROLE_CATALOGUE: tuple[Role, ...] = tuple(Role)


# RL-TODO: derive from observed distribution. K=5 is conservative
# (enough to discriminate between adjacent roles in our 10-role catalogue
# without overcommitting on a single observation).
ROLE_DISCOVERY_K = 5


# RL-TODO: derive from per-game Critic-reference distribution. Currently:
# 1 reference per 5 observations is treated as a strong TARGET signal.
TARGET_REFERENCE_RATE_DENOMINATOR = 5


@dataclass
class RoleSignature:
    """Per-entity causal counters used to infer role."""
    n_observations: int = 0
    # Click-on-self counters
    clicked_on_count: int = 0
    clicked_caused_self_change: int = 0
    clicked_caused_other_change: int = 0  # change >K cells elsewhere
    clicked_caused_global_change: int = 0  # mode-shift sized change
    # Key-action counters
    translated_under_key_count: int = 0
    # Persistence
    persistence: float = 1.0  # fraction of transitions where entity present
    was_removed_on_click: int = 0  # times entity disappeared after click on it
    # Goal-reference
    referenced_by_distance_goals: int = 0  # times appeared in active goal's
                                            # relevant_cells (Critic side)


def role_likelihood(sig: RoleSignature) -> dict[Role, float]:
    """Return normalized posterior over all 10 roles.

    Each role's score is a Spelke-grounded heuristic over the causal
    counters. UNKNOWN's score is high when n_observations is low.
    """
    n = max(1, sig.n_observations)
    clicked = max(1, sig.clicked_on_count)
    scores: dict[Role, float] = {}

    # UNKNOWN: dominant when undersampled
    scores[Role.UNKNOWN] = max(0.0, 1.0 - sig.n_observations / ROLE_DISCOVERY_K)

    # SELECTOR: clicks cause OTHER changes; not self changes
    other_rate = sig.clicked_caused_other_change / clicked
    self_rate = sig.clicked_caused_self_change / clicked
    scores[Role.SELECTOR] = other_rate * (1.0 - self_rate) * sig.persistence

    # CURSOR: translates under key actions
    key_rate = sig.translated_under_key_count / n
    scores[Role.CURSOR] = key_rate

    # PAINTER: distinct from selector — clicking it later produces visible
    # changes (deferred causality). v0 heuristic: high both other AND self.
    scores[Role.PAINTER] = other_rate * self_rate

    # TARGET: high persistence + referenced by Critic goals
    scores[Role.TARGET] = sig.persistence * min(
        1.0, sig.referenced_by_distance_goals / max(
            1, sig.n_observations // TARGET_REFERENCE_RATE_DENOMINATOR))

    # TOGGLE: clicking flips state (high clicked_caused_self with bounded
    # entity-state diversity — v0 approximation: high self-change rate
    # with persistence intact)
    scores[Role.TOGGLE] = self_rate * sig.persistence * 0.5
    # RL-TODO(TOGGLE): 0.5 dampens vs SELECTOR (both fire on click-self).
    # Replace with learned weight once role-attribution accuracy is measurable.

    # COUNTER: changes under non-click actions; monotone signature is hard
    # to express without temporal history — v0 approximation: high
    # translated rate AND high persistence
    scores[Role.COUNTER] = key_rate * sig.persistence * 0.3
    # RL-TODO(COUNTER): 0.3 dampens vs CURSOR (same numerator). Distinguishing
    # COUNTER from CURSOR needs temporal-history features we don't yet emit.

    # BARRIER: disappears on click
    barrier_rate = sig.was_removed_on_click / clicked
    scores[Role.BARRIER] = barrier_rate * (1.0 - sig.persistence)

    # CONTAINER: v0 stub — needs region-overlap tracking we haven't built
    scores[Role.CONTAINER] = 0.0

    # STATIC: nothing changes ever; high persistence; not referenced.
    # Being referenced by a distance goal disqualifies STATIC (that's
    # TARGET's territory) — collapse the score to near-zero when referenced.
    no_change_rate = 1.0 - (other_rate + self_rate + key_rate)
    scores[Role.STATIC] = max(0.0, no_change_rate) * sig.persistence * (
        1.0 if sig.referenced_by_distance_goals == 0 else 0.1)
    # RL-TODO(STATIC): 0.1 is the "referenced-disqualifier" — if a Critic
    # goal references this entity, it's almost certainly a TARGET, not STATIC.
    # Replace with a learned gate once Critic-reference frequency is calibrated.

    # Normalize
    total = sum(scores.values())
    if total <= 0:
        # Degenerate — assign full mass to UNKNOWN
        scores = {r: 0.0 for r in ROLE_CATALOGUE}
        scores[Role.UNKNOWN] = 1.0
        return scores
    return {r: v / total for r, v in scores.items()}


def assign_role(sig: RoleSignature) -> Role:
    """Pick the highest-likelihood role. Returns UNKNOWN if n_observations < K."""
    if sig.n_observations < ROLE_DISCOVERY_K:
        return Role.UNKNOWN
    likelihoods = role_likelihood(sig)
    # Tiebreak: prefer SELECTOR over UNKNOWN; otherwise alphabetical.
    return max(likelihoods, key=lambda r: (likelihoods[r],
                                            r != Role.UNKNOWN, r.value))


__all__ = ["Role", "RoleSignature", "ROLE_CATALOGUE",
           "ROLE_DISCOVERY_K", "role_likelihood", "assign_role"]
