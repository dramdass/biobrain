"""biobrain.curiosity.predicates — combinatorial predicate posterior, no hardcoded events.

Per docs/PROXY-COMBINATORIAL-DESIGN.md + user direction:
"i dont want specific events. i want the ability for all combos and
never have hardcoded proxies."

The architecture:
  1. emit_atomic_facts(before, after) — universal fact extractor over
     Spelke primitives (Object, Number, Place, Continuity, State).
     NO hardcoded events. NO thresholds chosen per game.
  2. PredicatePool — tracks Beta(α, β) posterior per (predicate, level).
     - Atomic predicates: spawned on first observation.
     - 2-conjunctions: spawned from recent facts when a score event fires.
     - LR-based dormancy: predicates with n_fires ≥ MIN_N and LR < 1.2
       are tracked but don't contribute to proxy_score.
     - LRU eviction when pool exceeds MAX_POOL_SIZE.

The only architectural commitments (all Spelke-justifiable, all game-agnostic):
  - Entity cohesion (object primitive): entities have id, color, size, position, velocity
  - Place quadrant: 4x4 spatial discretization of the 64x64 grid
  - Size buckets: tiny/small/medium/large/huge by area
  - Color: 0..15 as observed
  - Counts: number of entities per attribute value
  - Level: game level integer

Everything else (which atoms/conjunctions matter) emerges from Bayesian
posterior, not from architecture.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from biobrain.types import State


# ---------------------------------------------------------------------------
# Atomic fact extractor
# ---------------------------------------------------------------------------

# A Fact is a tuple: (kind, *params). Kind is the predicate template name;
# params are observed values. Both are hashable for set/dict use.
Fact = tuple


def _quadrant_of(cells) -> int:
    """4x4 grid quadrant of centroid → 0..15. Place primitive."""
    if not cells:
        return 0
    cy = sum(c[0] for c in cells) / len(cells)
    cx = sum(c[1] for c in cells) / len(cells)
    by = min(3, max(0, int(cy // 16)))
    bx = min(3, max(0, int(cx // 16)))
    return by * 4 + bx


def _size_bucket(area: int) -> str:
    """Spelke object-size: bounded, discrete classification."""
    if area <= 2: return "tiny"
    if area <= 8: return "small"
    if area <= 32: return "medium"
    if area <= 128: return "large"
    return "huge"


def emit_atomic_facts(before: Optional[State], after: State) -> set[Fact]:
    """Emit all observable atomic facts about this transition.

    Game-agnostic. Derived from Spelke primitives. NO hardcoded events.

    Returns: set of Fact tuples. Bounded by (entity_count × attribute_count).
    """
    facts: set[Fact] = set()

    # ENTITY-LEVEL (after state) — object primitives.
    # Per derivation principle: each entity's projection onto Spelke axes
    # (Color, Size, Place), plus pairwise joints (Color×Place, Color×Size)
    # for object-identity queries the Critic needs (symmetry, cohesion).
    size_counts: dict[str, int] = {}
    quad_counts: dict[int, int] = {}
    for e in after.entities:
        c = int(e.color)
        s = _size_bucket(e.region.area)
        q = _quadrant_of(e.region.cells)
        facts.add(("entity_color", c))
        facts.add(("entity_size", s))
        facts.add(("entity_quadrant", q))
        # JOINT predicates — required for relational Critic queries
        # (e.g., Symmetry pairing quadrants by color, Cohesion clustering
        # by color×place). Bounded: 16×16=256 max, ~20 active per state.
        facts.add(("entity_color_quadrant", c, q))
        facts.add(("entity_color_size", c, s))
        size_counts[s] = size_counts.get(s, 0) + 1
        quad_counts[q] = quad_counts.get(q, 0) + 1
        if e.velocity != (0, 0):
            facts.add(("any_motion",))

    # COUNT (Number primitive) — projection onto each Spelke axis.
    # Symmetry with entity-projections: every entity_X axis has a count_X
    # cardinality fact.
    color_counts: dict[int, int] = {}
    for e in after.entities:
        c = int(e.color)
        color_counts[c] = color_counts.get(c, 0) + 1
    for color, n in color_counts.items():
        facts.add(("count_color", color, n))
    for size, n in size_counts.items():
        facts.add(("count_size", size, n))
    for quad, n in quad_counts.items():
        facts.add(("count_quadrant", quad, n))
    facts.add(("total_entities", len(after.entities)))

    # DELTA / CONTINUITY (only when before exists)
    if before is not None:
        prev_ids = {e.id for e in before.entities}
        curr_ids = {e.id for e in after.entities}
        # Spawn / despawn (general)
        spawned_ids = curr_ids - prev_ids
        despawned_ids = prev_ids - curr_ids
        if spawned_ids:
            facts.add(("any_spawn",))
            for sid in spawned_ids:
                for e in after.entities:
                    if e.id == sid:
                        facts.add(("spawn_color", int(e.color)))
                        break
        if despawned_ids:
            facts.add(("any_despawn",))
            for sid in despawned_ids:
                for e in before.entities:
                    if e.id == sid:
                        facts.add(("despawn_color", int(e.color)))
                        break
        # Count deltas per color
        prev_color_counts: dict[int, int] = {}
        for e in before.entities:
            c = int(e.color)
            prev_color_counts[c] = prev_color_counts.get(c, 0) + 1
        for color in set(prev_color_counts) | set(color_counts):
            before_n = prev_color_counts.get(color, 0)
            after_n = color_counts.get(color, 0)
            if after_n > before_n:
                facts.add(("count_up_color", color))
            elif after_n < before_n:
                facts.add(("count_down_color", color))
            if before_n > 0 and after_n == 0:
                facts.add(("count_reached_zero_color", color))
            if before_n == 0 and after_n > 0:
                facts.add(("count_first_appeared_color", color))
        # Frame change (most-general delta)
        if before.grid_hash != after.grid_hash:
            facts.add(("any_change",))

    # GLOBAL state — level partition
    facts.add(("level", int(after.level)))

    return facts


# ---------------------------------------------------------------------------
# PredicateHypothesis — atomic or conjunctive
# ---------------------------------------------------------------------------

@dataclass
class PredicateHypothesis:
    """A predicate (atomic fact or conjunction of facts) with Beta posterior."""
    predicate: frozenset                 # frozenset of facts; size 1 = atomic
    level: int                            # per-level partition
    alpha: float = 0.0                    # n times predicate fired AND score in next K
    beta: float = 0.0                     # n times predicate fired AND no score in next K
    n_fires: int = 0                      # total times predicate has been observed

    @property
    def posterior_mean(self) -> float:
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    @property
    def is_atomic(self) -> bool:
        return len(self.predicate) == 1


# ---------------------------------------------------------------------------
# PredicatePool
# ---------------------------------------------------------------------------

@dataclass
class PredicatePool:
    """Bayesian posterior over atomic + combinatorial predicates.

    No fixed event set. Atomic predicates emerge from emit_atomic_facts.
    Conjunctions spawn on score events from recent active facts.
    LR-based dormancy + LRU eviction keeps pool bounded.

    Public interface (matches old ProxyPool):
        observe(prev, after, scored) — update from one transition
        proxy_score(facts_fired, level) — [0, 1] dense reward signal
        reset_attempt() / reset_game()
        report() — diagnostic snapshot
    """

    k_window: int = 10
    decay_rate: float = 0.99
    MAX_POOL_SIZE: int = 5000
    MAX_SPAWN_PER_SCORE: int = 200
    MIN_N_FOR_LR: int = 5         # min observations before LR is trusted
    MIN_ALPHA_FOR_ACTIVE: float = 1.0  # require ≥1 positive observation
    LR_DORMANT_THRESHOLD: float = 1.2
    # Bayesian-driven depth promotion: extend high-LR k-conjunctions to (k+1)
    ALPHA_PROMOTE_THRESHOLD: float = 2.0   # need ≥2 positives to promote
    LR_PROMOTE_THRESHOLD: float = 2.0      # need LR > 2.0 to promote
    MAX_CONJUNCTION_DEPTH: int = 5         # cap conjunction size

    hypotheses: dict[tuple[frozenset, int], PredicateHypothesis] = field(default_factory=dict)
    _window: deque = field(default_factory=lambda: deque(maxlen=15))
    _baseline_alpha: float = 0.0
    _baseline_beta: float = 0.0
    _step_count: int = 0

    def reset_game(self) -> None:
        self.hypotheses = {}
        self._window = deque(maxlen=self.k_window + 5)
        self._baseline_alpha = 0.0
        self._baseline_beta = 0.0
        self._step_count = 0

    def reset_attempt(self) -> None:
        self._window = deque(maxlen=self.k_window + 5)

    def observe(self, before: Optional[State], after: State, scored: bool) -> None:
        self._step_count += 1
        facts = emit_atomic_facts(before, after)
        level = int(after.level)

        # Append to window (rolling K-step history)
        self._window.append({
            "facts": facts, "scored": scored, "level": level,
        })

        # Credit-assign the K-step-ago entry: did score fire in the next K steps?
        if len(self._window) > self.k_window:
            oldest = self._window[0]
            future = list(self._window)[1:]
            score_in_window = any(e["scored"] for e in future)
            old_level = oldest["level"]
            old_facts = oldest["facts"]

            # Update baseline (priors for LR computation)
            if score_in_window:
                self._baseline_alpha += 1
            else:
                self._baseline_beta += 1

            # Update ATOMIC hypotheses for each fact in oldest
            for fact in old_facts:
                self._update(frozenset([fact]), old_level, score_in_window)

            # Update existing COMPOSITE hypotheses (their predicate ⊆ oldest facts)
            for key, h in list(self.hypotheses.items()):
                if h.level != old_level: continue
                if h.is_atomic: continue
                if h.predicate.issubset(old_facts):
                    h.n_fires += 1
                    if score_in_window:
                        h.alpha += 1
                    else:
                        h.beta += 1

        # SPAWN 2-conjunctions when score fires (Bayesian-evidence-driven generation)
        if scored:
            recent_facts: set = set()
            for entry in list(self._window):
                recent_facts.update(entry["facts"])
            spawned = 0
            for pair in combinations(sorted(recent_facts), 2):
                if spawned >= self.MAX_SPAWN_PER_SCORE:
                    break
                key = (frozenset(pair), level)
                if key not in self.hypotheses:
                    self.hypotheses[key] = PredicateHypothesis(
                        predicate=frozenset(pair),
                        level=level,
                    )
                    spawned += 1
            # PROMOTE: extend high-LR k-conjunctions to (k+1)-conjunctions.
            # Bayesian-driven depth expansion: only promote predicates with
            # genuine positive evidence (α ≥ ALPHA_PROMOTE_THRESHOLD AND
            # LR > LR_PROMOTE_THRESHOLD). Bounded by remaining spawn budget.
            for depth in range(2, self.MAX_CONJUNCTION_DEPTH):
                if spawned >= self.MAX_SPAWN_PER_SCORE:
                    break
                promotable = [
                    h for h in self.hypotheses.values()
                    if h.level == level
                    and len(h.predicate) == depth
                    and h.alpha >= self.ALPHA_PROMOTE_THRESHOLD
                    and h.n_fires >= self.MIN_N_FOR_LR
                    and self.likelihood_ratio_for(h) > self.LR_PROMOTE_THRESHOLD
                ]
                for h_k in promotable:
                    for fact in recent_facts - h_k.predicate:
                        if spawned >= self.MAX_SPAWN_PER_SCORE:
                            break
                        new_pred = h_k.predicate | {fact}
                        key = (frozenset(new_pred), level)
                        if key not in self.hypotheses:
                            self.hypotheses[key] = PredicateHypothesis(
                                predicate=frozenset(new_pred),
                                level=level,
                            )
                            spawned += 1

        # Periodic symmetric decay (per HP3 finding: handles non-stationarity)
        if self._step_count % 50 == 0:
            for h in self.hypotheses.values():
                h.alpha *= self.decay_rate
                h.beta *= self.decay_rate
            self._baseline_alpha *= self.decay_rate
            self._baseline_beta *= self.decay_rate

        # LRU prune if pool exceeds cap
        if len(self.hypotheses) > self.MAX_POOL_SIZE:
            self._prune_lru()

    def _update(self, predicate: frozenset, level: int, scored: bool) -> None:
        key = (predicate, level)
        h = self.hypotheses.get(key)
        if h is None:
            h = PredicateHypothesis(predicate=predicate, level=level)
            self.hypotheses[key] = h
        h.n_fires += 1
        if scored:
            h.alpha += 1
        else:
            h.beta += 1

    def _prune_lru(self) -> None:
        """Evict 10% of hypotheses with lowest LR among those with enough obs."""
        evictable = [
            (k, self.likelihood_ratio_for(h))
            for k, h in self.hypotheses.items()
            if h.n_fires >= self.MIN_N_FOR_LR
        ]
        evictable.sort(key=lambda kv: kv[1])  # lowest LR first
        n_evict = max(1, len(self.hypotheses) // 10)
        for k, _ in evictable[:n_evict]:
            del self.hypotheses[k]

    def baseline(self) -> float:
        """Bayesian-smoothed P(score in next K window)."""
        return (self._baseline_alpha + 1) / (
            self._baseline_alpha + self._baseline_beta + 2
        )

    def likelihood_ratio_for(self, h: PredicateHypothesis) -> float:
        b = self.baseline()
        return h.posterior_mean / b if b > 0 else 1.0

    def likelihood_ratio(self, predicate: frozenset, level: int) -> float:
        h = self.hypotheses.get((predicate, level))
        if h is None or h.n_fires < self.MIN_N_FOR_LR:
            return 1.0
        return self.likelihood_ratio_for(h)

    def _is_active(self, h: PredicateHypothesis) -> bool:
        """Active predicate: enough observations AND ≥1 positive AND LR > threshold.

        The α ≥ 1 requirement prevents Laplace-prior artifacts where
        predicates with zero positives appear high-LR when the baseline
        is tiny.
        """
        if h.n_fires < self.MIN_N_FOR_LR: return False
        if h.alpha < self.MIN_ALPHA_FOR_ACTIVE: return False
        return self.likelihood_ratio_for(h) > self.LR_DORMANT_THRESHOLD

    def proxy_score(self, fired_facts: set, level: int) -> float:
        """Compute proxy reward from atomic + composite hypotheses that fired.

        Sums (LR - 1) clamped to [0, 2] over ACTIVE predicates whose
        predicate is a subset of fired_facts. Then soft-caps to [0, 1].
        """
        score, _ = self.proxy_score_with_n_eff(fired_facts, level)
        return score

    def proxy_score_with_n_eff(
        self, fired_facts: set, level: int
    ) -> tuple[float, float]:
        """Return (score, n_eff) where n_eff is the effective sample size
        of the indirect prior — for hierarchical Bayesian combination.

        n_eff = number of active hypotheses that fired on this transition.
        This gives the bandit framework a data-derived prior strength
        instead of a hand-tuned weight: when many predicates support the
        prediction, indirect prior is strong; when few support, it's weak.
        """
        score = 0.0
        n_contributing = 0
        # Atomic contributions
        for fact in fired_facts:
            pred = frozenset([fact])
            h = self.hypotheses.get((pred, level))
            if h is None or not self._is_active(h):
                continue
            lr = self.likelihood_ratio_for(h)
            score += min(lr - 1.0, 2.0)
            n_contributing += 1
        # Composite contributions (predicate ⊆ fired_facts)
        for h in self.hypotheses.values():
            if h.level != level: continue
            if h.is_atomic: continue
            if not self._is_active(h): continue
            if not h.predicate.issubset(fired_facts): continue
            score += min(self.likelihood_ratio_for(h) - 1.0, 2.0)
            n_contributing += 1
        return min(1.0, score / 3.0), float(n_contributing)

    def report(self, top_n: int = 20, active_only: bool = True) -> dict:
        """Diagnostic snapshot — top hypotheses by LR.

        active_only filters to predicates with α >= 1 (avoid Laplace artifacts).
        """
        rows = []
        for (pred, lvl), h in self.hypotheses.items():
            if h.n_fires < self.MIN_N_FOR_LR:
                continue
            if active_only and h.alpha < self.MIN_ALPHA_FOR_ACTIVE:
                continue
            lr = self.likelihood_ratio_for(h)
            rows.append({
                "predicate": tuple(sorted(pred)),
                "level": lvl,
                "is_atomic": h.is_atomic,
                "alpha": h.alpha,
                "beta": h.beta,
                "n_fires": h.n_fires,
                "lr": lr,
            })
        rows.sort(key=lambda r: -r["lr"])
        return {
            "baseline": self.baseline(),
            "n_hypotheses": len(self.hypotheses),
            "n_with_data": len(rows),
            "n_active": sum(1 for r in rows
                            if r["lr"] > self.LR_DORMANT_THRESHOLD),
            "top": rows[:top_n],
        }
