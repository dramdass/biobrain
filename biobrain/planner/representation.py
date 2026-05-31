"""prism.representation — Stage B: predictive-sufficiency refinement.

The reviewer's Round-2 design (§2 + §C2 + Q4):

  Group transitions by (derived_state, action). Where outcomes
  diverge under a deterministic env, the representation is
  INSUFFICIENT → SPLIT: add a distinguishing feature. Where a
  feature never affects the future → MERGE it out. Halt when
  transitions are deterministic under the representation.

The same consistency test (does (s, a) deterministically map to a'?)
serves TWO masters (Round 2 Q4):
  1. Refinement: inconsistency triggers a SPLIT.
  2. Pruning safety: pruning is safe ONLY in consistent regions.

Roles (is_agent, is_target, is_container, is_filler) become
OUTPUTS of this loop — not a separate module gated on goal events.
This breaks the bootstrap circle: rep refinement halts on env
determinism, which is bounded; goal identification didn't halt
at all without scoring.

Citation chain:
  - Predictive sufficiency: Markov decision processes, Bellman
    1957; sufficient-statistic framing standard.
  - Split/merge by determinism: Givan-Dean-Greig 2003
    (equivalence and model reduction for MDPs).
  - The "stochastic-looking → refinement cue" framing in a
    deterministic env: Round-2 reviewer's §3 elaboration.

This module is Stage-B-minimum: the consistency test + a
DerivedState type that exposes the refinement state. Stage B+
will fold this into SchemaState's `to_registry()` so role tags
emerge as derived outputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from biobrain.types import Entity, State, Transition, action_kind


# ---------------------------------------------------------------------------
# DerivedState — the rep loop's output (Round 2 Q2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivedState:
    """A representation-loop output: a hashable signature of the
    state at the CURRENT refinement level.

    `features` is a sorted tuple of typed feature descriptors. Each
    descriptor is a tuple `(feature_name, value, ...)`. The set of
    feature_names defines the refinement level; split adds a name,
    merge removes one.

    Two states with the same DerivedState are considered EQUIVALENT
    under the current representation — their behavior under any
    action should match.
    """
    features: tuple[tuple, ...]

    def __hash__(self) -> int:
        return hash(self.features)


# ---------------------------------------------------------------------------
# Feature extractors — the building blocks of refinement
# ---------------------------------------------------------------------------
#
# Each extractor is a typed function State → tuple[feature_descriptor, ...].
# The rep loop starts with the coarsest feature set and SPLITS by adding
# extractors when inconsistency demands it.

def _entity_color_set(state: State) -> tuple[tuple, ...]:
    """Sorted multiset of entity colors. The coarsest perceptual feature."""
    colors = sorted(e.color for e in state.entities)
    return (("entity_colors", tuple(colors)),)


def _entity_centroid_quantized(state: State, q: int = 8) -> tuple[tuple, ...]:
    """Per-entity (color, quantized centroid). Adds spatial structure."""
    out = []
    for e in state.entities:
        if not e.region.cells:
            continue
        rows = [c[0] for c in e.region.cells]
        cols = [c[1] for c in e.region.cells]
        cr = (sum(rows) // len(rows)) // q
        cc = (sum(cols) // len(cols)) // q
        out.append(("centroid", int(e.color), int(cr), int(cc)))
    out.sort()
    return tuple(out)


def _entity_areas(state: State) -> tuple[tuple, ...]:
    """Sorted multiset of (color, area) — distinguishes objects of
    same color but different size."""
    out = sorted((int(e.color), int(e.region.area)) for e in state.entities)
    return (("areas", tuple(out)),)


def _level(state: State) -> tuple[tuple, ...]:
    """The level dimension. Always informative across ARC-AGI-3 levels."""
    return (("level", int(state.level)),)


# Ordered registry: each entry is (name, extractor, default_active).
# The rep loop activates them in order as needed.
_FEATURE_EXTRACTORS = [
    ("entity_colors",   _entity_color_set,         True),
    ("level",           _level,                    True),
    ("centroid_quant",  _entity_centroid_quantized, True),
    ("entity_areas",    _entity_areas,             False),  # opt-in via split
]


# ---------------------------------------------------------------------------
# RepLoop — incremental refinement state
# ---------------------------------------------------------------------------

@dataclass
class RepLoop:
    """Tracks per-(derived_state, action) outcome histories and
    decides when to split.

    For each (derived_state, action), we record the set of distinct
    `after` derived-states observed. If the set ever has > 1 entry,
    the region is INCONSISTENT and a split is warranted (refine the
    feature set).

    SPLIT: activate the next feature extractor in
    `_FEATURE_EXTRACTORS`; rebuild derived states from history.
    Bounded: the extractor list is finite.

    The reviewer's design also includes MERGE (remove a feature that
    never affects the future). Deferred to Stage B+ — split alone
    is the more important direction.

    The rep loop's `region_consistency` method exposes the
    consistency check for the pruning module (Round 2 Q4).
    """
    active_features: set[str] = field(default_factory=lambda: {
        name for (name, _, default) in _FEATURE_EXTRACTORS if default
    })

    # Per (derived_state, action_signature) → set of distinct
    # post-derived-states observed.
    outcomes: dict[tuple, set[DerivedState]] = field(
        default_factory=lambda: defaultdict(set))

    n_observations: int = 0
    n_splits: int = 0

    def derive(self, state: State) -> DerivedState:
        """Apply active extractors and return the derived state."""
        features: list[tuple] = []
        for (name, fn, _) in _FEATURE_EXTRACTORS:
            if name in self.active_features:
                features.extend(fn(state))
        return DerivedState(features=tuple(sorted(features)))

    def update(self, transition: Transition) -> bool:
        """Record one transition. Returns True if a SPLIT was triggered."""
        self.n_observations += 1
        s_before = self.derive(transition.before)
        s_after = self.derive(transition.after)
        a_sig = self._action_signature(transition.action)
        key = (s_before, a_sig)
        self.outcomes[key].add(s_after)
        # Inconsistency → split.
        if len(self.outcomes[key]) > 1:
            return self._try_split()
        return False

    def _try_split(self) -> bool:
        """Activate the next inactive feature extractor.

        Returns True if a split was made; False if the extractor
        list is exhausted (we've fully refined and the region's
        residual non-determinism is from hidden state — synth
        territory).
        """
        for (name, _, _) in _FEATURE_EXTRACTORS:
            if name not in self.active_features:
                self.active_features.add(name)
                self.n_splits += 1
                # Rebuild outcomes under the new feature set.
                self._rebuild()
                return True
        return False

    def _rebuild(self) -> None:
        """After a split, the previous derived-state cache is
        invalidated. Stage-B-minimum: clear outcomes; rebuild as new
        observations arrive. Stage-B+ could replay the history.
        """
        self.outcomes.clear()

    def region_consistency(self, state: State, action) -> str:
        """For Round-2 Q4: is this (derived_state, action) region
        consistent under the current representation?

          'consistent'   - same key has only one post-state observed
          'inconsistent' - same key has multiple post-states
          'unobserved'   - region not seen yet
        """
        key = (self.derive(state), self._action_signature(action))
        outs = self.outcomes.get(key)
        if not outs:
            return "unobserved"
        return "consistent" if len(outs) == 1 else "inconsistent"

    def n_distinct_derived_states(self) -> int:
        """For Gate 2: count of distinct derived-states observed."""
        seen = set()
        for (s_before, _), s_afters in self.outcomes.items():
            seen.add(s_before)
            seen.update(s_afters)
        return len(seen)

    def summary(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "active_features": sorted(self.active_features),
            "n_splits": self.n_splits,
            "n_keys": len(self.outcomes),
            "n_distinct_derived_states": self.n_distinct_derived_states(),
        }

    @staticmethod
    def _action_signature(action) -> tuple:
        kind = action_kind(action)
        if kind == "key" and len(action) >= 2:
            return (kind, int(action[1]))
        if kind == "click" and len(action) >= 3:
            return (kind, int(action[1]) // 8, int(action[2]) // 8)
        return (kind,)
