"""biobrain.protocols — the type contracts.

biobrain is parameterized over StateT, ActionT — the brain library does not
care what specific State and Action types an environment uses, as long as
they satisfy these protocols.

The protocols are the *minimum* contract the brain library needs. The
reference implementations in `biobrain.types` satisfy these and add more
fields used by the ARC adapter. Any other adapter (synthetic test env,
future non-ARC env) can supply its own types satisfying these protocols.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Type alias for atomic facts emitted by the Encoder.
# A Fact is a hashable tuple — typically (kind, *params).
# ---------------------------------------------------------------------------
Fact = tuple


# ---------------------------------------------------------------------------
# RegionLike — a spatial region of grid cells.
# ---------------------------------------------------------------------------
@runtime_checkable
class RegionLike(Protocol):
    """A set of (row, col) cells. Used for entity regions."""
    cells: Any  # frozenset[tuple[int, int]] in the reference impl
    area: int


# ---------------------------------------------------------------------------
# EntityLike — a perceptually-segmented object.
# ---------------------------------------------------------------------------
@runtime_checkable
class EntityLike(Protocol):
    """A connected same-color blob (Spelke-grounded object)."""
    id: Any  # stable across transitions when entity persists
    color: int
    region: RegionLike
    velocity: tuple[int, int]


# ---------------------------------------------------------------------------
# StateLike — what biobrain expects to see at each step.
# ---------------------------------------------------------------------------
@runtime_checkable
class StateLike(Protocol):
    """Brain-side view of an environment state.

    Contains Spelke-grounded entities + raw observation + auxiliary signals
    (score, level, available_actions).
    """
    entities: Sequence[EntityLike]
    raw_grid: Any  # numpy.ndarray in the reference impl; treated as opaque
                   # by the brain library outside the Encoder
    grid_hash: int
    score: float
    level: int
    available_actions: tuple


# ---------------------------------------------------------------------------
# ActionLike — what the brain emits to the environment.
# ---------------------------------------------------------------------------
@runtime_checkable
class ActionLike(Protocol):
    """An action — typically a tuple (kind, *params).

    Examples in the reference impl:
        ('click', col, row)
        ('key', k)
        ('spacebar',)
        ('undo',)
    """
    # No structural requirements beyond being iterable; brain library
    # treats actions as opaque tuples and delegates kind-extraction to the
    # adapter-provided action_kind helper.


# ---------------------------------------------------------------------------
# EventLike — score events, level events, etc.
# ---------------------------------------------------------------------------
@runtime_checkable
class EventLike(Protocol):
    """A discrete event produced during a transition."""
    kind: str  # 'ScoreIncreased', 'LevelIncreased', etc.


# ---------------------------------------------------------------------------
# TransitionLike — one (before, action, after) tuple with events.
# ---------------------------------------------------------------------------
@runtime_checkable
class TransitionLike(Protocol):
    """One state transition emitted by the environment loop."""
    before: StateLike | None
    action: ActionLike | None
    after: StateLike
    events: Sequence[EventLike]


# ---------------------------------------------------------------------------
# Encoder — the State → Fact mapping.
# ---------------------------------------------------------------------------
@runtime_checkable
class Encoder(Protocol):
    """Adapter-supplied: convert State to a fact set; resolve ActionSigs.

    Two responsibilities:
      1. State → fact set (the predicate alphabet projection).
         Optionally accepts an attention_hint (set of cells where finer
         perception is requested by Salience).
      2. ActionSig → concrete Action (the DSL's abstract action → env action).
    """

    def encode(self,
               state: StateLike,
               attention_hint: frozenset | None = None,
               ) -> frozenset[Fact]:
        """Emit the fact set for this State.

        attention_hint, if provided, is a set of (row, col) cells where
        Salience has requested finer-grained perceptual features.
        Implementations may use this to emit additional sub-entity
        predicates over those cells (or ignore the hint if multi-granularity
        is not supported).
        """
        ...

    def resolve(self, action_sig: tuple, state: StateLike,
                candidates: Sequence[ActionLike]) -> ActionLike | None:
        """Convert a DSL ActionSig (e.g., ('click_on_color', 5)) to a
        concrete Action from the candidate pool. Returns None when no
        candidate matches.
        """
        ...

    def candidate_actions(self, state: StateLike) -> list[ActionLike]:
        """Enumerate the candidate actions available in this state.

        Typically: clicks at entity centroids + keys + spacebar + undo
        per state.available_actions.
        """
        ...


# ---------------------------------------------------------------------------
# Adapter — the per-environment glue.
# ---------------------------------------------------------------------------
class Adapter(Protocol):
    """Adapter contract — per-environment code that connects biobrain
    to a specific env (e.g., ARC-AGI-3).

    Required: the Encoder; the env binding (reset/step/parse).
    Optional: initial_affordance_priors (covenant-relaxation slot).
    """

    encoder: Encoder

    def initial_affordance_priors(self) -> dict[str, tuple[float, float]]:
        """Optional: per-action-class Beta priors (alpha, beta) the brain
        seeds its affordance posterior with at game start.

        Default implementation should return {} (uniform start, emergent
        shaping). Covenant-relaxed adapters may return non-trivial priors.
        """
        ...


__all__ = [
    "Fact",
    "RegionLike", "EntityLike",
    "StateLike", "ActionLike", "EventLike", "TransitionLike",
    "Encoder", "Adapter",
]
