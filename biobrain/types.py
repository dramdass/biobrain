"""biobrain.types — the brain-agnostic typed substrate.

These are the only types the BrainEngine interface exposes.
Every type is hashable and immutable (NamedTuple or frozen).

Cross-references:
- Companion paper §3 (the kernel's 12 base types)
- Technical design §3 (data structures)
- Q1 design session: docs/DESIGN-Q1-KERNEL-BOUNDARY.md
"""

from __future__ import annotations

from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Primitive aliases
# ---------------------------------------------------------------------------

Color = int          # 0..15 per ARC-AGI-3 spec
Cell = tuple[int, int]  # (row, col)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
#
# Actions are tuples so they are hashable, picklable, and trivially
# comparable. The first element is the action's `kind`; the remainder
# are kind-specific parameters.
#
#   ("key", k)           — k in {0, 1, 2, 3, 4} per spec
#   ("undo",)            — undo last action
#   ("click", x, y)      — click at pixel (x, y)
#
# We deliberately do NOT use an Action class with subtypes — flat tuples
# are easier for the brain to enumerate and serialize, and the arena
# does the dispatch in env.py.

Action = tuple[Any, ...]


def action_key(k: int) -> Action:
    """Build a key-press action. k ∈ {0,1,2,3,4}."""
    if not 0 <= k <= 4:
        raise ValueError(f"key index out of range: {k}")
    return ("key", k)


def action_undo() -> Action:
    """Build an undo action."""
    return ("undo",)


def action_click(x: int, y: int) -> Action:
    """Build a click action at pixel (x, y)."""
    return ("click", int(x), int(y))


def action_kind(a: Action) -> str:
    """Extract the kind tag of an action."""
    return str(a[0])


# ---------------------------------------------------------------------------
# Region — a connected set of cells with a bounding box
# ---------------------------------------------------------------------------

class Region(NamedTuple):
    """A connected set of grid cells.

    Cells stored as a frozenset for hash stability and set operations.
    Bounding box is (row_min, col_min, row_max, col_max), inclusive.

    Cites: Spelke 1990 (cohesion principle — objects move as wholes).
    """

    cells: frozenset[Cell]
    bbox: tuple[int, int, int, int]

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def area(self) -> int:
        return len(self.cells)


# ---------------------------------------------------------------------------
# Entity — a persistent Region with stable identity and velocity
# ---------------------------------------------------------------------------

class Entity(NamedTuple):
    """A grid object with stable identity across frames.

    `id` is matched by the perception layer across consecutive frames
    via region overlap + color + size heuristics. `velocity` is the
    (drow, dcol) displacement since the previous frame, or (0, 0) at
    spawn.

    Cites: Spelke 1990 (continuity & persistence).
    """

    id: int
    region: Region
    color: Color
    velocity: Cell


# ---------------------------------------------------------------------------
# Event — a discrete state-change observation
# ---------------------------------------------------------------------------

class Event(NamedTuple):
    """A discrete observation tag attached to a Transition.

    `kind` is a short string (e.g., "ScoreIncreased", "LevelIncreased",
    "ContactGained", "ColorChanged"). `payload` is arbitrary
    kind-specific data. Brains read events to update posteriors but
    do not synthesize events themselves.

    Cites: Zacks & Tversky 2001 (event structure in perception).
    """

    kind: str
    payload: tuple[tuple[str, Any], ...]  # tuple of (key, value) for hashability

    @classmethod
    def make(cls, kind: str, **payload: Any) -> "Event":
        """Construct an Event with named payload fields."""
        return cls(kind=kind, payload=tuple(sorted(payload.items())))


# Standard event-kind tags. Brains MAY define their own additional
# kinds inside their internal state, but these are the arena-emitted
# tags every brain can rely on.
EVENT_SCORE_INCREASED = "ScoreIncreased"
EVENT_LEVEL_INCREASED = "LevelIncreased"
EVENT_CONTACT_GAINED = "ContactGained"
EVENT_CONTACT_LOST = "ContactLost"
EVENT_COLOR_CHANGED = "ColorChanged"
EVENT_ENTITY_SPAWNED = "EntitySpawned"
EVENT_ENTITY_DESPAWNED = "EntityDespawned"


# ---------------------------------------------------------------------------
# State — a snapshot of the world
# ---------------------------------------------------------------------------

class State(NamedTuple):
    """A snapshot of the world at one time step.

    `entities` is the parsed object set with stable IDs.
    `score` and `level` are the environment's auxiliary signals.
    `grid_hash` is a stable hash of the raw grid for state-graph indexing.
    `raw_grid` is the 64x64 uint8 numpy array; brains MAY ignore it.
    `available_actions` is the set of action IDs (1..5 keys, 6 click,
        7 undo) the env permits at this step. Brains MUST select from
        this set; emitting unavailable actions is a covenant violation.
    """

    entities: frozenset[Entity]
    score: int
    level: int
    grid_hash: int
    raw_grid: Any  # numpy.ndarray, 64x64 uint8
    available_actions: tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Transition — a (before, action, after, events) tuple
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """A single state transition observed during play.

    Brains consume Transitions through `BrainEngine.observe`. The arena
    emits a Transition after every step using `before = prev_state`,
    `after = current_state`, and the events derived by the arena's
    event-detection module.
    """

    before: State
    action: Action
    after: State
    events: frozenset[Event]


# ---------------------------------------------------------------------------
# ComputeBudget — remaining resources within an attempt
# ---------------------------------------------------------------------------

class ComputeBudget(NamedTuple):
    """Remaining compute budget at the current decision point.

    `actions_remaining` decreases by 1 per action.
    `time_remaining_ms` is wall-clock budget for the current attempt.
    `attempts_remaining` is the number of attempts left in the game.

    The horizon function `H(B)` in `prism/decide.py` consumes this to
    pick the planning depth.
    """

    actions_remaining: int
    time_remaining_ms: int
    attempts_remaining: int

    def consume(self, action_cost: int = 1, wall_ms: int = 0) -> "ComputeBudget":
        """Return a new budget with one action's cost subtracted."""
        return ComputeBudget(
            actions_remaining=max(0, self.actions_remaining - action_cost),
            time_remaining_ms=max(0, self.time_remaining_ms - wall_ms),
            attempts_remaining=self.attempts_remaining,
        )
