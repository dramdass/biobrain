"""biobrain.perception.perceive — raw 64×64 grid → typed State.

Pipeline:
  1. Find background color (most-frequent value).
  2. Segment connected same-color non-background components per Color.
  3. Build Regions and Entities.
  4. Match Entities to prev_state for ID stability + velocity.
  5. Detect Events between prev_state and the new state.

The segmentation uses pure-numpy BFS (no numba dependency). This is
slow per frame (~1-3ms on 64×64) but adequate for Stage 2; we can
swap in the optimized labeller later if it becomes a bottleneck.

Stage 2 deliberately keeps this simple: an entity is a maximal
4-connected same-color non-background region. Refinements (intra-
sprite structure, salience-foreground, group entities) come in
later stages.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from biobrain.types import (
    EVENT_COLOR_CHANGED,
    EVENT_ENTITY_DESPAWNED,
    EVENT_ENTITY_SPAWNED,
    EVENT_LEVEL_INCREASED,
    EVENT_SCORE_INCREASED,
    Cell,
    Entity,
    Event,
    Region,
    State,
)

GRID_SHAPE = (64, 64)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def background_color(grid: np.ndarray) -> int:
    """Most-frequent value in the grid."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _flood_fill(
    grid: np.ndarray,
    visited: np.ndarray,
    start: tuple[int, int],
) -> frozenset[Cell]:
    """4-connected flood fill from `start`. Marks visited along the way."""
    h, w = grid.shape
    color = grid[start]
    stack = [start]
    cells: list[Cell] = []
    while stack:
        r, c = stack.pop()
        if visited[r, c]:
            continue
        if grid[r, c] != color:
            continue
        visited[r, c] = True
        cells.append((r, c))
        if r > 0:
            stack.append((r - 1, c))
        if r + 1 < h:
            stack.append((r + 1, c))
        if c > 0:
            stack.append((r, c - 1))
        if c + 1 < w:
            stack.append((r, c + 1))
    return frozenset(cells)


def segment(
    grid: np.ndarray,
    salience_mask: Optional[np.ndarray] = None,
) -> list[tuple[int, frozenset[Cell]]]:
    """Return list of (color, cells) for every non-background component.

    Components are 4-connected and same-color. Background is dropped
    UNLESS the corresponding cell in `salience_mask` is True — in
    which case the cell is treated as foreground regardless of color
    (the salience hook for HUDs that toggle between bg and non-bg
    values).
    """
    bg = background_color(grid)
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, frozenset[Cell]]] = []
    has_salience = salience_mask is not None and salience_mask.any()
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            # Cell is foreground if it's non-bg OR salience-flagged.
            is_fg = grid[r, c] != bg
            if has_salience and not is_fg:
                is_fg = bool(salience_mask[r, c])
            if not is_fg:
                visited[r, c] = True
                continue
            cells = _flood_fill_salient(grid, visited, (r, c), bg,
                                         salience_mask, has_salience)
            if cells:
                components.append((int(grid[r, c]), cells))
    return components


def _flood_fill_salient(
    grid: np.ndarray,
    visited: np.ndarray,
    start: tuple[int, int],
    bg: int,
    salience_mask: Optional[np.ndarray],
    has_salience: bool,
) -> frozenset[Cell]:
    """4-connected flood fill from `start`. Treats salience-flagged cells
    as foreground EVEN IF their color == bg."""
    h, w = grid.shape
    color = grid[start]
    stack = [start]
    cells: list[Cell] = []
    while stack:
        r, c = stack.pop()
        if visited[r, c]:
            continue
        if grid[r, c] != color:
            continue
        # Skip bg-colored cells that aren't salience-flagged.
        if grid[r, c] == bg and not (has_salience and salience_mask[r, c]):
            continue
        visited[r, c] = True
        cells.append((r, c))
        if r > 0:
            stack.append((r - 1, c))
        if r + 1 < h:
            stack.append((r + 1, c))
        if c > 0:
            stack.append((r, c - 1))
        if c + 1 < w:
            stack.append((r, c + 1))
    return frozenset(cells)


# ---------------------------------------------------------------------------
# Region / Entity construction
# ---------------------------------------------------------------------------

def _bbox(cells: frozenset[Cell]) -> tuple[int, int, int, int]:
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return (min(rows), min(cols), max(rows), max(cols))


def build_region(cells: frozenset[Cell]) -> Region:
    return Region(cells=cells, bbox=_bbox(cells))


def _centroid(region: Region) -> tuple[float, float]:
    rows = [c[0] for c in region.cells]
    cols = [c[1] for c in region.cells]
    return (sum(rows) / len(rows), sum(cols) / len(cols))


# ---------------------------------------------------------------------------
# ID assignment + matching to prev_state
# ---------------------------------------------------------------------------

def _match_entities(
    candidates: list[tuple[int, Region]],
    prev_entities: frozenset[Entity],
) -> list[Entity]:
    """Match new candidates (color, region) to prev entities for stable IDs.

    Heuristic: greedy nearest-centroid match within same-color set.
    Tolerate moderate region shape changes. Compute velocity from
    centroid displacement.

    Unmatched candidates get fresh IDs (max existing + counter).
    """
    prev_by_color: dict[int, list[Entity]] = {}
    for e in prev_entities:
        prev_by_color.setdefault(e.color, []).append(e)

    # Sort prev entities for stable iteration.
    for c in prev_by_color:
        prev_by_color[c].sort(key=lambda e: e.id)

    used_prev_ids: set[int] = set()
    next_id = (max((e.id for e in prev_entities), default=0)) + 1

    results: list[Entity] = []
    for color, region in candidates:
        new_centroid = _centroid(region)
        # Find best matching prev entity of same color by centroid distance.
        best: Optional[Entity] = None
        best_d = float("inf")
        for prev in prev_by_color.get(color, ()):
            if prev.id in used_prev_ids:
                continue
            pc = _centroid(prev.region)
            d = (new_centroid[0] - pc[0]) ** 2 + (new_centroid[1] - pc[1]) ** 2
            if d < best_d:
                best_d = d
                best = prev
        # Accept the nearest same-color match as long as it's closer
        # than the grid's half-diagonal. Entities don't teleport across
        # the grid in one step; movements within that radius are the
        # same entity. Coarse, but adequate for Stage 2.
        if best is not None:
            # 64²/2 = 2048 as the soft threshold.
            if best_d <= 2048:
                pc = _centroid(best.region)
                vel: Cell = (
                    int(round(new_centroid[0] - pc[0])),
                    int(round(new_centroid[1] - pc[1])),
                )
                results.append(Entity(
                    id=best.id, region=region, color=color, velocity=vel,
                ))
                used_prev_ids.add(best.id)
                continue
        # New entity.
        results.append(Entity(
            id=next_id, region=region, color=color, velocity=(0, 0),
        ))
        next_id += 1
    return results


# ---------------------------------------------------------------------------
# Top-level perceive
# ---------------------------------------------------------------------------

def perceive(
    raw_obs: np.ndarray,
    prev_state: Optional[State],
    *,
    score: int = 0,
    level: int = 0,
    available_actions: tuple[int, ...] = (),
    salience_mask: Optional[np.ndarray] = None,
) -> State:
    """Raw grid + prev state → State.

    `raw_obs` is a (64, 64) uint8 ndarray. `prev_state` may be None at
    the start of an attempt. `available_actions` is the set of action
    IDs the env permits at this step (1..5 keys, 6 click, 7 undo).

    `salience_mask` (optional) is a bool ndarray same shape as the
    grid; True cells are forced to foreground in segmentation
    regardless of color. This is the causal-relevance salience hook
    (cited Spelke 1990 + the ls20 HUD-detection lesson): cells whose
    VALUES change across frames get pulled into the foreground even
    when their current value matches the dominant background. Without
    this, small HUD indicators that toggle between bg-colored and
    non-bg-colored values get treated as background half the time.
    """
    if raw_obs.shape != GRID_SHAPE:
        # Don't crash — measurement is sometimes done on smaller views.
        # Pad / crop to GRID_SHAPE.
        padded = np.zeros(GRID_SHAPE, dtype=np.uint8)
        h, w = raw_obs.shape
        padded[:min(h, 64), :min(w, 64)] = raw_obs[:min(h, 64), :min(w, 64)]
        raw_obs = padded

    components = segment(raw_obs, salience_mask=salience_mask)
    candidates: list[tuple[int, Region]] = [
        (color, build_region(cells)) for (color, cells) in components
    ]
    prev_entities = prev_state.entities if prev_state else frozenset()
    matched = _match_entities(candidates, prev_entities)

    return State(
        entities=frozenset(matched),
        score=int(score),
        level=int(level),
        grid_hash=int(hash(raw_obs.tobytes())),
        raw_grid=raw_obs,
        available_actions=tuple(int(a) for a in available_actions),
    )


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_events(
    before: State,
    after: State,
) -> frozenset[Event]:
    """Compare before and after to extract Events.

    Stage 2: ScoreIncreased, LevelIncreased, EntitySpawned/Despawned,
    ColorChanged. Contact events (ContactGained/Lost) are derivable
    from entity-pair adjacency analysis — added in a later stage if
    needed; not part of the Stage 2 gate.
    """
    events: list[Event] = []

    if after.score > before.score:
        events.append(Event.make(
            EVENT_SCORE_INCREASED, delta=after.score - before.score,
        ))
    if after.level > before.level:
        events.append(Event.make(
            EVENT_LEVEL_INCREASED, from_=before.level, to=after.level,
        ))

    before_ids = {e.id for e in before.entities}
    after_ids = {e.id for e in after.entities}
    for new_id in after_ids - before_ids:
        events.append(Event.make(EVENT_ENTITY_SPAWNED, entity_id=new_id))
    for gone_id in before_ids - after_ids:
        events.append(Event.make(EVENT_ENTITY_DESPAWNED, entity_id=gone_id))

    # Color change: same ID, color differs.
    before_by_id = {e.id: e for e in before.entities}
    for e in after.entities:
        prev = before_by_id.get(e.id)
        if prev is not None and prev.color != e.color:
            events.append(Event.make(
                EVENT_COLOR_CHANGED,
                entity_id=e.id, from_=prev.color, to=e.color,
            ))

    return frozenset(events)
