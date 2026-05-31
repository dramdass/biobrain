"""biobrain.critic.change_dynamics — ChangeDynamics extractor (the Y design).

Identifies proto-goals via the structural distinction between cells that
CHANGE under actions vs cells that stay STATIC. In most ARC-AGI-3
games this distinction is the canvas-vs-target structure: the agent
manipulates one region (active canvas) while another region encodes
the target/instructions (static).

Mechanism:
  1. Use TransitionHistory.change_rate_grid() to get per-cell change rate.
  2. Classify cells:
       DYNAMIC if change_rate >= DYNAMIC_THRESHOLD
       STATIC  if change_rate <  STATIC_THRESHOLD AND cell is foreground
       (cells in between are ambiguous and ignored)
  3. Group adjacent same-class cells into regions via connected components.
  4. For each (dynamic_region, static_region) pair:
       - Compute content-distance: how different is the dynamic region's
         cell content from the static region's?
       - Emit a ProtoGoal: minimize this distance.
  5. Weight proto-goals by how much DL would be saved if satisfied.

Works on: cd82 (cc_1 active vs cc_0 static), any canvas-vs-target game.

Bootstrap behavior: until at least MIN_TRANSITIONS observations are
recorded, this extractor emits nothing (no signal yet). Cold-start
games need other extractors to operate first.
"""

from __future__ import annotations

import numpy as np

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory


# Minimum number of transitions before this extractor produces output.
# Need enough samples to distinguish noise from genuine dynamics.
MIN_TRANSITIONS = 5

# A cell is DYNAMIC if it changed in >= this fraction of observed transitions.
DYNAMIC_THRESHOLD = 0.10

# A cell is STATIC if it changed in < this fraction.
# Cells in [STATIC_THRESHOLD, DYNAMIC_THRESHOLD) are ambiguous; ignored.
STATIC_THRESHOLD = 0.02

# Minimum region size (cells per side) to qualify as a goal-relevant region.
MIN_REGION_SIZE = 3
MAX_REGION_SIZE = 40


def _connected_components(mask: np.ndarray,
                          min_size: int,
                          max_size: int) -> list[tuple[int, int, int, int, list]]:
    """Return list of (r0, c0, r1, c1, cells) for each component in mask.

    4-connectivity. Filtered by bbox side length.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if visited[rr, cc] or not mask[rr, cc]:
                    continue
                visited[rr, cc] = True
                cells.append((rr, cc))
                stack.append((rr + 1, cc))
                stack.append((rr - 1, cc))
                stack.append((rr, cc + 1))
                stack.append((rr, cc - 1))
            if not cells:
                continue
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            ht = r1 - r0 + 1
            wd = c1 - c0 + 1
            if min_size <= ht <= max_size and min_size <= wd <= max_size:
                components.append((r0, c0, r1, c1, cells))
    return components


def _color_histogram(grid: np.ndarray, cells: list) -> np.ndarray:
    """16-bin color histogram over the cells. Normalized."""
    h = np.zeros(16, dtype=np.float32)
    for r, c in cells:
        v = int(grid[r, c])
        if 0 <= v < 16:
            h[v] += 1
    total = h.sum()
    if total > 0:
        h /= total
    return h


def _histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between normalized histograms ∈ [0, 2]; clip to [0,1]."""
    return min(1.0, float(np.abs(a - b).sum()) / 2.0)


class ChangeDynamics:
    """L3-Change extractor (Y design).

    Pairs DYNAMIC regions (cells that change under actions) with STATIC
    regions (cells that don't). The pairing is: each dynamic region
    paired with EVERY static region of comparable size. Goal: minimize
    histogram distance.

    The pairing is intentionally permissive — we let the brain's Thompson
    sample over multiple goals and let DL-weighting decide. This avoids
    encoding game-specific pairing heuristics.
    """
    name = "change_dynamics"

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        if history.n_transitions < MIN_TRANSITIONS:
            return []
        if not hasattr(state, "raw_grid") or state.raw_grid is None:
            return []
        raw = np.array(state.raw_grid)
        if raw.shape != history.change_rate_grid().shape:
            return []

        rate = history.change_rate_grid()
        # Background color = most-frequent in current state
        bg_color = int(np.bincount(raw.flatten(), minlength=16).argmax())
        fg_mask = raw != bg_color

        # DYNAMIC mask: cells changing often, regardless of fg/bg
        dynamic_mask = rate >= DYNAMIC_THRESHOLD
        # STATIC mask: foreground cells that rarely change
        static_mask = (rate < STATIC_THRESHOLD) & fg_mask

        dyn_components = _connected_components(
            dynamic_mask, MIN_REGION_SIZE, MAX_REGION_SIZE)
        stat_components = _connected_components(
            static_mask, MIN_REGION_SIZE, MAX_REGION_SIZE)

        if not dyn_components or not stat_components:
            return []

        goals: list[ProtoGoal] = []
        # For each dynamic region, pair with up to top-3 static regions
        # by area-similarity (closest-sized targets are likely the
        # "target version" of this canvas). Histogram-distance picks
        # which content-shape is closest.
        for dyn in dyn_components:
            d_r0, d_c0, d_r1, d_c1, d_cells = dyn
            d_area = len(d_cells)
            d_hist = _color_histogram(raw, d_cells)
            # Sort statics by area-similarity (smaller area-diff = more
            # similar size; ties broken by histogram closeness).
            scored = []
            for stat in stat_components:
                s_r0, s_c0, s_r1, s_c1, s_cells = stat
                s_area = len(s_cells)
                area_diff = abs(d_area - s_area) / max(d_area, s_area)
                s_hist = _color_histogram(raw, s_cells)
                hist_dist = _histogram_distance(d_hist, s_hist)
                scored.append((area_diff, hist_dist, stat))
            scored.sort()
            for area_diff, hist_dist, stat in scored[:3]:
                s_r0, s_c0, s_r1, s_c1, s_cells = stat
                # Goal distance: histogram distance between dyn & static
                # cells' content in the candidate state.
                dyn_cells_frozen = frozenset(d_cells)
                stat_cells_frozen = frozenset(s_cells)
                relevant_cells = dyn_cells_frozen | stat_cells_frozen

                def distance_fn(s, dcells=d_cells, scells=s_cells):
                    if not hasattr(s, "raw_grid") or s.raw_grid is None:
                        return 1.0
                    g = np.array(s.raw_grid)
                    try:
                        dh = _color_histogram(g, dcells)
                        sh = _color_histogram(g, scells)
                    except Exception:
                        return 1.0
                    return _histogram_distance(dh, sh)

                # Weight: how much DL would be saved if histograms aligned.
                # Approximate: log2(16) bits per cell × min(area) × match_potential
                dl_saving = min(d_area, len(s_cells)) * 4.0 * (1.0 - hist_dist)
                weight = min(1.0, dl_saving / 128.0)

                goals.append(ProtoGoal(
                    goal_id=f"change:dyn_{d_r0},{d_c0}_to_stat_{s_r0},{s_c0}",
                    description=(
                        f"change: dynamic region @({d_r0},{d_c0}) "
                        f"→ match static @({s_r0},{s_c0})"
                    ),
                    distance_fn=distance_fn,
                    weight=weight,
                    relevant_cells=relevant_cells,
                    source=self.name,
                    region_a_bbox=(d_r0, d_c0, d_r1, d_c1),
                    region_b_bbox=(s_r0, s_c0, s_r1, s_c1),
                ))
        return goals
