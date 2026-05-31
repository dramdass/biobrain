"""biobrain.critic.pattern_recurrence — StaticPatternRecurrence extractor.

Detects goals via static pattern recurrence between rectangular grid
regions. This is the original L3 v1/v2 detector, refactored to fit
the GoalExtractor protocol.

Mechanism:
  1. Find candidate regions via foreground/background segmentation
     + connected components.
  2. For each pair of same-shaped regions: compute cell match score.
  3. Emit a ProtoGoal "make A match B" for pairs with partial match
     (in (LOW_THRESHOLD, HIGH_THRESHOLD)).

Works on: static-puzzle games where the scene contains its own answer
(target pattern present + matching active region).

Limits: requires same-shape pairing, so misses shape-transformation
games where the active and target regions differ in size.
"""

from __future__ import annotations

import numpy as np

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory


# Threshold below which a region pair is uninteresting (random-level match).
MATCH_THRESHOLD_LOW = 0.30
# Threshold above which a region pair is already matched (goal satisfied).
MATCH_THRESHOLD_HIGH = 0.99


def _bbox_of_cells(cells) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def _candidate_regions(state: State,
                       min_size: int = 4,
                       max_size: int = 30) -> list[tuple]:
    """Find multi-color rectangular regions via background segmentation.

    1. Background = most-frequent color.
    2. Foreground mask = cells != background.
    3. Connected components in foreground mask (4-connectivity).
    4. Each component's bbox is a candidate region.

    Returns list of (r0, c0, r1, c1, label).
    """
    if not hasattr(state, "raw_grid") or state.raw_grid is None:
        return []
    raw = np.array(state.raw_grid)
    h, w = raw.shape
    bg_color = int(np.bincount(raw.flatten(), minlength=16).argmax())
    fg_mask = raw != bg_color

    visited = np.zeros_like(fg_mask, dtype=bool)
    regions: list[tuple] = []
    region_idx = 0

    for sr in range(h):
        for sc in range(w):
            if not fg_mask[sr, sc] or visited[sr, sc]:
                continue
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if visited[rr, cc] or not fg_mask[rr, cc]:
                    continue
                visited[rr, cc] = True
                cells.append((rr, cc))
                stack.append((rr + 1, cc))
                stack.append((rr - 1, cc))
                stack.append((rr, cc + 1))
                stack.append((rr, cc - 1))
            if not cells:
                continue
            r0, c0, r1, c1 = _bbox_of_cells(cells)
            ht = r1 - r0 + 1
            wd = c1 - c0 + 1
            if min_size <= ht <= max_size and min_size <= wd <= max_size:
                regions.append((r0, c0, r1, c1, f"cc_{region_idx}"))
                region_idx += 1
    return regions


def _cell_match_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float((a == b).mean())


class StaticPatternRecurrence:
    """L3-Recurrence extractor.

    Pairs same-shaped regions whose cell content partially matches.
    Emits ProtoGoals to close the remaining mismatch.
    """
    name = "static_pattern_recurrence"

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        if not hasattr(state, "raw_grid") or state.raw_grid is None:
            return []
        if not state.entities:
            return []
        regions = _candidate_regions(state)
        if len(regions) < 2:
            return []

        raw = np.array(state.raw_grid)
        goals: list[ProtoGoal] = []
        seen_pairs: set = set()
        for i, ra in enumerate(regions):
            for rb in regions[i + 1:]:
                ra_tup = tuple(ra[:4])
                rb_tup = tuple(rb[:4])
                a = raw[ra_tup[0]:ra_tup[2] + 1, ra_tup[1]:ra_tup[3] + 1]
                b = raw[rb_tup[0]:rb_tup[2] + 1, rb_tup[1]:rb_tup[3] + 1]
                if a.shape != b.shape:
                    continue
                match = _cell_match_score(a, b)
                if match < MATCH_THRESHOLD_LOW or match > MATCH_THRESHOLD_HIGH:
                    continue
                ra_label, rb_label = ra[4], rb[4]
                pair_key = (ra_label, rb_label)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                relevant_cells = frozenset(
                    (r, c)
                    for r in range(ra_tup[0], ra_tup[2] + 1)
                    for c in range(ra_tup[1], ra_tup[3] + 1)
                ) | frozenset(
                    (r, c)
                    for r in range(rb_tup[0], rb_tup[2] + 1)
                    for c in range(rb_tup[1], rb_tup[3] + 1)
                )
                dl_saving = a.size * match * 4.0

                def distance_fn(s, ra_tup=ra_tup, rb_tup=rb_tup):
                    if not hasattr(s, "raw_grid") or s.raw_grid is None:
                        return 1.0
                    try:
                        aa = np.array(s.raw_grid)[
                            ra_tup[0]:ra_tup[2] + 1, ra_tup[1]:ra_tup[3] + 1]
                        bb = np.array(s.raw_grid)[
                            rb_tup[0]:rb_tup[2] + 1, rb_tup[1]:rb_tup[3] + 1]
                    except Exception:
                        return 1.0
                    if aa.shape != bb.shape:
                        return 1.0
                    return 1.0 - float((aa == bb).mean())

                goals.append(ProtoGoal(
                    goal_id=f"recurrence:{ra_label}_{rb_label}",
                    description=f"recurrence: make {ra_label} match {rb_label}",
                    distance_fn=distance_fn,
                    weight=min(1.0, dl_saving / 64.0),
                    relevant_cells=relevant_cells,
                    source=self.name,
                    region_a_bbox=ra_tup,
                    region_b_bbox=rb_tup,
                ))
        return goals
