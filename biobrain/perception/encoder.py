"""biobrain.perception.encoder — the Encoder boundary.

The Encoder is the only thing in the brain library that knows how to map
States to fact sets. Everything else operates on facts.

`DefaultSpelkeEncoder` is the reference implementation. It wraps
`emit_atomic_facts` from biobrain.curiosity.predicates (the Spelke-axis
predicate vocabulary) plus an optional finer-attention mode that emits
sub-entity features when Salience requests them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from biobrain.curiosity.predicates import emit_atomic_facts
from biobrain.protocols import ActionLike, Fact, StateLike


GRID_DIM = 64


# ---------------------------------------------------------------------------
# Sub-entity / fine-attention predicates
# ---------------------------------------------------------------------------

def _emit_fine_attention_facts(state: StateLike,
                                attention_cells: frozenset) -> set[Fact]:
    """Sub-entity predicates over cells Salience has flagged for finer attention.

    Emits:
      ('fine_cell_color', row, col, color)   — exact color of attended cell
      ('fine_cell_distinct', row, col)       — cell differs from its 4-neighbors

    These are finer-grained than the coarse entity-level predicates. They
    let the WM and Critic see modal-state tells (e.g., a highlighted tile)
    that entity-level aggregation throws away.

    Bounded: only emitted for cells in attention_cells, so the predicate
    space stays small.
    """
    out: set[Fact] = set()
    if not attention_cells:
        return out
    if not hasattr(state, "raw_grid") or state.raw_grid is None:
        return out
    grid = np.asarray(state.raw_grid)
    h, w = grid.shape
    for r, c in attention_cells:
        if not (0 <= r < h and 0 <= c < w):
            continue
        color = int(grid[r, c])
        out.add(("fine_cell_color", int(r), int(c), color))
        # Distinct from neighbors? (cheap topology signal)
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append(int(grid[rr, cc]))
        if neighbors and all(n != color for n in neighbors):
            out.add(("fine_cell_distinct", int(r), int(c)))
    return out


# ---------------------------------------------------------------------------
# DefaultSpelkeEncoder — the reference Encoder implementation
# ---------------------------------------------------------------------------

class DefaultSpelkeEncoder:
    """Encoder implementing the Spelke-axis predicate vocabulary.

    Coarse (default): emit_atomic_facts over the State's entities.
    Fine (on attention_hint): adds sub-entity predicates for attended cells.
    """

    def encode(self,
               state: StateLike,
               attention_hint: frozenset | None = None,
               before: StateLike | None = None,
               ) -> frozenset[Fact]:
        """Emit fact set for `state`. If `before` is provided, delta facts
        (any_spawn, any_change, count_up_color, etc.) are included.

        attention_hint, if non-empty, adds fine-cell predicates for those cells.
        """
        base = set(emit_atomic_facts(before, state))
        if attention_hint:
            base |= _emit_fine_attention_facts(state, attention_hint)
        return frozenset(base)

    def resolve(self, action_sig: tuple, state: StateLike,
                candidates: Sequence[ActionLike]) -> ActionLike | None:
        """Convert a DSL ActionSig to a concrete Action from candidates.

        Sigs handled:
            ('click_on_color', c) → click at the centroid of any color-c entity
            ('key', k)            → key action with this k
            ('spacebar',)         → key 4 if available, else any key
            ('undo',)             → undo if available
            ('noop',)             → returns None (brain falls back)
        """
        if not action_sig:
            return None
        kind = action_sig[0]
        if kind == "click_on_color" and len(action_sig) >= 2:
            target_c = int(action_sig[1])
            for a in candidates:
                if len(a) >= 3 and a[0] == "click":
                    x, y = int(a[1]), int(a[2])
                    for e in state.entities:
                        if int(e.color) == target_c and (y, x) in e.region.cells:
                            return a
            return None
        if kind == "key" and len(action_sig) >= 2:
            target_k = int(action_sig[1])
            for a in candidates:
                if len(a) >= 2 and a[0] == "key" and int(a[1]) == target_k:
                    return a
            return None
        if kind == "spacebar":
            for sub_id in (4, 3, 2, 1, 0):
                for a in candidates:
                    if len(a) >= 2 and a[0] == "key" and int(a[1]) == sub_id:
                        return a
            return None
        if kind == "undo":
            for a in candidates:
                if a[0] == "undo":
                    return a
            return None
        return None

    def candidate_actions(self, state: StateLike) -> list[ActionLike]:
        """Enumerate candidate actions from state.available_actions + entities."""
        # Re-import inside to avoid a circular import surface
        from biobrain.types import action_click, action_key, action_undo
        out: list = []
        if not state.available_actions:
            return out
        for aid in state.available_actions:
            if aid in (1, 2, 3, 4, 5):
                out.append(action_key(aid - 1))
            elif aid == 6:
                # Click at the centroid of each entity
                for e in state.entities:
                    cells = list(e.region.cells)
                    if not cells:
                        continue
                    rows = [c[0] for c in cells]
                    cols = [c[1] for c in cells]
                    r = int(round(sum(rows) / len(rows)))
                    c = int(round(sum(cols) / len(cols)))
                    if 0 <= r < GRID_DIM and 0 <= c < GRID_DIM:
                        out.append(action_click(c, r))
            elif aid == 7:
                out.append(action_undo())
        return out


__all__ = ["DefaultSpelkeEncoder", "GRID_DIM"]
