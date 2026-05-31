"""biobrain.perception.salience — causal-relevance salience mask.

Tracks per-cell value changes across observed frames. Cells whose
values have CHANGED at least once are flagged as salient; the
perception layer treats salient cells as foreground regardless of
their current color.

This is the named, falsifiable representation fix the Round-3
reviewer specifically called out for ls20:

  ls20 has a HUD region (rows 61-62, cols 17-21 per prior notes)
  that toggles between values 11 and 3. Both values are sometimes
  background-colored in ls20's palette, so the default
  perception drops the HUD half the time. The HUD encodes
  causally-relevant state (lives counter, etc.) that the agent
  must attend to.

  The salience mask makes this representation-derivable, not
  game-tuned: ANY cell whose value has changed across observed
  frames becomes salient. ls20's HUD cells satisfy this; static
  background cells don't.

Citation chain:
  - Causal-relevance attention is a Core Knowledge primitive
    (Saxe, Carey, Kanwisher 2004; Csibra-Gergely 2003 — agents
    attend to entities whose state varies under intervention).
  - Operational implementation: per-cell value-change tracking,
    O(grid_size) per transition.

NO game-specific tuning. NO HUD position hardcoded. The HUD's
location is discovered, not stated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


GRID_SHAPE = (64, 64)


@dataclass
class Salience:
    """Per-cell value-change tracker.

    Maintains:
      - `last_value`: most recently seen grid value per cell
      - `changed`: bool mask, True if this cell's value has changed
        across any pair of observed frames

    Update is O(grid_size) per observation. Reset on game boundary
    per OOD covenant.
    """
    last_value: Optional[np.ndarray] = None
    changed: np.ndarray = field(
        default_factory=lambda: np.zeros(GRID_SHAPE, dtype=bool))
    n_frames: int = 0

    def observe(self, grid: np.ndarray) -> None:
        """Record one frame; update the changed mask."""
        if grid.shape != GRID_SHAPE:
            return  # robustness: skip unexpected shapes
        if self.last_value is not None:
            diff = (grid != self.last_value)
            self.changed |= diff
        self.last_value = grid.copy()
        self.n_frames += 1

    def mask(self) -> np.ndarray:
        """Return the current salience mask (cells that have changed)."""
        return self.changed.copy()

    def n_salient_cells(self) -> int:
        return int(self.changed.sum())

    def reset(self) -> None:
        """Game-boundary reset (OOD covenant)."""
        self.last_value = None
        self.changed = np.zeros(GRID_SHAPE, dtype=bool)
        self.n_frames = 0
