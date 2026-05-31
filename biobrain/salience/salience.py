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


# ---------------------------------------------------------------------------
# SalienceCurator — the curating / attending organ (8th cortical component)
# ---------------------------------------------------------------------------

class SalienceCurator:
    """Curates the small variable set the brain currently models, and
    flags where to attend finer when prediction fails.

    v0 — minimal: maintains a running "interesting features" set
    (observations that don't yet have a hypothesis explaining them) and
    a "fine-attention queue" (cells/contexts where coarse prediction
    failed).

    Per the phenomenology insight: "I'm always noting interesting
    evidence" — the brain banks salient-but-unexplained observations
    before having a hypothesis for them. When a hypothesis later
    explains a banked surprise, that's strong confirmation.

    Wipes at reset_game; persists across reset_attempt.
    """

    def __init__(self) -> None:
        # Banked salient observations awaiting explanation
        self._banked: list[dict] = []
        # Cells / contexts where finer perceptual attention is requested
        self._fine_attention_targets: set = set()
        # Modeled variables — the small curated set
        self._modeled_variables: set = set()

    def reset_game(self) -> None:
        self._banked = []
        self._fine_attention_targets = set()
        self._modeled_variables = set()

    def bank_observation(self, observation: dict) -> None:
        """Note something interesting that doesn't yet have a hypothesis."""
        self._banked.append(observation)
        if len(self._banked) > 100:
            self._banked.pop(0)

    def request_fine_attention(self, target) -> None:
        """Flag a cell/context where coarse prediction failed; perception
        should attend at finer granularity here next observation.
        """
        self._fine_attention_targets.add(target)

    def take_fine_attention_targets(self) -> set:
        """Return + clear the fine-attention queue."""
        out = set(self._fine_attention_targets)
        self._fine_attention_targets.clear()
        return out

    def add_modeled_variable(self, var_name: str) -> None:
        self._modeled_variables.add(var_name)

    def remove_modeled_variable(self, var_name: str) -> None:
        self._modeled_variables.discard(var_name)

    @property
    def modeled_variables(self) -> set:
        return set(self._modeled_variables)

    @property
    def banked_observations(self) -> list:
        return list(self._banked)
