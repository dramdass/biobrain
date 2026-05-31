"""biobrain.planner.agency — candidate action enumeration.

Game-agnostic enumeration of available actions from a State. The brain
calls this at each act() to get the set of candidate actions to choose
among. Click candidates are emitted at Spelke-entity centroids (the
principled discretization — see PRINCIPLES.md).
"""

from __future__ import annotations

from biobrain.types import (
    Action, State, action_click, action_key, action_undo,
)


GRID_DIM = 64


def _entity_click_candidates(state: State) -> list[Action]:
    """Click candidates at the centroid of each Spelke entity."""
    out: list[Action] = []
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
    return out


def _candidate_actions(state: State) -> list[Action]:
    """Enumerate candidate actions available in this state.

    available_actions encodes which ARC-AGI-3 action IDs are valid:
      1..5 — key actions (mapped to key(0)..key(4))
      6    — click (emits one candidate per entity centroid)
      7    — undo
    """
    out: list[Action] = []
    if not state.available_actions:
        return out
    for aid in state.available_actions:
        if aid in (1, 2, 3, 4, 5):
            out.append(action_key(aid - 1))
        elif aid == 6:
            out.extend(_entity_click_candidates(state))
        elif aid == 7:
            out.append(action_undo())
    return out


__all__ = ["_candidate_actions", "_entity_click_candidates", "GRID_DIM"]
