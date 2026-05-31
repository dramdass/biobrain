"""biobrain.env.arena_env — wraps arc_agi.Arcade for the arena layer.

Brain-agnostic. Maps arena `Action` tuples to `arc_agi.GameAction`
and exposes raw frames + auxiliary signals (score, level,
baselines).

The arc_agi SDK is imported lazily inside the wrapper so the rest of
the arena and the probe framework remain importable without the SDK
installed — useful for unit tests that don't need the env.
"""

from __future__ import annotations

from typing import Any, Optional

from biobrain.types import Action, action_kind


_VALID_KINDS = frozenset({"key", "click", "undo"})


def action_to_game_action(action: Action) -> tuple[Any, Optional[dict]]:
    """Map an arena Action tuple to (arc_agi.GameAction, kwargs|None).

    Validates the action shape BEFORE importing arc_agi, so unknown
    actions raise ValueError even in environments without the SDK
    installed (e.g., unit tests).
    """
    kind = action_kind(action)
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown action kind: {kind!r}")

    # Lazy import: only loads arcengine if the action is well-formed.
    from arcengine import GameAction  # type: ignore

    if kind == "key":
        return GameAction.from_id(int(action[1])), None
    if kind == "click":
        return GameAction.ACTION6, {"x": int(action[1]), "y": int(action[2])}
    # kind == "undo"
    return GameAction.from_id(7), None


import numpy as np


def _obs_to_frame_dict(obs: Any) -> dict[str, Any]:
    """Extract {grid, score, levels_completed, terminal} from an arc_agi
    observation. Per scripts/measurement_pass.py the grid is in
    obs.frame[-1] (last frame of a multi-frame obs); the level counter
    is obs.levels_completed; the terminal tag is obs.state.name.
    """
    if obs is None:
        return {"grid": None, "score": 0, "levels_completed": 0, "terminal": None}
    raw_frames = getattr(obs, "frame", None)
    if raw_frames is None:
        grid = None
    else:
        # obs.frame is a list/array of frames; take the last.
        try:
            grid = np.array(raw_frames[-1], dtype=np.uint8)
        except (TypeError, IndexError):
            grid = np.array(raw_frames, dtype=np.uint8)
    state_name = getattr(getattr(obs, "state", None), "name", None)
    return {
        "grid": grid,
        "score": float(getattr(obs, "score", 0)),
        "levels_completed": int(getattr(obs, "levels_completed", 0)),
        "terminal": state_name,
        "available_actions": getattr(obs, "available_actions", ()),
    }


class ArenaEnv:
    """Wrapper around `arc_agi.Arcade`'s env for the arena layer.

    Lifecycle:
        env = ArenaEnv("vc33", mode="OFFLINE")
        obs = env.reset()
        while not env.is_terminal(obs):
            frame = env.parse(obs)
            action = brain.act(...)
            obs = env.step(action)
        env.close()

    `obs` is the raw SDK observation. `parse(obs)` extracts the
    arena-friendly dict with grid + score + levels_completed +
    terminal.
    """

    def __init__(self, game: str, mode: str = "OFFLINE") -> None:
        # Lazy import: tests that only check action_to_game_action
        # don't need the SDK.
        from scripts._runtime import make_arcade  # type: ignore

        self._arc = make_arcade(mode)
        self._env = self._arc.make(game)
        self._game = game
        self._mode = mode

    @property
    def game(self) -> str:
        return self._game

    def reset(self) -> Any:
        """Reset env to start of attempt. Returns the initial observation."""
        return self._env.reset()

    def step(self, action: Action) -> Any:
        """Dispatch an action; return the new observation (or None on env failure)."""
        ga, kwargs = action_to_game_action(action)
        if kwargs is None:
            return self._env.step(ga)
        return self._env.step(ga, data=kwargs)

    def parse(self, obs: Any) -> dict[str, Any]:
        """Extract {grid, score, levels_completed, terminal, available_actions}."""
        return _obs_to_frame_dict(obs)

    @staticmethod
    def is_terminal(obs: Any) -> bool:
        """True if the observation indicates WIN or GAME_OVER."""
        if obs is None:
            return True
        name = getattr(getattr(obs, "state", None), "name", "")
        return name in ("WIN", "GAME_OVER")

    @staticmethod
    def is_win(obs: Any) -> bool:
        if obs is None:
            return False
        name = getattr(getattr(obs, "state", None), "name", "")
        return name == "WIN"

    def baseline_actions(self) -> Optional[list[int]]:
        return getattr(self._env.info, "baseline_actions", None)

    def n_levels(self) -> int:
        bl = self.baseline_actions()
        return len(bl) if bl is not None else 0

    def close(self) -> None:
        """Close scorecard if applicable (no-op for OFFLINE)."""
        try:
            from scripts._runtime import close_scorecard  # type: ignore
            close_scorecard(self._arc, self._env)
        except Exception:
            pass
