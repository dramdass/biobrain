"""biobrain.adapters.arc.env — env binding for ARC-AGI-3.

Wraps the `arc_agi` SDK. Standalone (no spelke imports). Provides
`ArenaEnv` for stepping through games.

The arc_agi SDK must be installed:
    pip install git+https://github.com/arcprize/ARC-AGI-3.git
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from biobrain.types import Action, action_kind


VALID_MODES = ("OFFLINE", "ONLINE", "COMPETITION")
_VALID_KINDS = frozenset({"key", "click", "undo"})


def _arc_mode(mode_str: str):
    """User-facing mode → arc_agi.OperationMode."""
    from arc_agi import OperationMode  # type: ignore
    s = mode_str.upper()
    if s == "OFFLINE":
        return OperationMode.OFFLINE
    if s == "ONLINE":
        return OperationMode.ONLINE
    if s == "COMPETITION":
        return OperationMode.COMPETITION
    raise ValueError(f"Unknown mode {mode_str!r}; expected one of {VALID_MODES}")


def make_arcade(mode_str: str, environments_dir: Optional[str] = None):
    """Construct an arc_agi.Arcade for the given mode string.

    environments_dir: path to the directory containing per-game subdirs
    (each with metadata.json + the game module). Defaults to the
    arc_agi SDK's default scan path ("environment_files" relative to
    cwd).
    """
    import arc_agi  # type: ignore
    kwargs = {"operation_mode": _arc_mode(mode_str)}
    if environments_dir is not None:
        kwargs["environments_dir"] = environments_dir
    return arc_agi.Arcade(**kwargs)


def action_to_game_action(action: Action) -> tuple[Any, Optional[dict]]:
    """Map a biobrain Action tuple to (arc_agi.GameAction, kwargs|None)."""
    kind = action_kind(action)
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown action kind: {kind!r}")
    from arcengine import GameAction  # type: ignore
    if kind == "key":
        return GameAction.from_id(int(action[1])), None
    if kind == "click":
        return GameAction.ACTION6, {"x": int(action[1]), "y": int(action[2])}
    return GameAction.from_id(7), None


def _obs_to_frame_dict(obs: Any) -> dict[str, Any]:
    """Extract {grid, score, levels_completed, terminal, available_actions}
    from an arc_agi observation.
    """
    if obs is None:
        return {"grid": None, "score": 0, "levels_completed": 0,
                "terminal": None, "available_actions": ()}
    raw_frames = getattr(obs, "frame", None)
    if raw_frames is None:
        grid = None
    else:
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
    """biobrain's wrapper around arc_agi.Arcade for a single game.

    Lifecycle:
        env = ArenaEnv("vc33", mode="OFFLINE")
        obs = env.reset()
        while not env.is_terminal(obs):
            frame = env.parse(obs)
            action = brain.act(...)
            obs = env.step(action)
        env.close()
    """

    def __init__(self, game: str, mode: str = "OFFLINE",
                 environments_dir: Optional[str] = None) -> None:
        """Build the env binding.

        environments_dir: path containing per-game directories. If None,
        falls back to BIOBRAIN_ENV_DIR env var, then to the SDK default
        ("environment_files" relative to cwd).
        """
        import os
        if environments_dir is None:
            environments_dir = os.environ.get("BIOBRAIN_ENV_DIR")
        self._arc = make_arcade(mode, environments_dir=environments_dir)
        self._env = self._arc.make(game)
        if self._env is None:
            avail = environments_dir or "(default ./environment_files)"
            raise RuntimeError(
                f"Game {game!r} not found in {avail}. "
                f"Set BIOBRAIN_ENV_DIR or pass environments_dir explicitly."
            )
        self._game = game
        self._mode = mode

    @property
    def game(self) -> str:
        return self._game

    def reset(self) -> Any:
        return self._env.reset()

    def step(self, action: Action) -> Any:
        ga, kwargs = action_to_game_action(action)
        if kwargs is None:
            return self._env.step(ga)
        return self._env.step(ga, data=kwargs)

    def parse(self, obs: Any) -> dict[str, Any]:
        return _obs_to_frame_dict(obs)

    @staticmethod
    def is_terminal(obs: Any) -> bool:
        if obs is None:
            return True
        state = getattr(obs, "state", None)
        name = getattr(state, "name", None) if state is not None else None
        return name in ("WIN", "GAME_OVER", "LOSE")

    def close(self) -> None:
        try:
            sc = getattr(self._env, "scorecard_id", None)
            if sc:
                self._arc.close_scorecard(sc)
        except Exception:
            pass


__all__ = ["ArenaEnv", "make_arcade", "action_to_game_action", "VALID_MODES"]
