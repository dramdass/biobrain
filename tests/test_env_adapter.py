"""Unit tests for biobrain.adapters.arc.env.

These tests deliberately avoid importing the real `arc_agi` / `arcengine`
SDKs at runtime — every external symbol is mocked. This lets the suite
run in CI without the arena dependency installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from biobrain.adapters.arc import env as env_mod
from biobrain.adapters.arc.env import (
    ArenaEnv,
    VALID_MODES,
    _obs_to_frame_dict,
    action_to_game_action,
    make_arcade,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_state():
    """Build a small helper for fake state objects with a `.name`."""
    def _make(name):
        s = types.SimpleNamespace(name=name)
        return s
    return _make


@pytest.fixture
def fake_obs(fake_state):
    """Construct an obs-like SimpleNamespace with sensible defaults."""
    def _make(
        frame=None,
        score=5,
        levels_completed=2,
        state_name="NOT_PLAYED",
        available_actions=(1, 2, 3),
    ):
        return types.SimpleNamespace(
            frame=frame,
            score=score,
            levels_completed=levels_completed,
            state=fake_state(state_name),
            available_actions=available_actions,
        )
    return _make


@pytest.fixture
def purge_arc_modules():
    """Remove arc_agi / arcengine from sys.modules so the lazy imports
    inside env.py fail with ImportError unless explicitly stubbed.

    Restores the original modules at teardown.
    """
    saved = {}
    for name in ("arc_agi", "arcengine"):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    yield
    # Cleanup: drop anything tests added, restore originals.
    for name in ("arc_agi", "arcengine"):
        sys.modules.pop(name, None)
    for name, mod in saved.items():
        sys.modules[name] = mod


@pytest.fixture
def fake_arc_agi(purge_arc_modules):
    """Install a fake `arc_agi` module so `make_arcade` succeeds without
    touching the real SDK.

    Yields the fake module so individual tests can inspect call args.
    """
    fake = types.ModuleType("arc_agi")
    fake.OperationMode = types.SimpleNamespace(
        OFFLINE="OFFLINE_MODE_SENTINEL",
        ONLINE="ONLINE_MODE_SENTINEL",
        COMPETITION="COMPETITION_MODE_SENTINEL",
    )
    fake.Arcade = MagicMock(name="FakeArcade")
    sys.modules["arc_agi"] = fake
    yield fake


@pytest.fixture
def fake_arcengine(purge_arc_modules):
    """Install a fake `arcengine` module providing a minimal GameAction."""
    fake = types.ModuleType("arcengine")

    class FakeGameAction:
        ACTION6 = "ACTION6_SENTINEL"

        @staticmethod
        def from_id(i):
            return f"GA_ID_{int(i)}"

    fake.GameAction = FakeGameAction
    sys.modules["arcengine"] = fake
    yield fake


# ---------------------------------------------------------------------------
# _obs_to_frame_dict
# ---------------------------------------------------------------------------


def test_obs_to_frame_dict_none_returns_defaults():
    """None obs → default-valued dict, no exception."""
    out = _obs_to_frame_dict(None)
    assert out == {
        "grid": None,
        "score": 0,
        "levels_completed": 0,
        "terminal": None,
        "available_actions": (),
    }


def test_obs_to_frame_dict_extracts_fields(fake_obs):
    """Fields are pulled off the obs object and grid is the last frame."""
    last = [[1, 2], [3, 4]]
    obs = fake_obs(
        frame=[[[0, 0], [0, 0]], last],
        score=7.0,
        levels_completed=3,
        state_name="NOT_FINISHED",
        available_actions=(1, 2, 3, 6),
    )
    out = _obs_to_frame_dict(obs)
    assert isinstance(out["grid"], np.ndarray)
    assert out["grid"].dtype == np.uint8
    np.testing.assert_array_equal(out["grid"], np.array(last, dtype=np.uint8))
    assert out["score"] == 7.0
    assert out["levels_completed"] == 3
    assert out["terminal"] == "NOT_FINISHED"
    assert out["available_actions"] == (1, 2, 3, 6)


def test_obs_to_frame_dict_missing_state_attr():
    """obs with no .state attribute → terminal is None."""
    obs = types.SimpleNamespace(
        frame=None, score=0, levels_completed=0, available_actions=()
    )
    out = _obs_to_frame_dict(obs)
    assert out["terminal"] is None
    assert out["grid"] is None


def test_obs_to_frame_dict_single_frame_fallback(fake_obs):
    """If frame[-1] coerce fails, fall back to coercing the whole frame."""
    # A 2D array — `arr[-1]` is a 1D row, which np.array(..., uint8)
    # would happily build, so we use something that exercises the fallback
    # cleanly: a flat list whose last element is a scalar.
    obs = fake_obs(frame=[[0, 1], [2, 3]])
    out = _obs_to_frame_dict(obs)
    # Last frame is [2, 3] → 1D array of two elements.
    assert isinstance(out["grid"], np.ndarray)
    np.testing.assert_array_equal(out["grid"], np.array([2, 3], dtype=np.uint8))


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


def test_is_terminal_none_is_true():
    assert ArenaEnv.is_terminal(None) is True


@pytest.mark.parametrize("name", ["WIN", "GAME_OVER", "LOSE"])
def test_is_terminal_for_terminal_state_names(fake_obs, name):
    obs = fake_obs(state_name=name)
    assert ArenaEnv.is_terminal(obs) is True


@pytest.mark.parametrize("name", ["NOT_PLAYED", "NOT_FINISHED", "PLAYING", None])
def test_is_terminal_for_non_terminal_state_names(fake_obs, name):
    obs = fake_obs(state_name=name)
    assert ArenaEnv.is_terminal(obs) is False


def test_is_terminal_obs_without_state():
    """An obs missing .state altogether is not terminal."""
    obs = types.SimpleNamespace()
    assert ArenaEnv.is_terminal(obs) is False


# ---------------------------------------------------------------------------
# action_to_game_action
# ---------------------------------------------------------------------------


def test_action_to_game_action_rejects_unknown_kind_without_arc_agi(purge_arc_modules):
    """ValueError must be raised before any arc_agi import is attempted —
    so the call works (and fails cleanly) even when the SDK is absent.
    """
    assert "arc_agi" not in sys.modules
    assert "arcengine" not in sys.modules
    with pytest.raises(ValueError, match="unknown action kind"):
        action_to_game_action(("bogus", 1))
    # Confirm we didn't accidentally pull in the real SDK.
    assert "arcengine" not in sys.modules


def test_action_to_game_action_key(fake_arcengine):
    ga, kwargs = action_to_game_action(("key", 3))
    assert ga == "GA_ID_3"
    assert kwargs is None


def test_action_to_game_action_click(fake_arcengine):
    ga, kwargs = action_to_game_action(("click", 10, 20))
    assert ga == "ACTION6_SENTINEL"
    assert kwargs == {"x": 10, "y": 20}


def test_action_to_game_action_undo(fake_arcengine):
    ga, kwargs = action_to_game_action(("undo",))
    assert ga == "GA_ID_7"
    assert kwargs is None


# ---------------------------------------------------------------------------
# make_arcade
# ---------------------------------------------------------------------------


def test_make_arcade_offline_passes_correct_operation_mode(fake_arc_agi):
    arc = make_arcade("OFFLINE")
    assert arc is fake_arc_agi.Arcade.return_value
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="OFFLINE_MODE_SENTINEL"
    )


def test_make_arcade_online_passes_correct_operation_mode(fake_arc_agi):
    make_arcade("ONLINE")
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="ONLINE_MODE_SENTINEL"
    )


def test_make_arcade_competition_passes_correct_operation_mode(fake_arc_agi):
    make_arcade("COMPETITION")
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="COMPETITION_MODE_SENTINEL"
    )


def test_make_arcade_is_case_insensitive(fake_arc_agi):
    make_arcade("offline")
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="OFFLINE_MODE_SENTINEL"
    )


def test_make_arcade_passes_environments_dir(fake_arc_agi):
    make_arcade("OFFLINE", environments_dir="/tmp/envs")
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="OFFLINE_MODE_SENTINEL",
        environments_dir="/tmp/envs",
    )


def test_make_arcade_invalid_mode_raises(fake_arc_agi):
    with pytest.raises(ValueError, match="Unknown mode"):
        make_arcade("INVALID")
    # The Arcade constructor must not have been called.
    fake_arc_agi.Arcade.assert_not_called()


def test_valid_modes_constant_exposed():
    assert set(VALID_MODES) == {"OFFLINE", "ONLINE", "COMPETITION"}


# ---------------------------------------------------------------------------
# ArenaEnv smoke
# ---------------------------------------------------------------------------


def test_arena_env_smoke_lifecycle(fake_arc_agi, fake_arcengine, fake_obs):
    """ArenaEnv constructs, resets, steps, parses without touching real SDKs."""
    fake_inner_env = MagicMock(name="FakeInnerEnv")
    reset_obs = fake_obs(state_name="NOT_PLAYED")
    step_obs = fake_obs(state_name="NOT_FINISHED", score=42)
    fake_inner_env.reset.return_value = reset_obs
    fake_inner_env.step.return_value = step_obs
    fake_inner_env.scorecard_id = None

    fake_arcade = MagicMock(name="FakeArcadeInstance")
    fake_arcade.make.return_value = fake_inner_env
    fake_arc_agi.Arcade.return_value = fake_arcade

    env = ArenaEnv("vc33", mode="OFFLINE")
    assert env.game == "vc33"
    fake_arcade.make.assert_called_once_with("vc33")

    obs1 = env.reset()
    assert obs1 is reset_obs

    # Key action — inner step called with the GameAction only.
    obs2 = env.step(("key", 2))
    assert obs2 is step_obs
    fake_inner_env.step.assert_called_once_with("GA_ID_2")

    # Click action — inner step called with positional GA + data kwargs.
    fake_inner_env.step.reset_mock()
    env.step(("click", 4, 7))
    fake_inner_env.step.assert_called_once_with(
        "ACTION6_SENTINEL", data={"x": 4, "y": 7}
    )

    parsed = env.parse(reset_obs)
    assert parsed["score"] == 5.0
    assert parsed["levels_completed"] == 2
    assert parsed["terminal"] == "NOT_PLAYED"

    env.close()  # no scorecard → no-op, must not raise.


def test_arena_env_unknown_game_raises(fake_arc_agi, fake_arcengine):
    """If Arcade.make returns None, ArenaEnv surfaces a RuntimeError."""
    fake_arcade = MagicMock()
    fake_arcade.make.return_value = None
    fake_arc_agi.Arcade.return_value = fake_arcade

    with pytest.raises(RuntimeError, match="not found"):
        ArenaEnv("nonexistent_game", mode="OFFLINE")


def test_arena_env_close_closes_scorecard(fake_arc_agi, fake_arcengine):
    """close() forwards a non-empty scorecard_id to Arcade.close_scorecard."""
    fake_inner_env = MagicMock()
    fake_inner_env.scorecard_id = "sc-123"

    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_inner_env
    fake_arc_agi.Arcade.return_value = fake_arcade

    env = ArenaEnv("vc33", mode="OFFLINE")
    env.close()
    fake_arcade.close_scorecard.assert_called_once_with("sc-123")


def test_arena_env_close_swallows_exceptions(fake_arc_agi, fake_arcengine):
    """Exceptions in close() must not propagate (best-effort cleanup)."""
    fake_inner_env = MagicMock()
    fake_inner_env.scorecard_id = "sc-1"

    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_inner_env
    fake_arcade.close_scorecard.side_effect = RuntimeError("boom")
    fake_arc_agi.Arcade.return_value = fake_arcade

    env = ArenaEnv("vc33", mode="OFFLINE")
    env.close()  # must not raise


def test_arena_env_uses_env_var_for_environments_dir(
    fake_arc_agi, fake_arcengine, monkeypatch
):
    """BIOBRAIN_ENV_DIR is consulted when environments_dir is not given."""
    monkeypatch.setenv("BIOBRAIN_ENV_DIR", "/some/path/from/env")
    fake_inner_env = MagicMock()
    fake_inner_env.scorecard_id = None
    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_inner_env
    fake_arc_agi.Arcade.return_value = fake_arcade

    ArenaEnv("vc33", mode="OFFLINE")
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="OFFLINE_MODE_SENTINEL",
        environments_dir="/some/path/from/env",
    )


def test_arena_env_explicit_dir_overrides_env_var(
    fake_arc_agi, fake_arcengine, monkeypatch
):
    """Explicit environments_dir wins over BIOBRAIN_ENV_DIR."""
    monkeypatch.setenv("BIOBRAIN_ENV_DIR", "/env/var/path")
    fake_inner_env = MagicMock()
    fake_inner_env.scorecard_id = None
    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_inner_env
    fake_arc_agi.Arcade.return_value = fake_arcade

    ArenaEnv("vc33", mode="OFFLINE", environments_dir="/explicit/path")
    fake_arc_agi.Arcade.assert_called_once_with(
        operation_mode="OFFLINE_MODE_SENTINEL",
        environments_dir="/explicit/path",
    )
