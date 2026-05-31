"""Fixtures for probes — cached observations, synthetic frames, game metadata.

Probes that need real game frames should call `load_transcript(game)`,
which returns a recorded transcript if available or raises FileNotFoundError.
Probes catch and convert to SKIP via:

    try:
        frames = load_transcript("vc33")
    except FileNotFoundError:
        return ProbeResult.skipped(...)

Probes that use synthetic data (most of them) construct frames inline.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = ROOT / "arena" / "fixtures" / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

GAMES_DIR = ROOT / "environment_files"

# 25 public games (per ARC-AGI-3 spec). The probe framework's
# game-set discipline references this list.
PUBLIC_GAMES: tuple[str, ...] = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
    "wa30",
)


# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------

def empty_frame(bg_color: int = 0) -> np.ndarray:
    """A 64×64 grid filled with `bg_color`."""
    return np.full((64, 64), bg_color, dtype=np.uint8)


def place_rect(
    frame: np.ndarray,
    *,
    color: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Paint a solid rectangle of `color` at (top, left) with given dims."""
    frame = frame.copy()
    frame[top:top + height, left:left + width] = color
    return frame


def two_entities_frame(
    *,
    bg: int = 0,
    agent_color: int = 1,
    target_color: int = 2,
    agent_pos: tuple[int, int] = (10, 10),
    target_pos: tuple[int, int] = (20, 20),
    entity_size: int = 3,
) -> np.ndarray:
    """A frame with one agent rect and one target rect.

    Useful for perception sanity checks (entity count = 2; colors
    distinct; positions known).
    """
    frame = empty_frame(bg)
    frame = place_rect(frame, color=agent_color, top=agent_pos[0],
                       left=agent_pos[1], height=entity_size, width=entity_size)
    frame = place_rect(frame, color=target_color, top=target_pos[0],
                       left=target_pos[1], height=entity_size, width=entity_size)
    return frame


# ---------------------------------------------------------------------------
# Cached transcripts (real game observations)
# ---------------------------------------------------------------------------

def transcript_path(game: str, attempt: int = 1) -> pathlib.Path:
    return TRANSCRIPTS_DIR / game / f"attempt-{attempt:03d}.pkl"


def has_transcript(game: str, attempt: int = 1) -> bool:
    return transcript_path(game, attempt).exists()


def load_transcript(game: str, attempt: int = 1) -> list[dict[str, Any]]:
    """Load a recorded transcript.

    Returns a list of dicts: [{frame, action, score, level, ...}].
    Raises FileNotFoundError if the transcript hasn't been recorded.
    Probes typically catch and convert to SKIP.
    """
    p = transcript_path(game, attempt)
    if not p.exists():
        raise FileNotFoundError(f"no transcript at {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def save_transcript(game: str, attempt: int, frames: list[dict[str, Any]]) -> pathlib.Path:
    """Persist a transcript for use by probes."""
    p = transcript_path(game, attempt)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(frames, f)
    return p


# ---------------------------------------------------------------------------
# Game metadata
# ---------------------------------------------------------------------------

def game_dir(game: str) -> pathlib.Path:
    """Path to environment_files/<game>/, where the game's Python lives."""
    return GAMES_DIR / game


def game_exists(game: str) -> bool:
    return game_dir(game).is_dir()
