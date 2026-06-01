"""biobrain.salience.fingerprint — role-based state identity for transfer.

A Fingerprint is the abstract identity of a state for cross-level transfer.
Three granularities:
  F_tight: multiset of (role, color, quadrant) tuples — discriminates layout
  F_mid:   multiset of (role, color) — color-anchored identity (the user's
           "same block lit up, different shape, same color" pattern)
  F_loose: multiset of just role tags — most permissive

The RoleFingerprintIndex stores subgoals under all three; lookup tries
tight first, then mid, then loose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from biobrain.salience.roles import Role


@dataclass(frozen=True)
class Fingerprint:
    """Immutable triple of fingerprint granularities."""
    tight: frozenset[tuple[Role, int, int]]
    mid: frozenset[tuple[Role, int]]
    loose: frozenset[Role]


def compute_fingerprint(
    entities: Iterable,
    role_assignments: dict[Any, Role],
    quadrant_of: Callable,
) -> Fingerprint:
    """Compute the 3-granularity fingerprint of a state.

    entities: iterable of objects with .id and .color
    role_assignments: map from entity id -> assigned Role
    quadrant_of: callable entity -> int (0..15), the 4x4 spatial quantization
    """
    tight_set: set[tuple[Role, int, int]] = set()
    mid_set: set[tuple[Role, int]] = set()
    loose_set: set[Role] = set()
    for e in entities:
        role = role_assignments.get(e.id, Role.UNKNOWN)
        color = int(e.color)
        quadrant = int(quadrant_of(e))
        tight_set.add((role, color, quadrant))
        mid_set.add((role, color))
        loose_set.add(role)
    return Fingerprint(
        tight=frozenset(tight_set),
        mid=frozenset(mid_set),
        loose=frozenset(loose_set),
    )


class RoleFingerprintIndex:
    """Stores subgoals under fingerprint keys; lookup at 3 granularities.

    Persists across attempts; wipes on reset_game.
    """

    def __init__(self) -> None:
        self._tight: dict[frozenset, list] = {}
        self._mid: dict[frozenset, list] = {}
        self._loose: dict[frozenset, list] = {}

    def reset_game(self) -> None:
        self._tight = {}
        self._mid = {}
        self._loose = {}

    def insert(self, fingerprint: Fingerprint, subgoal: Any) -> None:
        """Insert subgoal under all 3 granularity keys of this fingerprint."""
        self._tight.setdefault(fingerprint.tight, []).append(subgoal)
        self._mid.setdefault(fingerprint.mid, []).append(subgoal)
        self._loose.setdefault(fingerprint.loose, []).append(subgoal)

    def lookup(self, fingerprint: Fingerprint) -> list:
        """Lookup subgoals matching at the LOOSEST granularity that returns
        results. Tries tight -> mid -> loose; first non-empty wins.
        """
        results = self._tight.get(fingerprint.tight, [])
        if not results:
            results = self._mid.get(fingerprint.mid, [])
        if not results:
            results = self._loose.get(fingerprint.loose, [])
        seen = set()
        out = []
        for s in results:
            if id(s) not in seen:
                seen.add(id(s))
                out.append(s)
        return out

    def __len__(self) -> int:
        return sum(len(lst) for lst in self._tight.values())


__all__ = ["Fingerprint", "compute_fingerprint", "RoleFingerprintIndex"]
