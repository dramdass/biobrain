"""biobrain.salience.subgoals — subgoal detection by fingerprint delta.

Per spec §3 / §4: a subgoal is achieved when the state's F_mid
fingerprint changes between transitions. The action subsequence between
subgoals is accumulated and bound to the subgoal record.

Subgoals are indexed in the RoleFingerprintIndex under both start_fp
and end_fp (at all three granularities — handled by the index's
insert).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from biobrain.salience.fingerprint import Fingerprint


@dataclass(frozen=True)
class Subgoal:
    """A transferable unit — start fingerprint, action subsequence to
    achieve it, end fingerprint, validation flag, source metadata.
    """
    start_fp: Fingerprint
    action_subsequence: tuple  # tuple of action tuples
    end_fp: Fingerprint
    critic_validated: bool
    source_level: int
    source_attempt_id: int


class SubgoalDetector:
    """Detects subgoals on each transition via F_mid delta.

    Lifecycle:
      reset_attempt: clears the pending action accumulator (intra-attempt
                     state should not bleed into the next attempt).
      reset_game:    same as reset_attempt (House-model).
    """

    def __init__(self) -> None:
        self._pending_actions: list = []

    def reset_attempt(self) -> None:
        self._pending_actions = []

    def reset_game(self) -> None:
        self._pending_actions = []

    def observe_transition(
        self,
        fingerprint_before: Fingerprint,
        fingerprint_after: Fingerprint,
        action,
        critic_distance_dropped: bool,
        source_level: int,
        source_attempt_id: int,
    ) -> Optional[Subgoal]:
        """Append `action` to the accumulator. If F_mid changed, return a
        Subgoal containing the accumulator and reset it. Else return None.
        """
        self._pending_actions.append(tuple(action))
        if fingerprint_before.mid == fingerprint_after.mid:
            return None
        sg = Subgoal(
            start_fp=fingerprint_before,
            action_subsequence=tuple(self._pending_actions),
            end_fp=fingerprint_after,
            critic_validated=bool(critic_distance_dropped),
            source_level=source_level,
            source_attempt_id=source_attempt_id,
        )
        self._pending_actions = []
        return sg


__all__ = ["Subgoal", "SubgoalDetector"]
