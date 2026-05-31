"""biobrain.latent_inference.schemas — schema library for latent templates.

A LatentSchema is a TEMPLATE for a hidden state variable the brain
might hypothesize. Each schema knows:
  - How to detect when its instantiation is warranted (residual signature)
  - How to update its value given the observed transition (controlling-action)
  - How to project it as a fact tuple consumable by the WM and Critic

The library is intentionally small and Spelke-grounded:
  - ModeArmedBy:        a categorical mode set by an action, consumed by
                        another action (e.g., cd82's armed_color)
  - FlagSetBy:          a boolean toggle set by a specific action
  - CounterIncrementedBy: an integer count incremented by an action,
                        observable as the value of a position-bound cell

This is v0 of the Latent Inference architecture. The intent is NOT to
cover every possible latent — it is to cover the dominant modal patterns
in interactive grid games while remaining bounded and falsifiable.

# RL-TODO: schema selection (which schema best matches residual structure)
# is currently rule-based; could be learned from per-game outcomes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from biobrain.types import Action, State, action_kind


@dataclass
class LatentSchema(ABC):
    """Template for a latent fact. Subclasses define how the latent's
    value updates on each transition and how it projects as a fact.
    """
    name: str

    @abstractmethod
    def initial_value(self) -> Any:
        """The value assumed before any controlling action is observed."""

    @abstractmethod
    def update(self, current_value: Any, action: Action,
               before: State, after: State) -> Any:
        """Return the new latent value after this transition."""

    @abstractmethod
    def project(self, value: Any) -> Optional[tuple]:
        """Project the latent's current value as a fact tuple, or None
        if the latent has no value yet. The fact tuple will appear in
        emit_atomic_facts output once registered.
        """

    @abstractmethod
    def matches_residual(self, residual_signature: dict) -> float:
        """Score in [0, 1] of how well this schema explains the residual
        signature. Used by LatentInference to pick which schema to
        instantiate on detection of persistent prediction error.
        """


@dataclass
class ModeArmedBySchema(LatentSchema):
    """A categorical mode set by clicking on an entity of a specific color.

    Concrete example (cd82): armed_color. Default initial = None. Every
    time the agent does click_on_color(c) targeting a "selector"-ish
    entity, the armed_color = c. Subsequent paint actions then have
    deterministic outcome given armed_color.
    """
    controlling_action_kind: str = "click"

    def initial_value(self) -> Optional[int]:
        return None

    def update(self, current_value: Optional[int], action: Action,
               before: State, after: State) -> Optional[int]:
        if action_kind(action) != self.controlling_action_kind:
            return current_value
        if len(action) < 3:
            return current_value
        x, y = int(action[1]), int(action[2])
        # Find the entity clicked on
        for e in before.entities:
            if (y, x) in e.region.cells:
                return int(e.color)
        return current_value

    def project(self, value: Optional[int]) -> Optional[tuple]:
        if value is None:
            return None
        return (self.name, int(value))

    def matches_residual(self, residual_signature: dict) -> float:
        """Match when residuals correlate with the color of the most
        recent click action.
        """
        return residual_signature.get("correlates_with_recent_click_color", 0.0)


@dataclass
class FlagSetBySchema(LatentSchema):
    """A boolean toggle set by a specific action kind."""
    controlling_action_kind: str = "key"
    controlling_action_param: int = 0

    def initial_value(self) -> bool:
        return False

    def update(self, current_value: bool, action: Action,
               before: State, after: State) -> bool:
        if action_kind(action) != self.controlling_action_kind:
            return current_value
        if len(action) >= 2 and int(action[1]) == self.controlling_action_param:
            return not current_value
        return current_value

    def project(self, value: bool) -> tuple:
        return (self.name, bool(value))

    def matches_residual(self, residual_signature: dict) -> float:
        return residual_signature.get("correlates_with_recent_key", 0.0)


@dataclass
class CounterIncrementedBySchema(LatentSchema):
    """An integer count incremented by an action.

    Useful for games where progress is tracked by a status indicator the
    perception layer can't extract (e.g., a small numeric counter at the
    grid edge).
    """
    controlling_action_kind: str = "click"

    def initial_value(self) -> int:
        return 0

    def update(self, current_value: int, action: Action,
               before: State, after: State) -> int:
        if action_kind(action) != self.controlling_action_kind:
            return current_value
        # Increment only when grid changed in some meaningful way
        if before.grid_hash != after.grid_hash:
            return current_value + 1
        return current_value

    def project(self, value: int) -> tuple:
        # Bucket the count to keep the predicate space bounded
        if value <= 1:
            bucket = "0_1"
        elif value <= 3:
            bucket = "2_3"
        elif value <= 7:
            bucket = "4_7"
        else:
            bucket = "8+"
        return (self.name, bucket)

    def matches_residual(self, residual_signature: dict) -> float:
        return residual_signature.get("correlates_with_action_count", 0.0)


# Default schemas the LatentInference module tries to instantiate from.
DEFAULT_SCHEMAS = [
    ModeArmedBySchema(name="armed_color"),
    FlagSetBySchema(name="flag_key0", controlling_action_param=0),
    CounterIncrementedBySchema(name="action_counter"),
]
