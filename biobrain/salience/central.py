"""biobrain.salience.central — the central organ (subsumes Latent Inference).

Per the phenomenology insight (user playing ARC-AGI-3): salience is not a
perception utility, it's the *curating organ* that:
  1. Curates the small variable set the brain models (filters fact-space)
  2. Triggers finer perception when the WM is persistently wrong
  3. Banks salient-but-unexplained observations awaiting hypothesis
  4. Proposes new predicate templates (observable fine features OR latent
     schemas) when an explanation arrives
  5. Maintains the affordance posterior (which action classes have
     historically produced informative outcomes)

Modal-state strategy: attend-finer first (try to find an observable tell),
schema-fallback only if no observable feature explains the residual.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from biobrain.protocols import Fact, StateLike
from biobrain.latent_inference.schemas import (
    LatentSchema, DEFAULT_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Tunable thresholds (all flagged as RL-TODO)
# ---------------------------------------------------------------------------

# RL-TODO: derive from observed surprise distribution per game
SURPRISE_BANK_THRESHOLD = 0.4

# RL-TODO: window for residual-clustering hypothesis test
RESIDUAL_WINDOW = 30

# RL-TODO: minimum observations before declaring persistent error
MIN_OBSERVATIONS_FOR_PERSISTENT = 10

# RL-TODO: rate above which we declare a context "persistently wrong"
PERSISTENT_ERROR_RATE = 0.3

# RL-TODO: how many fine-attention cells to request per failure context
FINE_ATTENTION_CELLS_PER_CONTEXT = 9


# ---------------------------------------------------------------------------
# Banked surprise — observation awaiting explanation
# ---------------------------------------------------------------------------

@dataclass
class BankedSurprise:
    """One salient-but-unexplained observation."""
    transition_idx: int  # which transition (for retroactive backprop)
    context: tuple       # (action_kind, target_color, level) etc.
    surprise: float
    actual_facts: frozenset
    predicted_facts: frozenset
    explained_by: Optional[str] = None  # name of predicate that explains it


# ---------------------------------------------------------------------------
# Affordance posterior — per-action-class Beta of "informative"
# ---------------------------------------------------------------------------

@dataclass
class AffordancePosterior:
    """Per-action-class Beta(alpha, beta) of P(action produces information).

    Initialized either uniformly (covenant-respecting default) or from an
    adapter-supplied prior (covenant-relaxed mode).

    Updates on every transition: high surprise → alpha+, low surprise → beta+.
    The Planner consumes this at cold-path decision time as one of several
    candidate-ranking inputs.
    """
    counts: dict[str, tuple[float, float]] = field(default_factory=dict)

    def seed(self, priors: dict[str, tuple[float, float]]) -> None:
        """Seed from adapter-supplied priors (or empty for uniform)."""
        for kind, (a, b) in priors.items():
            self.counts[kind] = (float(a), float(b))

    def update(self, action_kind: str, surprise: float) -> None:
        alpha, beta = self.counts.get(action_kind, (0.0, 0.0))
        if surprise > 0:
            self.counts[action_kind] = (alpha + abs(surprise), beta)
        else:
            self.counts[action_kind] = (alpha, beta + abs(surprise))

    def posterior_mean(self, action_kind: str) -> float:
        alpha, beta = self.counts.get(action_kind, (0.0, 0.0))
        return (alpha + 1) / (alpha + beta + 2)


# ---------------------------------------------------------------------------
# Curated variable set
# ---------------------------------------------------------------------------

@dataclass
class CuratedVariables:
    """The small set of fact templates the brain currently models.

    A variable IS a fact-template (e.g., 'entity_color_quadrant'). Each
    is in one of three states:
      - active: emitted in the fact set, predicted by WM, used by Critic
      - inactive: known but currently filtered out
      - latent: instantiated from a schema (invisible variable)
    """
    active: set[str] = field(default_factory=set)
    inactive: set[str] = field(default_factory=set)
    latent: set[str] = field(default_factory=set)

    def activate(self, template_name: str) -> None:
        self.active.add(template_name)
        self.inactive.discard(template_name)

    def deactivate(self, template_name: str) -> None:
        self.inactive.add(template_name)
        self.active.discard(template_name)

    def is_active(self, template_name: str) -> bool:
        return template_name in self.active


# ---------------------------------------------------------------------------
# CentralSalience — the 3rd cortical component
# ---------------------------------------------------------------------------

class CentralSalience:
    """The central representation-state-management organ.

    Responsibilities:
      - Curate active variable set
      - Maintain fine-attention queue (cells to attend at finer perception)
      - Bank salient-but-unexplained surprises
      - Update affordance posterior
      - When persistent error detected: try attend-finer first, then
        schema-fallback to instantiate a latent variable

    Lifecycle:
      reset_game: wipe everything
      reset_attempt: keep modeled vars, banked surprises, affordance,
                     schemas; clear short-term fine-attention queue
      on_level_change: re-evaluate active vars for new level
    """

    def __init__(self,
                 schemas: Optional[list[LatentSchema]] = None) -> None:
        self.curated = CuratedVariables()
        self.affordance = AffordancePosterior()
        self._banked: list[BankedSurprise] = []
        self._fine_attention_queue: set[tuple[int, int]] = set()
        self._schema_pool = list(schemas if schemas is not None
                                  else DEFAULT_SCHEMAS)
        self._instantiated_schemas: list = []
        self._residual_buffer: deque = deque(maxlen=RESIDUAL_WINDOW)
        self._transition_idx = 0
        # All Spelke-axis templates are active by default
        for t in ("entity_color", "entity_size", "entity_quadrant",
                  "entity_color_quadrant", "entity_color_size",
                  "count_color", "count_size", "count_quadrant",
                  "total_entities", "level",
                  "any_motion", "any_spawn", "any_despawn", "any_change",
                  "spawn_color", "despawn_color",
                  "count_up_color", "count_down_color",
                  "count_reached_zero_color", "count_first_appeared_color"):
            self.curated.activate(t)

    # ----------------------------------------------------------- lifecycle

    def reset_game(self) -> None:
        self.curated = CuratedVariables()
        self.affordance = AffordancePosterior()
        self._banked = []
        self._fine_attention_queue = set()
        self._instantiated_schemas = []
        self._residual_buffer = deque(maxlen=RESIDUAL_WINDOW)
        self._transition_idx = 0
        # Re-activate defaults
        for t in ("entity_color", "entity_size", "entity_quadrant",
                  "entity_color_quadrant", "entity_color_size",
                  "count_color", "count_size", "count_quadrant",
                  "total_entities", "level",
                  "any_motion", "any_change"):
            self.curated.activate(t)

    def reset_attempt(self) -> None:
        # Keep modeled vars, banked surprises, affordance, schemas;
        # clear only the short-term attention queue.
        self._fine_attention_queue = set()

    def on_level_change(self, prev_level: int, new_level: int) -> None:
        # v0: no-op. v1: rebalance active variable set based on per-level
        # observed predictive utility.
        pass

    # ----------------------------------------------------------- observe

    def observe(self,
                surprise: float,
                context: tuple,
                action: Any,
                predicted_facts: frozenset,
                actual_facts: frozenset,
                ) -> None:
        """Process one transition's prediction outcome."""
        self._transition_idx += 1
        # Update affordance posterior
        action_kind = action[0] if action and len(action) > 0 else "unknown"
        self.affordance.update(action_kind, surprise)
        # Bank if salient
        if surprise >= SURPRISE_BANK_THRESHOLD:
            self._banked.append(BankedSurprise(
                transition_idx=self._transition_idx,
                context=context,
                surprise=surprise,
                actual_facts=actual_facts,
                predicted_facts=predicted_facts,
            ))
            if len(self._banked) > 200:
                self._banked.pop(0)
        # Track residuals per context for persistent-error detection
        had_error = (surprise > 0.2)
        self._residual_buffer.append({
            "context": context,
            "had_error": had_error,
            "action_kind": action_kind,
        })
        # Check whether to escalate to attend-finer or schema-instantiation
        self._maybe_escalate(context)

    def _maybe_escalate(self, context: tuple) -> None:
        """If recent residuals at this context cluster, escalate."""
        if len(self._residual_buffer) < MIN_OBSERVATIONS_FOR_PERSISTENT:
            return
        # Filter recent residuals matching this context
        same_ctx = [r for r in self._residual_buffer if r["context"] == context]
        if len(same_ctx) < MIN_OBSERVATIONS_FOR_PERSISTENT:
            return
        error_rate = sum(1 for r in same_ctx if r["had_error"]) / len(same_ctx)
        if error_rate < PERSISTENT_ERROR_RATE:
            return
        # Step 1: try attend-finer. Request fine perception at recent
        # surprise-banked cells (if we can extract them).
        recent_banked = [s for s in self._banked[-20:]
                          if s.context == context]
        if recent_banked:
            for s in recent_banked:
                for f in s.actual_facts:
                    if isinstance(f, tuple) and len(f) >= 2 \
                            and f[0] == "entity_quadrant":
                        # Quadrant index ∈ 0..15 → broad area; add center
                        q = int(f[1])
                        qy, qx = q // 4, q % 4
                        # 16x16 quadrant; center cells
                        for dr in range(2):
                            for dc in range(2):
                                self._fine_attention_queue.add(
                                    (qy * 16 + 8 + dr, qx * 16 + 8 + dc)
                                )
                        if len(self._fine_attention_queue) > 32:
                            return  # bounded
        # Step 2 (deferred to v0.3): schema fallback. If after K more
        # observations with finer attention the error remains, instantiate
        # the best-matching schema.

    # ----------------------------------------------------------- consumers

    def fact_filter(self, facts: frozenset[Fact]) -> frozenset[Fact]:
        """Filter the fact set to only active curated variables."""
        return frozenset(
            f for f in facts
            if isinstance(f, tuple) and self.curated.is_active(f[0])
        )

    def take_attention_hints(self) -> frozenset[tuple[int, int]]:
        """Get + clear the fine-attention queue for next observation."""
        out = frozenset(self._fine_attention_queue)
        self._fine_attention_queue = set()
        return out

    @property
    def banked_surprises(self) -> list[BankedSurprise]:
        return list(self._banked)

    @property
    def n_banked(self) -> int:
        return len(self._banked)

    def get_affordance(self, action_kind: str) -> float:
        return self.affordance.posterior_mean(action_kind)


__all__ = [
    "CentralSalience",
    "BankedSurprise",
    "AffordancePosterior",
    "CuratedVariables",
]
