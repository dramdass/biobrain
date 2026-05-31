"""biobrain.latent_inference.inference — residual-clustering hypothesizer.

The LatentInference module watches World Model prediction residuals.
When residuals at a context cluster cleanly by some signature (e.g.,
recent click's target color), it instantiates the best-matching schema
and registers a latent fact. From then on, `emit_atomic_facts_extended`
includes the latent's projected fact, sharpening the WM's context.

This is v0: rule-based residual signature detection + schema-template
library. Future iterations will replace these with learned components.

# RL-TODO: residual signature detection thresholds are hand-set.
# RL-TODO: schema selection rule could be learned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from biobrain.types import Action, State, Transition, action_kind
from biobrain.curiosity.predicates import emit_atomic_facts
from biobrain.latent_inference.schemas import (
    LatentSchema, DEFAULT_SCHEMAS,
)


# Number of recent transitions tracked for residual analysis
RESIDUAL_WINDOW = 30
# Minimum residual count before we attempt to hypothesize a latent
MIN_RESIDUAL_OBSERVATIONS = 10
# Residual rate above which we declare "persistent prediction error"
PERSISTENT_ERROR_THRESHOLD = 0.30
# Schema match score above which we instantiate
SCHEMA_MATCH_THRESHOLD = 0.5


@dataclass
class LatentHypothesis:
    """An instantiated latent fact derived from a schema."""
    schema: LatentSchema
    value: Any
    n_updates: int = 0

    def project(self) -> Optional[tuple]:
        return self.schema.project(self.value)


class LatentInference:
    """Hypothesize and maintain latent facts based on WM residual analysis.

    Interface:
        observe(before, action, after, residual_facts):
            update latent values; possibly instantiate new latents
        extra_facts() → set:
            latent facts to UNION with emit_atomic_facts output
        reset_game() / reset_attempt()

    Where residual_facts = (actual_facts - predicted_high_prob_facts) ∪
                          (predicted_high_prob_facts - actual_facts).

    v0 is conservative: at most a few latents per game. The architecture
    is set up for more, but the schema library is intentionally small.
    """

    def __init__(self, schemas: Optional[list[LatentSchema]] = None,
                 max_latents: int = 4) -> None:
        self._schema_pool = list(schemas if schemas is not None
                                  else DEFAULT_SCHEMAS)
        self._max_latents = max_latents
        self._latents: list[LatentHypothesis] = []
        self._residual_buffer: deque = deque(maxlen=RESIDUAL_WINDOW)
        # Recent action history for signature computation
        self._recent_actions: deque = deque(maxlen=RESIDUAL_WINDOW)

    def reset_game(self) -> None:
        self._latents = []
        self._residual_buffer = deque(maxlen=RESIDUAL_WINDOW)
        self._recent_actions = deque(maxlen=RESIDUAL_WINDOW)

    def reset_attempt(self) -> None:
        # Latents persist intra-game; only clear short-term buffer
        self._residual_buffer = deque(maxlen=RESIDUAL_WINDOW)
        self._recent_actions = deque(maxlen=RESIDUAL_WINDOW)

    def observe(self, before: State, action: Action, after: State,
                residual_facts: set) -> None:
        """Update latents and possibly hypothesize new ones.

        residual_facts: the symmetric-diff of predicted vs actual fact
        sets. Non-empty → World Model was wrong here.
        """
        # 1. Update existing latents' values
        for hyp in self._latents:
            new_val = hyp.schema.update(hyp.value, action, before, after)
            if new_val != hyp.value:
                hyp.value = new_val
                hyp.n_updates += 1

        # 2. Buffer this residual for future hypothesis testing
        self._residual_buffer.append({
            "action": action,
            "n_residual": len(residual_facts),
            "had_error": bool(residual_facts),
        })
        self._recent_actions.append(action)

        # 3. Check whether to instantiate a new latent
        if len(self._latents) >= self._max_latents:
            return
        if len(self._residual_buffer) < MIN_RESIDUAL_OBSERVATIONS:
            return

        error_rate = sum(1 for r in self._residual_buffer if r["had_error"]) \
                     / len(self._residual_buffer)
        if error_rate < PERSISTENT_ERROR_THRESHOLD:
            return

        # Compute a residual signature: simple correlations with recent
        # action features. This is the bulk of what would become a richer
        # learned analysis; v0 uses rule-based heuristics.
        signature = self._compute_residual_signature()

        # Score each schema against the signature
        active_names = {h.schema.name for h in self._latents}
        scored: list[tuple[float, LatentSchema]] = []
        for sch in self._schema_pool:
            if sch.name in active_names:
                continue
            score = sch.matches_residual(signature)
            scored.append((score, sch))
        scored.sort(reverse=True, key=lambda t: t[0])
        if not scored:
            return
        best_score, best_schema = scored[0]
        if best_score >= SCHEMA_MATCH_THRESHOLD:
            self._latents.append(LatentHypothesis(
                schema=best_schema,
                value=best_schema.initial_value(),
            ))

    def _compute_residual_signature(self) -> dict:
        """Rule-based correlation features for hypothesis scoring.

        v0 heuristics:
          - correlates_with_recent_click_color: high if errors cluster
            after click actions (a ModeArmedBy candidate)
          - correlates_with_recent_key: high if errors cluster after a
            specific key (a FlagSetBy candidate)
          - correlates_with_action_count: how proportional errors are to
            cumulative click counts (a CounterIncrementedBy candidate)

        These are hand-tuned heuristics that should be replaced with
        learned residual-clustering when the architecture matures.
        # RL-TODO above.
        """
        click_errors = sum(
            1 for entry, act in zip(self._residual_buffer, self._recent_actions)
            if entry["had_error"] and action_kind(act) == "click"
        )
        total_clicks = sum(
            1 for act in self._recent_actions if action_kind(act) == "click"
        )
        key_errors = sum(
            1 for entry, act in zip(self._residual_buffer, self._recent_actions)
            if entry["had_error"] and action_kind(act) == "key"
        )
        total_keys = sum(
            1 for act in self._recent_actions if action_kind(act) == "key"
        )
        # Default: zero if no observations of that action class
        click_error_rate = (click_errors / total_clicks) if total_clicks > 0 else 0.0
        key_error_rate = (key_errors / total_keys) if total_keys > 0 else 0.0
        return {
            "correlates_with_recent_click_color": click_error_rate,
            "correlates_with_recent_key": key_error_rate,
            "correlates_with_action_count": 0.0,  # placeholder for v0
        }

    def extra_facts(self) -> set:
        """Facts to UNION into emit_atomic_facts output.

        These represent the brain's currently-believed latent state.
        """
        out: set = set()
        for hyp in self._latents:
            f = hyp.project()
            if f is not None:
                out.add(f)
        return out

    @property
    def latents(self) -> list[LatentHypothesis]:
        """For diagnostics."""
        return list(self._latents)
