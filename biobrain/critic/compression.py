"""biobrain.critic.compression — Compression extractor.

Aesthetic primitive: states with fewer unique facts compress better.
The compression objective (Chollet/Schmidhuber) operationalized for
ARC-AGI-3: prefer transitions that reduce predicate cardinality.

Two flavors of compression detected:
  1. Total-fact compression: emit a goal whose distance = (n_facts / cap).
     Lower n_facts → state closer to "minimal description."
  2. Color-palette compression: emit a goal whose distance scales with
     number of distinct active colors. Many ARC-AGI-3 levels reduce to
     a single-color or two-color final state.

The Compression extractor is the most direct expression of the
compression-as-critic principle — a state's "win likelihood" rises as
its predicate representation shrinks.
"""

from __future__ import annotations

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory
from biobrain.curiosity.predicates import emit_atomic_facts


# Caps for normalizing distance into [0, 1]. Derived from observed bounds:
# typical states emit 20-50 facts; total entities 5-30.
N_FACTS_CAP = 80.0
N_COLORS_CAP = 12.0  # 14 usable colors; bg + ~10 active is typical max


class Compression:
    """Goal: reduce predicate cardinality (total facts and color count).

    Always-on extractor; emits goals from any state with ≥1 entity.
    Goals are state-agnostic (no region pairing) — the gradient is
    "shrink the fact set."
    """
    name = "compression"

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        if not state.entities:
            return []
        # Compute current cardinalities from emitted facts
        facts = emit_atomic_facts(None, state)
        n_facts = len(facts)
        n_colors = sum(1 for f in facts if isinstance(f, tuple)
                       and len(f) >= 1 and f[0] == "entity_color")

        # Distance scales with cardinality, normalized by cap. Clamp to [0, 1].
        # distance_fn accepts either a State or a fact set (the latter for
        # lookahead's predicted-state evaluation).
        def fact_distance(s_or_facts, n_facts_cap=N_FACTS_CAP):
            if isinstance(s_or_facts, (set, frozenset)):
                f = s_or_facts
            else:
                if not s_or_facts.entities:
                    return 0.0
                f = emit_atomic_facts(None, s_or_facts)
            return min(1.0, len(f) / n_facts_cap)

        def color_distance(s_or_facts, n_colors_cap=N_COLORS_CAP):
            if isinstance(s_or_facts, (set, frozenset)):
                f = s_or_facts
            else:
                if not s_or_facts.entities:
                    return 0.0
                f = emit_atomic_facts(None, s_or_facts)
            nc = sum(1 for x in f if isinstance(x, tuple)
                     and len(x) >= 1 and x[0] == "entity_color")
            return min(1.0, nc / n_colors_cap)

        goals = [
            ProtoGoal(
                goal_id="compression:total_facts",
                description=(
                    f"compression: shrink fact set (currently {n_facts})"
                ),
                distance_fn=fact_distance,
                weight=min(1.0, n_facts / N_FACTS_CAP),
                source=self.name,
            ),
            ProtoGoal(
                goal_id="compression:color_palette",
                description=(
                    f"compression: shrink color palette (currently {n_colors})"
                ),
                distance_fn=color_distance,
                weight=min(1.0, n_colors / N_COLORS_CAP),
                source=self.name,
            ),
        ]
        return goals
