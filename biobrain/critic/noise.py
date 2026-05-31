"""biobrain.critic.noise — Noise / entropy-reduction extractor.

Aesthetic primitive: tiny scattered entities are visually "noisy"; states
with few large entities are "clean." Per the bp35 mechanic (click-to-
remove-barriers) and similar "eliminate the mess" levels, the goal is
explicitly noise reduction.

Mechanism:
  - Count tiny/small entities (from `count_size` facts in the extended
    predicate vocabulary).
  - Distance = fraction of entities that are tiny or small.
  - Goal-satisfied when no tiny entities remain.

Complements Compression: Compression shrinks the fact set globally; Noise
specifically targets small-cluster elimination as a cohesion-adjacent
signal. Both ultimately reduce DL.
"""

from __future__ import annotations

from biobrain.types import State
from biobrain.critic.base import GoalExtractor, ProtoGoal, TransitionHistory
from biobrain.curiosity.predicates import emit_atomic_facts


class Noise:
    """Goal: reduce tiny/small entity counts.

    Reads `count_size` facts emitted by the extended predicate vocabulary.
    """
    name = "noise"

    def detect(self, state: State,
               history: TransitionHistory) -> list[ProtoGoal]:
        if not state.entities:
            return []
        facts = emit_atomic_facts(None, state)
        # Extract count_size facts: ('count_size', bucket, n)
        size_counts: dict[str, int] = {}
        for f in facts:
            if isinstance(f, tuple) and len(f) == 3 and f[0] == "count_size":
                size_counts[f[1]] = int(f[2])
        n_tiny = size_counts.get("tiny", 0)
        n_small = size_counts.get("small", 0)
        total = len(state.entities)
        if total == 0:
            return []
        noisy_fraction = (n_tiny + 0.5 * n_small) / total
        if noisy_fraction < 0.05:
            # Already clean — no goal needed
            return []

        def distance_fn(s_or_facts):
            if isinstance(s_or_facts, (set, frozenset)):
                f = s_or_facts
                # Recover total_entities from the fact set
                total = 0
                for x in f:
                    if isinstance(x, tuple) and len(x) == 2 and x[0] == "total_entities":
                        total = int(x[1]); break
            else:
                if not s_or_facts.entities:
                    return 0.0
                f = emit_atomic_facts(None, s_or_facts)
                total = len(s_or_facts.entities)
            nt = ns = 0
            for x in f:
                if isinstance(x, tuple) and len(x) == 3 and x[0] == "count_size":
                    if x[1] == "tiny": nt = int(x[2])
                    elif x[1] == "small": ns = int(x[2])
            return (nt + 0.5 * ns) / total if total > 0 else 0.0

        return [ProtoGoal(
            goal_id="noise:eliminate_small",
            description=(
                f"noise: eliminate tiny/small entities "
                f"(tiny={n_tiny}, small={n_small} of {total})"
            ),
            distance_fn=distance_fn,
            weight=min(1.0, noisy_fraction),
            source=self.name,
        )]
