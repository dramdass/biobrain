"""biobrain.planner.planner_brain — the fourth ARC-AGI-3 capability.

Per the ARC-AGI-3 framework's four capabilities (Exploration, Modeling,
Goal-setting, Planning and Execution), the fourth — course-correction
in response to environmental feedback — is what our prior brains lack.

Mechanism: goal-distance reward shaping.

At observe(), the brain computes:
  ΔDistance = distance_to_goals(before) - distance_to_goals(after)

If ΔDistance > 0 (action moved closer to a proto-goal), inject positive
credit to the action's signature. If ΔDistance < 0 (moved away), inject
negative credit. This is TD-style learning over the proto-goal value
function.

This is course-correction in the literal sense: the brain sees the
ENVIRONMENTAL FEEDBACK (state change) and ADJUSTS its action posterior
accordingly. Subsequent Thompson sampling reflects the updated posterior;
the brain's behavior changes course step-by-step based on outcomes.

Note: this is intentionally simple — no forward simulation, no multi-step
planning. The principle is that Thompson over a substrate that's
posterior-shaped by goal-distance signals IS reactive planning.
Each act() is a fresh re-plan; commitment is one step at a time.

If this scores on cd82-class games: the four capabilities suffice.
If not: cd82 needs richer exploration upstream (the candidate action set
doesn't reach the cells that produce goal-distance reduction at all,
making this layer's reward signal degenerate to zero).
"""

from __future__ import annotations

from biobrain.types import Transition
from biobrain.planner.posterior import ActionScoreTable
from biobrain.planner.goal_brain import MemoryBrainGoal
from biobrain.critic.base import state_distance_to_goals
from biobrain.curiosity.residual import SURPRISE_CLIP


class MemoryBrainPlanner(MemoryBrainGoal):
    """Goal-distance reward shaping over MemoryBrainGoal.

    Inherits act() (goal-cell bias) from MemoryBrainGoal; inherits
    surprise injection from MemoryBrainResidual. Adds course-correction
    credit at observe(): action gets +credit if it reduced distance to
    any active proto-goal, -credit if it increased distance.

    The fourth capability instantiated as a symbolic TD update.
    """

    def observe(self, transition: Transition) -> None:
        # 1. Substrate update + surprise injection (inherited)
        super().observe(transition)
        # 2. Course-correction: detect proto-goals on before-state via
        #    the brain's L3 instance (with history), then compute distance
        #    delta. Use BEFORE-state goals so the goal set is stable
        #    across the transition.
        # IMPORTANT: history was already updated in super().observe(),
        # so the goals reflect this transition's dynamics signal too.
        # That's fine — extractors that need history will still see a
        # consistent snapshot.
        if transition.before is None:
            return
        goals_before = self._l3.detect(transition.before, self._history)
        if not goals_before:
            return
        d_before = state_distance_to_goals(transition.before, goals_before)
        d_after = state_distance_to_goals(transition.after, goals_before)
        delta = d_before - d_after  # positive = moved CLOSER to goals
        if abs(delta) < 1e-3:
            return
        # Clip ΔDistance to the same scale as surprise credit so the two
        # signals contribute on equal footing. No multiplier; the raw
        # delta IS the credit magnitude, bounded by SURPRISE_CLIP.
        credit = max(-SURPRISE_CLIP, min(SURPRISE_CLIP, delta))
        sig = ActionScoreTable._signature(transition.action, transition.before)
        n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
        if credit >= 0:
            self._action_table.counts[sig] = (n_obs, n_goal + credit)
        else:
            self._action_table.counts[sig] = (n_obs + (-credit), n_goal)
