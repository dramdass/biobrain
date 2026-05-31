"""biobrain.critic — heuristic value function over win-state aesthetics.

A library of GoalExtractors. Each emits ProtoGoals — soft state-distance
functions the planner consumes for action selection.

See PRINCIPLES.md: the Critic is NOT a single objective function. It is
a disjunction of goal-detectors. Compression-progress is the dominant
motif, but reference-matching (ChangeDynamics family) is a parallel
motif. Future motifs (growth, attractor convergence) will be added as
the evidence demands.
"""

from biobrain.critic.base import (
    GoalExtractor, ProtoGoal, TransitionHistory, state_distance_to_goals,
)
from biobrain.critic.l3 import L3, default_extractors
from biobrain.critic.compression import Compression
from biobrain.critic.noise import Noise
from biobrain.critic.symmetry import Symmetry
from biobrain.critic.change_dynamics_facts import ChangeDynamicsFactSpace
from biobrain.critic.change_dynamics import ChangeDynamics
from biobrain.critic.pattern_recurrence import StaticPatternRecurrence
from biobrain.critic.critic_facade import Critic

__all__ = [
    "Critic", "L3", "default_extractors",
    "ProtoGoal", "GoalExtractor", "TransitionHistory",
    "state_distance_to_goals",
    "Compression", "Noise", "Symmetry",
    "ChangeDynamicsFactSpace", "ChangeDynamics", "StaticPatternRecurrence",
]
