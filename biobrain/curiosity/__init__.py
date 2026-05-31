"""biobrain.curiosity — ICM-style intrinsic reward via Bayesian world model."""
from biobrain.curiosity.world_model import BayesianWorldModel
from biobrain.curiosity.predicates import emit_atomic_facts, Fact
from biobrain.curiosity.residual import MemoryBrainResidual, SURPRISE_CLIP
from biobrain.curiosity.curiosity_facade import Curiosity
__all__ = ["Curiosity", "BayesianWorldModel", "MemoryBrainResidual",
           "emit_atomic_facts", "Fact", "SURPRISE_CLIP"]
