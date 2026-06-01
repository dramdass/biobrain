"""biobrain.salience — the central representation-state-management organ.

Subsumes role discovery (Latent Inference in v0.2). Per God Mode §2.3:
salience curates variables, banks surprises, requests fine attention,
proposes new predicate templates, AND discovers entity roles for
cross-level transfer.
"""
from biobrain.salience.salience import Salience  # legacy salience-mask utility
from biobrain.salience.central import (
    CentralSalience, BankedSurprise, AffordancePosterior, CuratedVariables,
)
from biobrain.salience.roles import (
    Role, RoleSignature, ROLE_CATALOGUE, role_likelihood, assign_role,
)
from biobrain.salience.fingerprint import (
    Fingerprint, compute_fingerprint, RoleFingerprintIndex,
)
from biobrain.salience.subgoals import Subgoal, SubgoalDetector

__all__ = [
    "CentralSalience", "Salience",
    "BankedSurprise", "AffordancePosterior", "CuratedVariables",
    "Role", "RoleSignature", "ROLE_CATALOGUE",
    "role_likelihood", "assign_role",
    "Fingerprint", "compute_fingerprint", "RoleFingerprintIndex",
    "Subgoal", "SubgoalDetector",
]
