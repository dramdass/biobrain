"""biobrain.salience — the central representation-state-management organ.

Per the phenomenology (user playing ARC-AGI-3): Salience is not a perception
utility. It's the central organ that curates the modeled variable set,
triggers finer perception at prediction failures, banks salient surprises,
and proposes new predicate templates (observable first, latent fallback).

Subsumes the prior Latent Inference module.
"""
from biobrain.salience.salience import Salience  # legacy salience-mask utility
from biobrain.salience.central import (
    CentralSalience, BankedSurprise, AffordancePosterior, CuratedVariables,
)
__all__ = ["CentralSalience", "Salience",
           "BankedSurprise", "AffordancePosterior", "CuratedVariables"]
