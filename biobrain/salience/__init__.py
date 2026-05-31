"""biobrain.salience — the curating / attending organ (the 8th component).

Per the phenomenological insight from playing ARC-AGI-3:
  - Salience CURATES the small variable set the brain models
  - Salience TRIGGERS finer perceptual attention at prediction failures
  - Salience PRIORITIZES probe order ("guess most useful first")

This makes Salience cortically central, not a perception utility.
"""
from biobrain.salience.salience import Salience, SalienceCurator
__all__ = ["Salience", "SalienceCurator"]
