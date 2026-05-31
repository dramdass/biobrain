"""biobrain.latent_inference — state-representation evolution (the 7th component).

When the World Model has persistent prediction error at a context, this
module hypothesizes that a hidden state variable (latent fact) explains
the residual variance. It registers the latent into the predicate
vocabulary; the WM and Critic consume it from then on.

This is the imperfect-but-tractable v0: schema-driven hypothesis from a
small library of latent templates. The brain doesn't invent
free-form representations; it instantiates from a bounded set of
cognitively-grounded templates.

See docs/COMPONENTS.md for the full design + rationale and
APPROACH.md for why this is load-bearing for cd82-class games.
"""

from biobrain.latent_inference.schemas import (
    LatentSchema, ModeArmedBySchema, FlagSetBySchema,
    CounterIncrementedBySchema, DEFAULT_SCHEMAS,
)
from biobrain.latent_inference.inference import (
    LatentInference, LatentHypothesis,
)

__all__ = [
    "LatentInference", "LatentHypothesis",
    "LatentSchema", "ModeArmedBySchema", "FlagSetBySchema",
    "CounterIncrementedBySchema", "DEFAULT_SCHEMAS",
]
