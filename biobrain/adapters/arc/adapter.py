"""biobrain.adapters.arc.adapter — the ARC-AGI-3 adapter object.

Implements the `biobrain.protocols.Adapter` protocol. Plugs into BioBrain
at construction time; brain library reads `adapter.encoder` and (optionally)
`adapter.initial_affordance_priors()`.

Strict covenant-respecting default: no affordance priors supplied.
"""

from __future__ import annotations

from biobrain.perception.encoder import DefaultSpelkeEncoder
from biobrain.protocols import Encoder


class ArcAdapter:
    """ARC-AGI-3 adapter for biobrain.

    Currently supplies just the Encoder. The arc_agi SDK env binding
    (ArenaEnv) is consumed by bench/probes directly, not via the brain
    library, since the brain library only takes State + Transition as
    input.
    """

    def __init__(self, encoder: Encoder | None = None) -> None:
        self.encoder: Encoder = encoder or DefaultSpelkeEncoder()

    def initial_affordance_priors(self) -> dict[str, tuple[float, float]]:
        """Strict covenant-respecting default: no priors.

        Returns empty dict; brain seeds affordance posterior uniformly.
        Override this in a subclass if covenant-relaxation is desired.
        """
        return {}


__all__ = ["ArcAdapter"]
