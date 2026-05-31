"""biobrain — Active Inference Architecture for ARC-AGI-3.

See docs/DESIGN.md for the v0.2 architecture (8 components, generic-typed,
commit-and-monitor control loop).

Public API:

    # v0.2 — the canonical brain (8 components, commit-and-monitor)
    from biobrain import BioBrainV2

    # v0.1 — legacy composer kept for transition (will be deprecated)
    from biobrain import BioBrain

    # Components (for diagnostics + custom composition)
    from biobrain import Critic, Curiosity, Ledger, Simulator
    from biobrain import CentralSalience

    # DSL atoms
    from biobrain import click_on_color, key, spacebar, SEQ

    # Adapter
    from biobrain.adapters.arc import ArcAdapter
"""

from biobrain.brain import BioBrain
from biobrain.brain_v2 import BioBrainV2
from biobrain.critic import Critic
from biobrain.curiosity import Curiosity
from biobrain.ledger import Ledger, LedgerEntry
from biobrain.salience import CentralSalience
from biobrain.simulator import Simulator
from biobrain.planner.planner_facade import Planner
from biobrain.motor_cortex import (
    Program, Predicate,
    click_on_color, key, spacebar, noop,
    has_color, always, never,
    SEQ, REPEAT, IF, WHILE_NOT,
)

__version__ = "0.2.0"

__all__ = [
    "BioBrain", "BioBrainV2",
    "Critic", "Curiosity", "Ledger", "LedgerEntry",
    "CentralSalience",
    "Planner", "Simulator",
    "Program", "Predicate",
    "click_on_color", "key", "spacebar", "noop",
    "has_color", "always", "never",
    "SEQ", "REPEAT", "IF", "WHILE_NOT",
]
