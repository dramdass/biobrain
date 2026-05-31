"""biobrain — Active Inference Architecture for ARC-AGI-3.

See APPROACH.md for the canonical architecture overview and the bio-RL
→ ML/RL component mapping.

Public API:

    from biobrain import BioBrain
    from biobrain import Critic, Curiosity, Ledger, Simulator, Planner
    from biobrain import click_on_color, key, spacebar, SEQ
"""

from biobrain.brain import BioBrain
from biobrain.critic import Critic
from biobrain.curiosity import Curiosity
from biobrain.ledger import Ledger, LedgerEntry
from biobrain.simulator import Simulator
from biobrain.planner.planner_facade import Planner
from biobrain.motor_cortex import (
    Program, Predicate,
    click_on_color, key, spacebar, noop,
    has_color, always, never,
    SEQ, REPEAT, IF, WHILE_NOT,
)

__version__ = "0.1.0"

__all__ = [
    "BioBrain",
    "Critic", "Curiosity", "Ledger", "LedgerEntry",
    "Planner", "Simulator",
    "Program", "Predicate",
    "click_on_color", "key", "spacebar", "noop",
    "has_color", "always", "never",
    "SEQ", "REPEAT", "IF", "WHILE_NOT",
]
