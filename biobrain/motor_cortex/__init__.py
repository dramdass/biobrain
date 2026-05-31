"""biobrain.motor_cortex — DSL action vocabulary."""
from biobrain.motor_cortex.core import (
    Program, Predicate, ActionSig,
    click_on_color, key, spacebar, noop,
    has_color, always, never,
    SEQ, REPEAT, IF, WHILE_NOT,
)
__all__ = [
    "Program", "Predicate", "ActionSig",
    "click_on_color", "key", "spacebar", "noop",
    "has_color", "always", "never",
    "SEQ", "REPEAT", "IF", "WHILE_NOT",
]
