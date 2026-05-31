"""biobrain.motor_cortex — the action vocabulary.

Brain region: Motor cortex.
ML/RL term: Action space / DSL.

Wraps `biobrain.motor_cortex`. The Motor Cortex is the agent's "muscles" — atomic
environment actions plus combinators that chain them into multi-step
programs. The agent cannot invent new motor primitives; it can only
compose what's defined here.

Atoms (environment actions ARC-AGI-3 exposes):
    click_on_color(c)   — click any entity of color c
    key(k)              — keyboard input k ∈ {0..4}
    spacebar()          — spacebar input
    noop()              — pass

Combinators:
    SEQ(p1, p2)         — p1 then p2
    REPEAT(n, p)        — p run n times
    IF(cond, p_then, p_else)
    WHILE_NOT(cond, p)

Predicates (for IF/WHILE):
    has_color(c)        — state has entity of color c
    always(), never()

Asymmetry with idealized DSL: ARC-AGI-3 doesn't expose grid-transformation
primitives like rotate_90/flood_fill directly. Those are *learned effects*
of environment actions, discovered per-game by the world model.
"""

from __future__ import annotations

from biobrain.motor_cortex.core import (
    Program, Predicate,
    click_on_color, key, spacebar, noop,
    has_color, always, never,
    SEQ, REPEAT, IF, WHILE_NOT,
)

__all__ = [
    "Program", "Predicate",
    "click_on_color", "key", "spacebar", "noop",
    "has_color", "always", "never",
    "SEQ", "REPEAT", "IF", "WHILE_NOT",
]
