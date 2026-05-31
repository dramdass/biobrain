"""biobrain.motor_cortex.core — minimal compositional DSL.

Programs are State → (ActionSig, NextProgram | None) callables.
- ActionSig is a (kind, *params) tuple matching obs_action_sig output.
- NextProgram = None means "program done; brain should pick a new one."
- NextProgram = some Program means "use this for next step."

Combinators implemented as continuation-rewriting per the rubber-duck
refinement: SEQ(p1, p2).step returns (a, SEQ(p1', p2)) if p1 has more
to do, else delegates to p2.

The brain converts ActionSig → concrete Action by:
- ('click_on_color', c) → click at centroid of any entity with that color
- ('key', k)            → action_key(k)
- ('spacebar',)         → ('spacebar',)
- ('undo',)             → action_undo()
- ('noop',)             → brain falls back to Thompson over substrate

This way the DSL never has to compute exact click coordinates — the
brain handles concrete-action resolution from the signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


ActionSig = tuple  # e.g., ('click_on_color', 8), ('key', 1), ('spacebar',)


# ---------------------------------------------------------------------------
# Program type
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """Wrapper around a step function with metadata for MDL / display."""
    step_fn: Callable[[object], tuple[ActionSig, Optional['Program']]]
    repr_str: str
    dl: int  # description length for MDL prior

    def step(self, state):
        return self.step_fn(state)


# ---------------------------------------------------------------------------
# Primitive atoms
# ---------------------------------------------------------------------------

def click_on_color(c: int) -> Program:
    """Step: emits a click on any entity of color c. Done after one step."""
    def f(state):
        return (("click_on_color", c), None)
    return Program(step_fn=f, repr_str=f"click({c})", dl=2)


def key(k: int) -> Program:
    def f(state):
        return (("key", k), None)
    return Program(step_fn=f, repr_str=f"key({k})", dl=2)


def spacebar() -> Program:
    def f(state):
        return (("spacebar",), None)
    return Program(step_fn=f, repr_str="spacebar", dl=1)


def noop() -> Program:
    def f(state):
        return (("noop",), None)
    return Program(step_fn=f, repr_str="noop", dl=1)


# ---------------------------------------------------------------------------
# State predicates (for IF)
# ---------------------------------------------------------------------------

@dataclass
class Predicate:
    pred_fn: Callable[[object], bool]
    repr_str: str
    dl: int


def has_color(c: int) -> Predicate:
    def f(state):
        return any(int(e.color) == c for e in state.entities)
    return Predicate(pred_fn=f, repr_str=f"has({c})", dl=2)


def always() -> Predicate:
    return Predicate(pred_fn=lambda s: True, repr_str="True", dl=1)


def never() -> Predicate:
    return Predicate(pred_fn=lambda s: False, repr_str="False", dl=1)


# Trajectory-based predicates: query a buffer the brain provides.
# For the brain to use these, it needs to pass recent_action history.
# We support a simple recent-K mechanism via a thread-local buffer on
# the Program. For minimal impl, predicates use state-only.


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------

def IF(cond: Predicate, then_p: Program, else_p: Program) -> Program:
    def f(state):
        if cond.pred_fn(state):
            a, k = then_p.step(state)
        else:
            a, k = else_p.step(state)
        return (a, k)
    return Program(
        step_fn=f,
        repr_str=f"if {cond.repr_str} then {then_p.repr_str} else {else_p.repr_str}",
        dl=1 + cond.dl + then_p.dl + else_p.dl,
    )


def SEQ(p1: Program, p2: Program) -> Program:
    """Run p1 until it returns None continuation, then p2."""
    def f(state):
        a, k = p1.step(state)
        if k is None:
            # p1 done: next step is p2
            return (a, p2)
        # p1 has more: continuation = SEQ(p1's continuation, p2)
        return (a, SEQ(k, p2))
    return Program(
        step_fn=f,
        repr_str=f"seq({p1.repr_str}; {p2.repr_str})",
        dl=1 + p1.dl + p2.dl,
    )


def REPEAT(n: int, p: Program) -> Program:
    """Run p exactly n times."""
    if n <= 0:
        return noop()
    if n == 1:
        return p
    return SEQ(p, REPEAT(n - 1, p))


def WHILE_NOT(cond: Predicate, body: Program) -> Program:
    """Run body until cond becomes True."""
    def f(state):
        if cond.pred_fn(state):
            return (("noop",), None)  # done
        a, k = body.step(state)
        if k is None:
            # body done one iteration; loop continues with fresh body
            return (a, WHILE_NOT(cond, body))
        # body has more: keep current body running, then loop
        return (a, WHILE_NOT(cond, SEQ(k, body)))
    # Use original body's dl for the description length
    return Program(
        step_fn=f,
        repr_str=f"while_not {cond.repr_str}: {body.repr_str}",
        dl=1 + cond.dl + body.dl,
    )
