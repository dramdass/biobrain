"""Smoke tests for the biobrain scaffold.

Verifies the package imports cleanly, the BioBrain can be constructed
and exercised through observe/act on a real game without errors.
"""

from __future__ import annotations

import logging

import pytest

from arena.env import ArenaEnv
from arena.perceive import detect_events, perceive
from arena.salience import Salience
from arena.types import ComputeBudget, Transition
from biobrain import (
    BioBrain, Critic, Curiosity, Ledger, Simulator, Planner,
    click_on_color, key, spacebar, SEQ,
)


def test_imports():
    """All named components importable."""
    assert BioBrain is not None
    assert Critic is not None
    assert Curiosity is not None
    assert Ledger is not None
    assert Simulator is not None
    assert Planner is not None


def test_motor_cortex_dsl():
    """DSL primitives compose into Programs."""
    p = SEQ(click_on_color(5), key(3))
    assert p is not None
    assert p.dl > 0


def test_ledger_empty_initial_state():
    """Ledger starts empty; confidence defaults to 0.5 (uninformative)."""
    L = Ledger()
    assert len(L) == 0
    assert L.confidence("does_not_exist", level=0) == 0.5
    assert L.promote_at_level(level=0) == []


def test_biobrain_lifecycle():
    """BioBrain reset/observe/act on a real game (smoke test)."""
    logging.disable(logging.CRITICAL)
    brain = BioBrain(seed=0)
    brain.reset_game("vc33")
    brain.reset_attempt()

    env = ArenaEnv("vc33", mode="OFFLINE")
    sal = Salience()
    obs = env.reset()
    prev_state = None
    last_action = None

    for step in range(10):
        if env.is_terminal(obs):
            break
        parsed = env.parse(obs)
        if parsed["grid"] is None:
            break
        avail = tuple(int(a) for a in parsed.get("available_actions", ()) or ())
        sal.observe(parsed["grid"])
        state = perceive(
            parsed["grid"], prev_state, score=parsed["score"],
            level=parsed["levels_completed"],
            available_actions=avail, salience_mask=sal.mask(),
        )
        if prev_state is not None and last_action is not None:
            events = detect_events(prev_state, state)
            brain.observe(Transition(before=prev_state, action=last_action,
                                      after=state, events=events))
        action = brain.act(state, ComputeBudget(actions_remaining=100,
                                                  time_remaining_ms=5000,
                                                  attempts_remaining=1))
        assert action is not None
        obs = env.step(action)
        prev_state = state
        last_action = action

    brain.end_of_attempt()
    env.close()


def test_critic_evaluate_returns_list():
    """Critic returns ProtoGoal list (possibly empty) on a real state."""
    logging.disable(logging.CRITICAL)
    env = ArenaEnv("cd82", mode="OFFLINE")
    obs = env.reset()
    parsed = env.parse(obs)
    sal = Salience()
    sal.observe(parsed["grid"])
    state = perceive(
        parsed["grid"], None, score=0, level=0,
        available_actions=tuple(parsed.get("available_actions", ()) or ()),
        salience_mask=sal.mask(),
    )
    critic = Critic()
    goals = critic.evaluate(state)
    assert isinstance(goals, list)
    # cd82 has detectable static-pattern goal even at step 0
    assert len(goals) >= 1
    env.close()
