"""Smoke tests for the biobrain scaffold.

Verifies the package imports cleanly and the legacy BioBrain composer
can be constructed without errors. Uses biobrain's own perception layer
(no spelke imports).

End-to-end tests against the real env live in tests/test_env_adapter.py
and bench/probe_v2_*.py.
"""

from __future__ import annotations

import logging

import pytest

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


def test_biobrain_v1_instantiates():
    """Legacy BioBrain v1 composer constructs and resets cleanly."""
    logging.disable(logging.CRITICAL)
    brain = BioBrain(seed=0)
    brain.reset_game("smoke")
    brain.reset_attempt()
    assert brain.critic is not None
    assert brain.curiosity is not None
    assert brain.ledger is not None
    assert brain.simulator is not None
    assert brain.planner is not None


def test_critic_evaluate_returns_list_synthetic():
    """Critic returns ProtoGoal list on a synthetic state."""
    class MockE:
        def __init__(self, color, cells):
            self.color = color
            self.id = id(self)
            self.velocity = (0, 0)
            class R:
                area = len(cells)
            self.region = R()
            self.region.cells = cells

    class MockS:
        entities = [MockE(5, [(10, 10), (10, 11)])]
        level = 0
        raw_grid = None
        grid_hash = 0

    critic = Critic()
    goals = critic.evaluate(MockS())
    assert isinstance(goals, list)
