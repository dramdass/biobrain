"""Tests for the Spelke-axis predicate vocabulary."""

import pytest
from biobrain.curiosity.predicates import emit_atomic_facts


def test_emit_no_before():
    """emit_atomic_facts works with no before state (initial state)."""
    class MockEntity:
        def __init__(self, color, cells):
            self.color = color
            self.id = id(self)
            self.velocity = (0, 0)
            class R:
                area = len(cells)
            self.region = R()
            self.region.cells = cells

    class MockState:
        entities = [MockEntity(5, [(10, 10), (10, 11)])]
        level = 0

    facts = emit_atomic_facts(None, MockState())
    # Should emit entity_color, entity_size, entity_quadrant, plus joints,
    # plus count_color, count_size, count_quadrant, total_entities, level
    assert ('entity_color', 5) in facts
    assert ('total_entities', 1) in facts
    assert ('level', 0) in facts
    # New principled joint
    assert any(f[0] == 'entity_color_quadrant' for f in facts if isinstance(f, tuple))
    assert any(f[0] == 'entity_color_size' for f in facts if isinstance(f, tuple))
    # New count predicates
    assert any(f[0] == 'count_size' for f in facts if isinstance(f, tuple))
    assert any(f[0] == 'count_quadrant' for f in facts if isinstance(f, tuple))
