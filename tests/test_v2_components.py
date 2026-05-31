"""Unit tests for biobrain v0.2 — one test per component, plus integration."""

import pytest
from biobrain import BioBrainV2, CentralSalience
from biobrain.adapters.arc import ArcAdapter
from biobrain.curiosity.world_model import BayesianWorldModel
from biobrain.critic import Critic
from biobrain.ledger import Ledger
from biobrain.simulator import Simulator
from biobrain.perception.encoder import DefaultSpelkeEncoder
from biobrain.planner.commit_monitor import CommitMonitorPlanner


# ---------------------------------------------------------------- 1. Encoder

def test_encoder_default_coarse():
    """DefaultSpelkeEncoder emits coarse facts on a minimal state."""
    enc = DefaultSpelkeEncoder()

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
        entities = [MockE(5, [(10, 10)])]
        level = 0
        raw_grid = None

    facts = enc.encode(MockS())
    assert isinstance(facts, frozenset)
    assert ("entity_color", 5) in facts
    assert ("total_entities", 1) in facts


def test_encoder_resolve_click_on_color():
    enc = DefaultSpelkeEncoder()

    class R:
        def __init__(self, cells):
            self.cells = cells
            self.area = len(cells)

    class E:
        def __init__(self, color, cells):
            self.color = color
            self.id = id(self)
            self.velocity = (0, 0)
            self.region = R(cells)

    class S:
        entities = [E(7, frozenset({(5, 5)}))]
        level = 0
        raw_grid = None

    # click at (col=5, row=5) → resolves to click_on_color(7)
    candidates = [("click", 5, 5)]
    a = enc.resolve(("click_on_color", 7), S(), candidates)
    assert a == ("click", 5, 5)


# ---------------------------------------------------------------- 2. Curiosity (WM)

def test_world_model_predict_empty():
    wm = BayesianWorldModel()
    # No data → predict returns empty for any context
    class S:
        entities = []
        level = 0
    pred = wm.predict(S(), ("noop",))
    assert pred == {}


# ---------------------------------------------------------------- 3. CentralSalience

def test_central_salience_initial_state():
    s = CentralSalience()
    assert s.n_banked == 0
    # Default-active curated variables
    assert s.curated.is_active("entity_color")
    assert s.curated.is_active("entity_color_quadrant")
    # Affordance posterior starts empty (uniform via Beta(1,1))
    assert s.get_affordance("click") == 0.5  # uninformed → mean = 1/2


def test_central_salience_observe_updates_affordance():
    s = CentralSalience()
    # Bank multiple surprises on 'click' to shift posterior up
    for _ in range(10):
        s.observe(surprise=0.6,
                   context=("click", None, 0),
                   action=("click", 10, 10),
                   predicted_facts=frozenset(),
                   actual_facts=frozenset())
    # Affordance for 'click' should be > 0.5 now (surprise treated as informative)
    assert s.get_affordance("click") > 0.5


def test_central_salience_reset_lifecycle():
    s = CentralSalience()
    s.observe(surprise=0.6, context=("click", None, 0),
              action=("click", 1, 1), predicted_facts=frozenset(),
              actual_facts=frozenset())
    assert s.n_banked > 0
    # reset_attempt preserves
    s.reset_attempt()
    assert s.n_banked > 0
    # reset_game wipes
    s.reset_game()
    assert s.n_banked == 0


# ---------------------------------------------------------------- 4. Critic

def test_critic_returns_list():
    c = Critic()
    class MockE:
        color = 5
        id = "e1"
        velocity = (0, 0)
        class region:
            cells = frozenset({(1, 1)})
            area = 1

    class S:
        entities = [MockE()]
        level = 0
        raw_grid = None
        grid_hash = 0

    goals = c.evaluate(S())
    assert isinstance(goals, list)


# ---------------------------------------------------------------- 5. Simulator

def test_simulator_simulate_one_empty_wm():
    wm = BayesianWorldModel()
    sim = Simulator(wm)
    class S:
        entities = []
        level = 0
    # No WM evidence → no predicted facts
    preds = sim.simulate_one(S(), ("noop",))
    assert preds == set()


# ---------------------------------------------------------------- 6. Ledger

def test_ledger_starts_empty():
    L = Ledger()
    assert len(L) == 0
    assert L.confidence("nonexistent", 0) == 0.5


def test_ledger_reset_game_wipes():
    L = Ledger()
    assert len(L) == 0
    L.reset_game()
    assert len(L) == 0


# ---------------------------------------------------------------- 7. Planner (commit-and-monitor)

def test_commit_monitor_planner_lifecycle():
    p = CommitMonitorPlanner(seed=0)
    p.reset_game("test")
    p.reset_attempt()
    assert p.hot_call_count == 0
    assert p.cold_call_count == 0
    assert not p.violation_pending


def test_commit_monitor_violation_trigger():
    p = CommitMonitorPlanner(seed=0)
    # Low surprise → no violation
    p.observe(surprise=0.1, action_sig=("click", None, 0), scored=False)
    assert not p.violation_pending
    # High surprise → violation flag set
    p.observe(surprise=0.5, action_sig=("click", None, 0), scored=False)
    assert p.violation_pending


# ---------------------------------------------------------------- 8. BioBrainV2 integration

def test_biobrain_v2_instantiates():
    b = BioBrainV2(seed=0)
    assert b.critic is not None
    assert b.salience is not None
    assert b.simulator is not None
    assert b.ledger is not None
    assert b.planner is not None
    assert b.encoder is not None
    assert b.adapter is not None


def test_biobrain_v2_lifecycle():
    b = BioBrainV2(seed=0)
    b.reset_game("game_a")
    assert b.salience.n_banked == 0
    assert b.planner.hot_call_count == 0
    assert b.planner.cold_call_count == 0
    b.reset_attempt()
    # Still empty after attempt reset on fresh game
    assert b.salience.n_banked == 0


def test_biobrain_v2_with_custom_adapter():
    """Adapter slot accepts custom adapter with affordance prior."""
    class MyAdapter(ArcAdapter):
        def initial_affordance_priors(self):
            return {"click": (5.0, 2.0), "key": (3.0, 1.0)}

    b = BioBrainV2(seed=0, adapter=MyAdapter())
    # Affordance seeded from adapter
    assert b.salience.affordance.counts == {"click": (5.0, 2.0),
                                              "key": (3.0, 1.0)}
    # click affordance > 0.5 (alpha=5, beta=2)
    assert b.salience.get_affordance("click") > 0.5
