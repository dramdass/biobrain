"""Probe — does the four-of-four architecture (with Planner course-
correction) score on cd82?

The full ARC-AGI-3 capability stack:
  1. EXPLORATION       (substrate Beta+Thompson with surprise injection)
  2. MODELING          (BayesianWorldModel; L1 residual)
  3. GOAL-SETTING      (proto_goals: pattern-recurrence detection)
  4. PLANNING+EXECUTION (this brain: goal-distance reward shaping)

Compares four brains:
  (1) Residual baseline (1+2)
  (2) Goal (1+2+3)
  (3) GoalDream (1+2+3 + DSL bridge)
  (4) Planner (1+2+3+4) — adds course-correction credit injection

Methodology: cd82 × N=20 × 200 steps. Track scoring rate +
goal-distance reduction per attempt.

Wall budget: ~10 min.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.disable(logging.CRITICAL)

from arena.env import ArenaEnv  # noqa: E402
from arena.perceive import detect_events, perceive  # noqa: E402
from arena.salience import Salience  # noqa: E402
from arena.types import ComputeBudget, Transition  # noqa: E402
from prism.residual_brain import MemoryBrainResidual  # noqa: E402
from prism.goal_brain import MemoryBrainGoal  # noqa: E402
from prism.dsl.goal_dream_brain import MemoryBrainGoalDream  # noqa: E402
from prism.planner_brain import MemoryBrainPlanner  # noqa: E402
from prism.goals import L3, TransitionHistory  # noqa: E402
from prism.proto_goals import state_distance_to_goals  # noqa: E402


GAME = "cd82"
N_ATTEMPTS = 20
MAX_STEPS = 200


def run_attempt(brain, env, salience, max_steps):
    brain.reset_attempt()
    obs = env.reset()
    prev_state = None
    last_action = None
    score_events = 0
    initial_d = None
    min_d = None
    final_d = None
    # Track our own history so the diagnostic mirrors what the brain sees
    diag_l3 = L3()
    diag_hist = TransitionHistory()

    for step in range(max_steps):
        if env.is_terminal(obs): break
        parsed = env.parse(obs)
        if parsed["grid"] is None: break
        avail = tuple(int(a) for a in parsed.get("available_actions", ()) or ())
        salience.observe(parsed["grid"])
        state = perceive(parsed["grid"], prev_state, score=parsed["score"],
                         level=parsed["levels_completed"], available_actions=avail,
                         salience_mask=salience.mask())
        # Track goal distance over time (using same L3 + history pattern
        # the brain itself uses, so the diagnostic is comparable).
        goals = diag_l3.detect(state, diag_hist)
        if goals:
            d = state_distance_to_goals(state, goals)
            if initial_d is None: initial_d = d
            final_d = d
            if min_d is None or d < min_d: min_d = d
        if prev_state is not None and last_action is not None:
            events = detect_events(prev_state, state)
            trans = Transition(before=prev_state, action=last_action,
                               after=state, events=events)
            brain.observe(trans)
            diag_hist.update(trans)
            for e in events:
                if e.kind in ("ScoreIncreased", "LevelIncreased"):
                    score_events += 1
        budget = ComputeBudget(actions_remaining=max_steps - step,
                               time_remaining_ms=10000, attempts_remaining=1)
        try:
            action = brain.act(state, budget)
        except ValueError:
            break
        obs = env.step(action)
        prev_state = state
        last_action = action
    brain.end_of_attempt()
    return {
        "score_events": score_events,
        "initial_d": initial_d,
        "min_d": min_d,
        "final_d": final_d,
    }


def measure(brain_factory, label, n_attempts):
    brain = brain_factory()
    brain.reset_game(GAME)
    salience = Salience()
    results = []
    for _ in range(n_attempts):
        try:
            env = ArenaEnv(GAME, mode="OFFLINE")
            r = run_attempt(brain, env, salience, MAX_STEPS)
            env.close()
            results.append(r)
        except Exception as e:
            print(f"    error {e}")
            continue
    n_scored = sum(1 for r in results if r["score_events"] > 0)
    rate = 100 * n_scored / max(1, len(results))
    initials = [r["initial_d"] for r in results if r["initial_d"] is not None]
    mins = [r["min_d"] for r in results if r["min_d"] is not None]
    avg_initial = sum(initials) / max(1, len(initials))
    avg_min = sum(mins) / max(1, len(mins))
    reduction = avg_initial - avg_min
    print(f"  {label}: scored {n_scored}/{len(results)} = {rate:.0f}%, "
          f"goal_dist {avg_initial:.3f} → min {avg_min:.3f} "
          f"(reduction {reduction:.3f}, {100*reduction/max(0.001, avg_initial):.1f}%)")


def main() -> int:
    print(f"Probe: 4-of-4 capabilities on {GAME} — N={N_ATTEMPTS} × {MAX_STEPS} steps")
    print()
    t0 = time.time()
    print("(1) Residual (Exploration + Modeling):")
    measure(lambda: MemoryBrainResidual(seed=0), "residual", N_ATTEMPTS)
    print()
    print("(2) Goal (1 + Goal-setting):")
    measure(lambda: MemoryBrainGoal(seed=0), "goal", N_ATTEMPTS)
    print()
    print("(3) GoalDream (1 + 2 + DSL programs):")
    measure(lambda: MemoryBrainGoalDream(seed=0), "goal_dream", N_ATTEMPTS)
    print()
    print("(4) Planner (1 + 2 + 3 + course-correction):")
    measure(lambda: MemoryBrainPlanner(seed=0), "planner", N_ATTEMPTS)
    print()
    print(f"Wall time: {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
