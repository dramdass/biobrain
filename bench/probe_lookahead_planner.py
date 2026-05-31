"""Probe — does 1-step forward-simulation lookahead lift action selection?

A/B test:
  (1) Planner       — MemoryBrainPlanner (substrate + L1 + course-correction)
  (2) Lookahead     — same + 1-step forward-simulation bonus per candidate

Test games:
  vc33   — F1=0.82 WM accuracy; substrate scores; should benefit
  lp85   — F1=0.94 WM accuracy; substrate scores; should benefit
  cd82   — F1=0.73 WM accuracy; substrate floors at 0%; lookahead is biggest
           potential lever here
  g50t   — F1=0.86 WM accuracy; "neither"-set game (lookahead novel ground)
  bp35   — F1=0.58 WM accuracy; floored substrate; weak lookahead signal

Methodology: N=15 attempts × 200 steps per (brain, game). Track scoring
rate + goal-distance reduction.

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
from prism.planner_brain import MemoryBrainPlanner  # noqa: E402
from prism.lookahead_planner import MemoryBrainLookahead  # noqa: E402


GAMES = ("vc33", "lp85", "cd82", "g50t", "bp35")
N_ATTEMPTS = 15
MAX_STEPS = 200


def run_attempt(brain, env, salience, max_steps):
    brain.reset_attempt()
    obs = env.reset()
    prev_state = None
    last_action = None
    score_events = 0
    max_level = 0
    for step in range(max_steps):
        if env.is_terminal(obs):
            break
        parsed = env.parse(obs)
        if parsed["grid"] is None:
            break
        avail = tuple(int(a) for a in parsed.get("available_actions", ()) or ())
        salience.observe(parsed["grid"])
        state = perceive(parsed["grid"], prev_state, score=parsed["score"],
                         level=parsed["levels_completed"], available_actions=avail,
                         salience_mask=salience.mask())
        max_level = max(max_level, state.level)
        if prev_state is not None and last_action is not None:
            events = detect_events(prev_state, state)
            trans = Transition(before=prev_state, action=last_action,
                               after=state, events=events)
            brain.observe(trans)
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
    return {"score_events": score_events, "max_level": max_level}


def measure(brain_factory, label, game, n_attempts):
    brain = brain_factory()
    brain.reset_game(game)
    salience = Salience()
    results = []
    for _ in range(n_attempts):
        try:
            env = ArenaEnv(game, mode="OFFLINE")
            r = run_attempt(brain, env, salience, MAX_STEPS)
            env.close()
            results.append(r)
        except Exception:
            continue
    n_scored = sum(1 for r in results if r["score_events"] > 0)
    max_level = max((r["max_level"] for r in results), default=0)
    rate = 100 * n_scored / max(1, len(results))
    return {"label": label, "rate": rate, "n_scored": n_scored,
            "n": len(results), "max_level": max_level}


def main() -> int:
    print(f"Probe: 1-step lookahead A/B — N={N_ATTEMPTS} × {MAX_STEPS} steps")
    print()
    print(f"{'game':>6s}  {'brain':>11s}  {'scored':>10s}  {'rate':>5s}  {'maxL':>4s}")
    print("-" * 50)
    t0 = time.time()
    for game in GAMES:
        for factory, label in [
            (lambda: MemoryBrainPlanner(seed=0), "Planner"),
            (lambda: MemoryBrainLookahead(seed=0), "Lookahead"),
        ]:
            r = measure(factory, label, game, N_ATTEMPTS)
            print(f"  {game:>6s}  {label:>11s}  "
                  f"{r['n_scored']:>2d}/{r['n']:<2d}      "
                  f"{r['rate']:>4.0f}%  {r['max_level']:>4d}")
        print()
    print(f"Wall time: {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
