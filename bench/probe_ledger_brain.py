"""Probe — does the Ledger (scientific method) lift multi-level games?

A/B test:
  (1) Lookahead — MemoryBrainLookahead (substrate + L1 + Critic + 1-step
                   lookahead over fact-space)
  (2) Ledger    — same + Ledger (trajectory abstraction on score events,
                   cross-level program promotion)

Test games: vc33, lp85, r11l. All three score on substrate alone and
have multi-level structure → Ledger CAN potentially transfer the
mechanic from Level 1 to Level 2+. cd82 added as a litmus.

Methodology: N=15 attempts × 250 steps per (brain, game). Track scoring
rate, max-level-reached, ledger-entry-count at end.

If Ledger is doing useful work:
  - max_level_reached should be HIGHER (cross-level transfer)
  - scoring rate per attempt should be higher
  - ledger should accumulate entries that match the underlying mechanic

Wall budget: ~12 min.
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
from prism.lookahead_planner import MemoryBrainLookahead  # noqa: E402
from prism.ledger_brain import MemoryBrainLedger  # noqa: E402


GAMES = ("vc33", "lp85", "r11l", "cd82")
N_ATTEMPTS = 15
MAX_STEPS = 250


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
    # Extra: Ledger-specific diagnostics
    n_ledger = 0
    if hasattr(brain, "ledger"):
        n_ledger = len(brain.ledger)
    return {"label": label, "rate": rate, "n_scored": n_scored,
            "n": len(results), "max_level": max_level, "n_ledger": n_ledger}


def main() -> int:
    print(f"Probe: Ledger A/B — N={N_ATTEMPTS} × {MAX_STEPS} steps")
    print()
    print(f"{'game':>6s}  {'brain':>11s}  {'scored':>10s}  "
          f"{'rate':>5s}  {'maxL':>4s}  {'#ledger':>7s}")
    print("-" * 60)
    t0 = time.time()
    for game in GAMES:
        for factory, label in [
            (lambda: MemoryBrainLookahead(seed=0), "Lookahead"),
            (lambda: MemoryBrainLedger(seed=0), "Ledger"),
        ]:
            r = measure(factory, label, game, N_ATTEMPTS)
            print(f"  {game:>6s}  {label:>11s}  "
                  f"{r['n_scored']:>2d}/{r['n']:<2d}      "
                  f"{r['rate']:>4.0f}%  {r['max_level']:>4d}  "
                  f"{r['n_ledger']:>7d}")
        print()
    print(f"Wall time: {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
