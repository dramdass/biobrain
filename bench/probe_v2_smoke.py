"""Probe — BioBrainV2 smoke test on a single game.

Usage:
    BIOBRAIN_ENV_DIR=/path/to/environment_files python bench/probe_v2_smoke.py [game]

Default game: vc33 (the most reliable scorer in our prior measurements).
"""

import logging
import sys
import time

logging.disable(logging.CRITICAL)

from biobrain.perception.perceive import detect_events, perceive
from biobrain.perception.salience import Salience
from biobrain import BioBrainV2
from biobrain.adapters.arc import ArenaEnv
from biobrain.types import ComputeBudget, Transition


def run_attempt(game: str, max_steps: int, seed: int):
    env = ArenaEnv(game, mode="OFFLINE")
    brain = BioBrainV2(seed=seed)
    brain.reset_game(game)
    sal = Salience()
    brain.reset_attempt()
    obs = env.reset()
    prev = None
    last_a = None
    n_score = 0
    max_level = 0
    for step in range(max_steps):
        if env.is_terminal(obs):
            break
        parsed = env.parse(obs)
        if parsed["grid"] is None:
            break
        avail = tuple(int(a) for a in parsed.get("available_actions") or ())
        sal.observe(parsed["grid"])
        state = perceive(parsed["grid"], prev,
                         score=parsed["score"],
                         level=parsed["levels_completed"],
                         available_actions=avail,
                         salience_mask=sal.mask())
        max_level = max(max_level, state.level)
        if prev is not None and last_a is not None:
            events = detect_events(prev, state)
            brain.observe(Transition(before=prev, action=last_a,
                                      after=state, events=events))
            for e in events:
                if e.kind in ("ScoreIncreased", "LevelIncreased"):
                    n_score += 1
        a = brain.act(state, ComputeBudget(actions_remaining=max_steps - step,
                                              time_remaining_ms=10000,
                                              attempts_remaining=1))
        obs = env.step(a)
        prev = state
        last_a = a
    env.close()
    return {
        "game": game,
        "score_events": n_score,
        "max_level": max_level,
        "hot_calls": brain.n_hot_calls,
        "cold_calls": brain.n_cold_calls,
        "ledger_entries": len(brain.ledger),
        "banked_surprises": brain.salience.n_banked,
    }


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "vc33"
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    print(f"Probe: BioBrainV2 smoke on {game}, "
          f"N={n_attempts} × {max_steps} steps")
    print()
    t0 = time.time()
    scored = 0
    max_levels = []
    for i in range(n_attempts):
        r = run_attempt(game, max_steps, seed=i)
        max_levels.append(r["max_level"])
        if r["score_events"] > 0:
            scored += 1
        print(f"  attempt {i}: scored={r['score_events']} "
              f"maxL={r['max_level']} hot={r['hot_calls']} "
              f"cold={r['cold_calls']} ledger={r['ledger_entries']} "
              f"banked={r['banked_surprises']}")
    print()
    print(f"Summary: {scored}/{n_attempts} attempts scored. "
          f"maxL across all: {max(max_levels)}. "
          f"Wall: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
