"""Probe — Ledger transfer on vc33 (the Level-2-reaching game).

vc33 is the one public game where the substrate reliably reaches Level 2
within attempt budgets. This is the cheapest probe of cross-level
program promotion (the Ledger's whole point).

Methodology:
  - Multiple long attempts (250-300 steps each).
  - Track: scoring rate, max-level reached, Ledger entries, on-level-change
    promotions, whether promoted programs actually scored on the new level.
  - A/B if requested: BioBrainV2 with vs without Ledger.

Pass criterion for "Ledger transfer works":
  - At least one attempt reaches maxL ≥ 2
  - On Level 2 entry, the Ledger surfaces ≥ 1 promoted Program
  - The promoted Program at least once scores or moves goal-distance

Usage:
    BIOBRAIN_ENV_DIR=/path/to/environment_files \\
    python bench/probe_ledger_transfer_vc33.py [n_attempts] [max_steps]
"""

import logging
import sys
import time
from collections import Counter

logging.disable(logging.CRITICAL)

from biobrain.perception.perceive import detect_events, perceive
from biobrain.perception.salience import Salience
from biobrain import BioBrainV2
from biobrain.adapters.arc import ArenaEnv
from biobrain.types import ComputeBudget, Transition


def run_attempt(seed: int, max_steps: int):
    env = ArenaEnv("vc33", mode="OFFLINE")
    brain = BioBrainV2(seed=seed)
    brain.reset_game("vc33")
    sal = Salience()
    brain.reset_attempt()
    obs = env.reset()
    prev = None
    last_a = None
    n_score = 0
    max_level = 0
    score_events_per_level = Counter()
    level_transitions = []
    promotions_per_level = {}

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
        if state.level > max_level:
            level_transitions.append((max_level, state.level, step))
            max_level = state.level
            # On level entry, check what the Ledger surfaces
            try:
                promotions = brain.ledger.promote_at_level(state.level)
                promotions_per_level[state.level] = len(promotions)
            except Exception:
                promotions_per_level[state.level] = 0
        if prev is not None and last_a is not None:
            events = detect_events(prev, state)
            brain.observe(Transition(before=prev, action=last_a,
                                      after=state, events=events))
            for e in events:
                if e.kind in ("ScoreIncreased", "LevelIncreased"):
                    n_score += 1
                    score_events_per_level[state.level] += 1
        a = brain.act(state, ComputeBudget(actions_remaining=max_steps - step,
                                              time_remaining_ms=10000,
                                              attempts_remaining=1))
        obs = env.step(a)
        prev = state
        last_a = a

    env.close()
    return {
        "seed": seed,
        "score_events": n_score,
        "max_level": max_level,
        "score_per_level": dict(score_events_per_level),
        "level_transitions": level_transitions,
        "promotions_per_level": promotions_per_level,
        "ledger_entries": len(brain.ledger),
        "hot_calls": brain.n_hot_calls,
        "cold_calls": brain.n_cold_calls,
        "banked_surprises": brain.salience.n_banked,
    }


def main():
    n_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print(f"Ledger transfer probe on vc33 — N={n_attempts} × {max_steps} steps")
    print(f"  goal: reach maxL ≥ 2 AND Ledger promotions fire on Level 2 entry")
    print()
    print(f"{'seed':>4s}  {'scored':>6s}  {'maxL':>4s}  "
          f"{'score/level':>16s}  {'promos/level':>14s}  "
          f"{'ledger':>6s}  {'hot':>4s}  {'cold':>4s}")
    print("-" * 90)
    t0 = time.time()
    results = []
    n_with_l2 = 0
    n_with_l2_promo = 0
    for i in range(n_attempts):
        r = run_attempt(seed=i, max_steps=max_steps)
        results.append(r)
        if r["max_level"] >= 2:
            n_with_l2 += 1
            if r["promotions_per_level"].get(2, 0) > 0:
                n_with_l2_promo += 1
        score_str = str(r["score_per_level"])
        promo_str = str(r["promotions_per_level"])
        print(f"  {r['seed']:>2d}  {r['score_events']:>6d}  "
              f"{r['max_level']:>4d}  {score_str:>16s}  "
              f"{promo_str:>14s}  "
              f"{r['ledger_entries']:>6d}  "
              f"{r['hot_calls']:>4d}  {r['cold_calls']:>4d}")
    print()
    print(f"Summary:")
    print(f"  Attempts reaching Level 2:           {n_with_l2}/{n_attempts}")
    print(f"  Attempts with Level-2 Ledger promo:  {n_with_l2_promo}/{n_attempts}")
    if n_with_l2_promo > 0:
        print(f"  ✓ LEDGER TRANSFER FIRED on at least one attempt")
    elif n_with_l2 > 0:
        print(f"  ⚠ Reached Level 2 but no promotions — investigate "
              f"(maybe confidence threshold too high?)")
    else:
        print(f"  ✗ Never reached Level 2; need more attempts or different game")
    print(f"  Wall time: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
