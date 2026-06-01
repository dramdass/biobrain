"""Probe — Ledger transfer on vc33 with PERSISTENT brain.

vc33 is the one public game where the substrate reliably reaches Level 2.
This probe uses a single BioBrainV2 instance across all attempts, so the
intra-game memory (Ledger, WM, substrate posterior) accumulates — which
is what the House model lifecycle was designed for.

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


def run_attempt(brain, sal, attempt_idx: int, max_steps: int):
    """Run one attempt with a persistent brain."""
    env = ArenaEnv("vc33", mode="OFFLINE")
    brain.reset_attempt()
    obs = env.reset()
    prev = None
    last_a = None
    n_score = 0
    max_level = 0
    score_events_per_level = Counter()
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
        # Observe transition BEFORE checking promotions (so the score-event
        # has been processed by the Ledger).
        if prev is not None and last_a is not None:
            events = detect_events(prev, state)
            brain.observe(Transition(before=prev, action=last_a,
                                      after=state, events=events))
            for e in events:
                if e.kind in ("ScoreIncreased", "LevelIncreased"):
                    n_score += 1
                    score_events_per_level[state.level] += 1
        # Check Ledger promotions when entering a NEW level
        if state.level > max_level:
            max_level = state.level
            try:
                promotions = brain.ledger.promote_at_level(state.level)
                promotions_per_level[state.level] = len(promotions)
            except Exception:
                promotions_per_level[state.level] = 0
        a = brain.act(state, ComputeBudget(actions_remaining=max_steps - step,
                                              time_remaining_ms=10000,
                                              attempts_remaining=1))
        obs = env.step(a)
        prev = state
        last_a = a

    env.close()
    return {
        "attempt": attempt_idx,
        "score_events": n_score,
        "max_level": max_level,
        "score_per_level": dict(score_events_per_level),
        "promotions_per_level": promotions_per_level,
        "ledger_entries_now": len(brain.ledger),
        "hot_calls": brain.n_hot_calls,
        "cold_calls": brain.n_cold_calls,
        "banked_surprises": brain.salience.n_banked,
    }


def main():
    n_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    print(f"Ledger transfer probe on vc33 — N={n_attempts} × {max_steps} steps")
    print(f"  PERSISTENT brain across attempts (intra-game memory accumulates)")
    print(f"  goal: reach maxL ≥ 2 AND Ledger promotions fire on Level 2 entry")
    print()

    # ONE brain, used across all attempts
    brain = BioBrainV2(seed=0)
    brain.reset_game("vc33")
    sal = Salience()  # one salience tracker too

    print(f"{'#':>3s}  {'scored':>6s}  {'maxL':>4s}  "
          f"{'score/level':>16s}  {'promos/level':>14s}  "
          f"{'ledger':>6s}  {'hot':>4s}  {'cold':>4s}")
    print("-" * 90)
    t0 = time.time()
    n_with_l2 = 0
    n_with_l2_promo = 0
    n_with_promo_anywhere = 0
    hot_total_history = []
    for i in range(n_attempts):
        r = run_attempt(brain, sal, attempt_idx=i, max_steps=max_steps)
        hot_total_history.append(r["hot_calls"])
        if r["max_level"] >= 2:
            n_with_l2 += 1
            if r["promotions_per_level"].get(2, 0) > 0:
                n_with_l2_promo += 1
        if any(v > 0 for v in r["promotions_per_level"].values()):
            n_with_promo_anywhere += 1
        score_str = str(r["score_per_level"])
        promo_str = str(r["promotions_per_level"])
        print(f"  {r['attempt']:>2d}  {r['score_events']:>6d}  "
              f"{r['max_level']:>4d}  {score_str:>16s}  "
              f"{promo_str:>14s}  "
              f"{r['ledger_entries_now']:>6d}  "
              f"{r['hot_calls']:>4d}  {r['cold_calls']:>4d}")
    print()
    print(f"Summary:")
    print(f"  Attempts reaching Level 2:               {n_with_l2}/{n_attempts}")
    print(f"  Attempts with Level-2 Ledger promo:      {n_with_l2_promo}/{n_attempts}")
    print(f"  Attempts with ANY level promo fired:     {n_with_promo_anywhere}/{n_attempts}")
    print(f"  Final Ledger entries:                    {len(brain.ledger)}")
    print(f"  Total hot calls (commit-and-monitor):    {sum(hot_total_history)}")
    if n_with_l2 > 0:
        print(f"  ✓ Reached Level 2 on some attempts")
    if n_with_promo_anywhere > 0:
        print(f"  ✓ Cross-level program promotion FIRED on some attempts")
    if n_with_l2 == 0:
        print(f"  ✗ Never reached Level 2")
    # Per-entry diagnostics — what did the Ledger actually learn?
    print()
    print("Ledger entries at end (programs the brain abstracted):")
    for entry in brain.ledger.all_entries()[:10]:
        per_lvl = ", ".join(f"L{l}={a}/{a+b}"
                             for l, (a, b) in entry.per_level.items())
        print(f"  {entry.program_id[:60]:<60s}  origin_L{entry.origin_level}  {per_lvl}")
    print(f"  Wall time: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
