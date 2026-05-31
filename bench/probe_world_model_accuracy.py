"""Probe — World Model fact-prediction accuracy. Phase 1 go/no-go.

Question: when the BayesianWorldModel sees a (state, action), how
accurately does it predict the fact set of the next state?

This determines whether the Simulator is viable. If the world model can't
predict next-state facts well, MCTS / lookahead over it would chase phantoms.

Methodology:
  - Pick games where we have many transitions to learn from (vc33, lp85,
    r11l — substrate-scoring games), plus cd82 (the litmus) and bp35
    (Noise-extractor flagship).
  - Run N=20 attempts × 200 steps with random actions on each game.
  - For each transition after step K=50 (warm-up):
      predicted_facts = {f : P(f) >= 0.5 from WM.predict(before, action)}
      actual_facts    = emit_atomic_facts(before, after)
      true_positives  = predicted ∩ actual
      false_positives = predicted - actual
      false_negatives = actual - predicted
      precision       = TP / (TP + FP)
      recall          = TP / (TP + FN)
      F1              = 2PR / (P + R)
  - Report per-game F1 + precision + recall.

Thresholds:
  F1 >= 0.7 — Simulator viable. Build Phase 2.
  F1 in [0.5, 0.7) — viable for shallow lookahead, deep MCTS risky.
  F1 < 0.5 — WM needs richer predictions before Simulator is built.

Wall budget: ~5 min.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.disable(logging.CRITICAL)

from arena.env import ArenaEnv  # noqa: E402
from arena.perceive import perceive  # noqa: E402
from arena.salience import Salience  # noqa: E402
from arena.types import Transition  # noqa: E402
from prism.bayes_world_model import BayesianWorldModel  # noqa: E402
from prism.empowerment_brain import _candidate_actions  # noqa: E402
from prism.predicate_pool import emit_atomic_facts  # noqa: E402


GAMES = ("vc33", "lp85", "r11l", "cd82", "bp35", "g50t")
N_ATTEMPTS = 20
MAX_STEPS = 200
WARMUP_STEPS = 50  # let the WM learn before scoring its predictions


def measure(game: str) -> dict:
    env = ArenaEnv(game, mode="OFFLINE")
    wm = BayesianWorldModel()
    sal = Salience()
    rng = random.Random(0)
    tp = fp = fn = 0
    n_scored = 0

    for attempt in range(N_ATTEMPTS):
        obs = env.reset()
        prev_state = None
        last_action = None
        for step in range(MAX_STEPS):
            if env.is_terminal(obs):
                break
            parsed = env.parse(obs)
            if parsed["grid"] is None:
                break
            avail = tuple(int(a) for a in parsed.get("available_actions", ()) or ())
            sal.observe(parsed["grid"])
            state = perceive(parsed["grid"], prev_state, score=parsed["score"],
                             level=parsed["levels_completed"],
                             available_actions=avail, salience_mask=sal.mask())
            # Score WM prediction (only after warmup, and only if we have
            # both a before-state and an action to predict from).
            if prev_state is not None and last_action is not None:
                if step >= WARMUP_STEPS:
                    predicted = wm.predict(prev_state, last_action)
                    predicted_set = {f for f, p in predicted.items() if p >= 0.5}
                    actual_set = emit_atomic_facts(prev_state, state)
                    tp += len(predicted_set & actual_set)
                    fp += len(predicted_set - actual_set)
                    fn += len(actual_set - predicted_set)
                    n_scored += 1
                # Train AFTER scoring (so we test on next-unseen)
                wm.observe(prev_state, last_action, state)
            cands = _candidate_actions(state)
            if not cands:
                break
            action = rng.choice(cands)
            obs = env.step(action)
            prev_state = state
            last_action = action

    env.close()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n_scored": n_scored, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> int:
    print(f"Probe: BayesianWorldModel prediction accuracy")
    print(f"  N={N_ATTEMPTS} × {MAX_STEPS} steps × random actions")
    print(f"  warm-up: {WARMUP_STEPS} steps before scoring")
    print(f"  scoring: TP/FP/FN over predicted-vs-actual fact sets per transition")
    print()
    print(f"{'game':>6s}  {'n_xitions':>10s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  "
          f"{'P':>5s}  {'R':>5s}  {'F1':>5s}  verdict")
    print("-" * 90)
    t0 = time.time()
    for game in GAMES:
        try:
            r = measure(game)
        except Exception as e:
            print(f"  {game:>6s}  error: {e}")
            continue
        f1 = r['f1']
        if f1 >= 0.7:
            verdict = "✓ simulator viable"
        elif f1 >= 0.5:
            verdict = "⚠ shallow only"
        else:
            verdict = "✗ WM too noisy"
        print(f"  {game:>6s}  {r['n_scored']:>10d}  {r['tp']:>6d}  {r['fp']:>6d}  "
              f"{r['fn']:>6d}  {r['precision']:>5.2f}  {r['recall']:>5.2f}  "
              f"{f1:>5.2f}  {verdict}")

    print()
    print(f"Wall time: {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
