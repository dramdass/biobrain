"""Probe — which L3 extractors fire on which games?

Coverage analysis for the pluggable L3 architecture. For each public
game, run 50 random-action steps to populate TransitionHistory, then
ask both extractors what proto-goals they detect.

Output per game:
  StaticPatternRecurrence: N goals (top weight, top description)
  ChangeDynamics:           N goals (top weight, top description)

Identifies:
  - Games matched by both extractors (architecture works)
  - Games matched only by one (specialist coverage)
  - Games matched by neither (extractor gap — new extractor type needed)

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
from arena.probes.fixtures import PUBLIC_GAMES  # noqa: E402
from arena.salience import Salience  # noqa: E402
from arena.types import Transition  # noqa: E402
from prism.empowerment_brain import _candidate_actions  # noqa: E402
from prism.goals import (  # noqa: E402
    L3, TransitionHistory,
    StaticPatternRecurrence, ChangeDynamics,
    Compression, Noise, Symmetry,
)
from prism.goals.change_dynamics_facts import ChangeDynamicsFactSpace  # noqa: E402


N_STEPS = 50


def random_steps(game, n_steps):
    """Run n_steps of random actions and return (final_state, history)."""
    env = ArenaEnv(game, mode="OFFLINE")
    hist = TransitionHistory()
    sal = Salience()
    obs = env.reset()
    prev = None
    rng = random.Random(0)
    state = None
    for _ in range(n_steps):
        if env.is_terminal(obs): break
        parsed = env.parse(obs)
        if parsed["grid"] is None: break
        avail = tuple(int(a) for a in parsed.get("available_actions", ()) or ())
        sal.observe(parsed["grid"])
        state = perceive(parsed["grid"], prev, score=parsed["score"],
                         level=parsed["levels_completed"], available_actions=avail,
                         salience_mask=sal.mask())
        if prev is not None:
            hist.update(Transition(before=prev, action=last_a,
                                   after=state, events=[]))
        cands = _candidate_actions(state)
        if not cands: break
        last_a = rng.choice(cands)
        obs = env.step(last_a)
        prev = state
    env.close()
    return state, hist


def main() -> int:
    print(f"Probe: L3 extractor coverage on {len(PUBLIC_GAMES)} games "
          f"(N={N_STEPS} random steps each)")
    print()
    print(f"{'game':>6s}  {'SPR':>4s}  {'CD':>4s}  {'CDf':>4s}  {'Cmp':>4s}  {'Noi':>4s}  {'Sym':>4s}  {'top goal':<50s}")
    print("-" * 95)
    t0 = time.time()
    spr = StaticPatternRecurrence()
    cd = ChangeDynamics()
    cdf = ChangeDynamicsFactSpace()
    cmp_ = Compression()
    noi = Noise()
    sym = Symmetry()
    matched, neither_set = [], []
    for game in PUBLIC_GAMES:
        try:
            state, hist = random_steps(game, N_STEPS)
            if state is None:
                print(f"  {game:>6s}  (no state)")
                continue
            spr_goals = spr.detect(state, hist)
            cd_goals = cd.detect(state, hist)
            cdf_goals = cdf.detect(state, hist)
            cmp_goals = cmp_.detect(state, hist)
            noi_goals = noi.detect(state, hist)
            sym_goals = sym.detect(state, hist)
            counts = [len(spr_goals), len(cd_goals), len(cdf_goals),
                      len(cmp_goals), len(noi_goals), len(sym_goals)]
            all_g = (spr_goals + cd_goals + cdf_goals + cmp_goals
                     + noi_goals + sym_goals)
            top_goal = ""
            if all_g:
                all_g.sort(key=lambda g: -g.weight)
                top_goal = f"[{all_g[0].source[:4]}] " + all_g[0].description[:46]
            print(f"  {game:>6s}  {counts[0]:>4d}  {counts[1]:>4d}  "
                  f"{counts[2]:>4d}  {counts[3]:>4d}  {counts[4]:>4d}  "
                  f"{counts[5]:>4d}  {top_goal:<50s}")
            if any(c > 0 for c in counts):
                matched.append(game)
            else:
                neither_set.append(game)
        except Exception as e:
            print(f"  {game:>6s}  error: {e}")
            continue
    print()
    print(f"Coverage summary:")
    print(f"  matched:  {len(matched):>2d}/25  {matched}")
    print(f"  neither:  {len(neither_set):>2d}/25  {neither_set}")
    print()
    print(f"Wall time: {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
