# bench/probe_within_game_search_e2e.py
"""Probe — end-to-end within-game search measurement.

Runs BioBrainV2 v0.3 against a single game across multiple attempts
with a PERSISTENT brain (intra-game memory accumulates), and tracks
scoring rate, max-level, search-graph size, fingerprint index size,
and role-assignment count per attempt.

Usage:
    BIOBRAIN_ENV_DIR=... python bench/probe_within_game_search_e2e.py [game] [n_attempts] [max_steps]

Default: vc33, 20 attempts, 300 steps each.
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
from biobrain.salience.roles import Role


def run_attempt(brain, sal, game, max_steps):
    env = ArenaEnv(game, mode="OFFLINE")
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
        a = brain.act(state, ComputeBudget(max_steps - step, 10000, 1))
        obs = env.step(a)
        prev = state
        last_a = a
    env.close()
    return {"score_events": n_score, "max_level": max_level}


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "vc33"
    n_attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    print(f"=" * 60)
    print(f"Probe: within-game search e2e on {game}")
    print(f"N={n_attempts} × {max_steps} steps, persistent brain")
    print(f"=" * 60)
    print()

    brain = BioBrainV2(seed=0)
    brain.reset_game(game)
    sal = Salience()

    print(f"{'attempt':>7s}  {'scored':>6s}  {'maxL':>4s}  "
          f"{'graph':>6s}  {'index':>6s}  {'roles':>6s}")
    print("-" * 60)
    t0 = time.time()
    total_scored = 0
    overall_max_level = 0
    for i in range(n_attempts):
        r = run_attempt(brain, sal, game, max_steps)
        if r["score_events"] > 0:
            total_scored += 1
        overall_max_level = max(overall_max_level, r["max_level"])
        n_roles_assigned = sum(
            1 for role in brain.salience._role_assignments.values()
            if role != Role.UNKNOWN)
        print(f"  {i:>5d}  {r['score_events']:>6d}  {r['max_level']:>4d}  "
              f"{len(brain.planner.search_graph):>6d}  "
              f"{len(brain.salience.fingerprint_index):>6d}  "
              f"{n_roles_assigned:>6d}")

    print()
    print(f"Summary:")
    print(f"  Attempts scored:        {total_scored}/{n_attempts}")
    print(f"  Max level reached:      {overall_max_level}")
    print(f"  Final graph nodes:      {len(brain.planner.search_graph)}")
    print(f"  Final index entries:    {len(brain.salience.fingerprint_index)}")
    print(f"  Scoring-tagged nodes:   {len(brain.planner.search_graph.scoring_nodes())}")
    print(f"  Wall time:              {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
