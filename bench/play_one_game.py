"""Play one game with biobrain in ONLINE mode — produces a watchable
scorecard URL on arcprize.org.

Usage:
    BIOBRAIN_ENV_DIR=... python bench/play_one_game.py [game] [max_steps]

Default: vc33, 200 steps.
"""

import logging
import sys

logging.disable(logging.WARNING)  # keep INFO so we see the scorecard URL

from biobrain.perception.perceive import detect_events, perceive
from biobrain.perception.salience import Salience
from biobrain import BioBrainV2
from biobrain.adapters.arc import ArenaEnv
from biobrain.types import ComputeBudget, Transition


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "vc33"
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"=" * 60)
    print(f"biobrain v0.2 playing {game}")
    print(f"max_steps={max_steps}, n_attempts={n_attempts}, mode=ONLINE")
    print(f"=" * 60)

    env = ArenaEnv(game, mode="ONLINE")
    scorecard_id = getattr(env._env, "scorecard_id", None)
    if scorecard_id:
        url = f"https://arcprize.org/scorecards/{scorecard_id}"
        print(f"\n🎮 WATCH LIVE: {url}\n")

    brain = BioBrainV2(seed=0)
    brain.reset_game(game)
    sal = Salience()

    total_score = 0
    max_level = 0

    for attempt_i in range(n_attempts):
        brain.reset_attempt()
        obs = env.reset()
        prev = None
        last_a = None
        attempt_score = 0
        attempt_max_level = 0
        terminal_reason = "step_limit"

        for step in range(max_steps):
            if env.is_terminal(obs):
                state_name = getattr(getattr(obs, "state", None), "name", "?")
                terminal_reason = f"terminal({state_name})"
                break
            parsed = env.parse(obs)
            if parsed["grid"] is None:
                terminal_reason = "no_grid"
                break
            avail = tuple(int(a) for a in parsed.get("available_actions") or ())
            sal.observe(parsed["grid"])
            state = perceive(parsed["grid"], prev,
                             score=parsed["score"],
                             level=parsed["levels_completed"],
                             available_actions=avail,
                             salience_mask=sal.mask())
            attempt_max_level = max(attempt_max_level, state.level)
            if prev is not None and last_a is not None:
                events = detect_events(prev, state)
                brain.observe(Transition(before=prev, action=last_a,
                                          after=state, events=events))
                for e in events:
                    if e.kind in ("ScoreIncreased", "LevelIncreased"):
                        attempt_score += 1
                        print(f"    [step {step:>3d}] 🎯 SCORE at level {state.level}")
            a = brain.act(state, ComputeBudget(max_steps - step, 10000, 1))
            obs = env.step(a)
            prev = state
            last_a = a

        total_score += attempt_score
        max_level = max(max_level, attempt_max_level)
        print(f"  attempt {attempt_i + 1}/{n_attempts}: "
              f"score={attempt_score} maxL={attempt_max_level} "
              f"steps={step + 1} terminal={terminal_reason}")

    env.close()

    print()
    print(f"=" * 60)
    print(f"FINAL: {total_score} score events, max level reached: {max_level}")
    print(f"  hot calls:    {brain.n_hot_calls}")
    print(f"  cold calls:   {brain.n_cold_calls}")
    print(f"  ledger:       {len(brain.ledger)} programs banked")
    print(f"  banked surp:  {brain.salience.n_banked}")
    if scorecard_id:
        print(f"\n🎮 WATCH REPLAY: https://arcprize.org/scorecards/{scorecard_id}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
