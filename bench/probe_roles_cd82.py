# bench/probe_roles_cd82.py
"""Probe — does cd82's dark-selector entity get tagged correctly?

Per spec §6.1 validation table: "After 20 observations on cd82,
dark-selector tags as PAINTER/SELECTOR; target tags as TARGET; framing
as STATIC."

This is the first per-component diagnostic gate. If roles don't tag
correctly, the rest of the within-game search pipeline cannot work.

Usage:
    BIOBRAIN_ENV_DIR=... python bench/probe_roles_cd82.py
"""
import logging
import sys

logging.disable(logging.CRITICAL)

from biobrain.perception.perceive import detect_events, perceive
from biobrain.perception.salience import Salience
from biobrain import BioBrainV2
from biobrain.adapters.arc import ArenaEnv
from biobrain.types import ComputeBudget, Transition, action_click, action_key
from biobrain.salience.roles import Role


# Approximate cd82 selector positions (from earlier inspection)
SELECTOR_WHITE = (43, 4)  # (col, row) center of white selector
SELECTOR_DARK = (37, 4)   # center of dark selector


def main():
    print("=" * 60)
    print("Probe: cd82 role assignment after deliberate clicks")
    print("=" * 60)
    print()

    env = ArenaEnv("cd82", mode="OFFLINE")
    brain = BioBrainV2(seed=0)
    brain.reset_game("cd82")
    sal = Salience()
    brain.reset_attempt()

    obs = env.reset()
    prev = None
    last_a = None

    # Pre-defined click sequence: alternate WHITE and DARK selectors
    # to give the brain explicit observations of each entity's causal
    # signature.
    click_sequence = [
        action_click(*SELECTOR_DARK),
        action_click(*SELECTOR_WHITE),
        action_click(*SELECTOR_DARK),
        action_click(*SELECTOR_WHITE),
        action_click(*SELECTOR_DARK),
        action_click(*SELECTOR_WHITE),
        action_key(0),  # cursor key
        action_key(1),
        action_click(*SELECTOR_DARK),
        action_click(*SELECTOR_WHITE),
    ]

    for step, action in enumerate(click_sequence):
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
        if prev is not None and last_a is not None:
            events = detect_events(prev, state)
            brain.observe(Transition(before=prev, action=last_a,
                                      after=state, events=events))
        obs = env.step(action)
        prev = state
        last_a = action

    env.close()

    # Report
    print(f"Observations recorded: {len(brain.salience._role_counters)} entities")
    print(f"Role assignments:")
    counter_summary = []
    for eid, sig in brain.salience._role_counters.items():
        role = brain.salience._role_assignments.get(eid, Role.UNKNOWN)
        counter_summary.append((eid, role, sig))

    counter_summary.sort(key=lambda x: x[2].n_observations, reverse=True)
    for eid, role, sig in counter_summary[:10]:
        print(f"  entity {eid}: role={role.value:>10s}  "
              f"n={sig.n_observations:>3d}  "
              f"clicked={sig.clicked_on_count:>2d}  "
              f"self_change={sig.clicked_caused_self_change:>2d}  "
              f"other_change={sig.clicked_caused_other_change:>2d}  "
              f"persistence={sig.persistence:.2f}")

    print()
    print(f"Final SearchGraph: {len(brain.planner.search_graph)} nodes")
    print(f"Final fingerprint index: {len(brain.salience.fingerprint_index)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
