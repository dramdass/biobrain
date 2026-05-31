"""MemoryBrainDream — Thompson sampling over a library of Programs.

Architecture:
  - Library of compositional Programs (DSL.core).
  - Per-program Beta(α, β) posterior of P(score | use this program).
  - Substrate (ActionScoreTable + Thompson) always available as fallback.
  - At each step: sample one program from library by per-program Thompson;
    if it returns a non-noop action, use it; else fall through to substrate.
  - When program is mid-sequence (continuation != None), keep using it until
    it returns None.

The library is SEEDED with parameterized templates from Spelke primitives.
For cd82-class validation: we want to see whether ANY compositional template
in the seed library produces a score event on a game where atomic primitives
have failed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from biobrain.types import (
    Action, ComputeBudget, State, Transition, action_kind, action_key,
    action_undo, EVENT_LEVEL_INCREASED, EVENT_SCORE_INCREASED,
)
from biobrain.planner.agency import _candidate_actions
from biobrain.planner.representation import RepLoop
from biobrain.planner.posterior import ActionScoreTable
from biobrain.motor_cortex.core import (
    Program, IF, SEQ, WHILE_NOT, REPEAT,
    click_on_color, key, spacebar, noop,
    has_color, always, never,
)


def _resolve_sig_to_action(sig: tuple, state, candidates: list) -> Optional[Action]:
    """Convert a DSL action signature to a concrete Action that's available.

    Returns None if no matching available action (caller falls back to substrate).
    """
    kind = sig[0]
    if kind == "click_on_color":
        c = sig[1]
        # Find a click candidate landing on an entity of this color.
        for a in candidates:
            if action_kind(a) == "click" and len(a) >= 3:
                x, y = int(a[1]), int(a[2])
                for e in state.entities:
                    if int(e.color) == c and (y, x) in e.region.cells:
                        return a
        return None
    if kind == "key":
        k = sig[1]
        for a in candidates:
            if action_kind(a) == "key" and len(a) >= 2 and int(a[1]) == k:
                return a
        return None
    if kind == "spacebar":
        # Try key=4 first (often spacebar's sub-id), then any unused key
        for sub_id in (4, 3, 2, 1, 0):
            for a in candidates:
                if action_kind(a) == "key" and len(a) >= 2 and int(a[1]) == sub_id:
                    return a
        return None
    if kind == "undo":
        for a in candidates:
            if action_kind(a) == "undo":
                return a
        return None
    return None


def seed_library(observed_colors: list, observed_keys: list) -> list[tuple[str, Program]]:
    """Seed the library with parameterized templates from Spelke primitives.

    Each entry is (id, Program). Templates are bounded depth, game-agnostic;
    parameters bound from observed values.
    """
    lib = []

    # 1. Atomic clicks per observed color
    for c in observed_colors:
        lib.append((f"click({c})", click_on_color(c)))

    # 2. Atomic keys per observed key
    for k in observed_keys:
        lib.append((f"key({k})", key(k)))

    # 3. Spacebar atom
    lib.append(("spacebar", spacebar()))

    # 4. SEQ(click, spacebar) — click-then-paint per color
    for c in observed_colors:
        lib.append((f"click({c}) then spacebar",
                    SEQ(click_on_color(c), spacebar())))

    # 5. SEQ(key, spacebar) — navigate-then-paint per key
    for k in observed_keys:
        lib.append((f"key({k}) then spacebar",
                    SEQ(key(k), spacebar())))

    # 6. REPEAT(2, key) then spacebar — multi-step nav
    for k in observed_keys:
        lib.append((f"key({k})×2 then spacebar",
                    SEQ(REPEAT(2, key(k)), spacebar())))

    # 7. SEQ(click(c1), key, spacebar) — modal switch + navigate + paint
    for c in observed_colors[:3]:  # bound combinatorial explosion
        for k in observed_keys[:2]:
            lib.append((f"click({c}) → key({k}) → spacebar",
                        SEQ(SEQ(click_on_color(c), key(k)), spacebar())))

    return lib


class MemoryBrainDream:
    """Library-of-programs brain with Thompson over per-program posterior.

    At decision time:
      1. If a multi-step program is in progress (continuation), step it.
      2. Else: Thompson-sample a program from library; execute one step.
      3. Resolve signature to concrete action; if no match, fall back to
         substrate's Thompson over ActionScoreTable.
    """

    def __init__(self, *, seed: int = 0,
                 observed_colors: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15),
                 observed_keys: tuple = (0, 1, 2, 3, 4)) -> None:
        self._seed = seed
        self._rng: random.Random = random.Random(seed)
        self._rep: RepLoop = RepLoop()
        self._action_table: ActionScoreTable = ActionScoreTable()
        # Library = list of (id, Program)
        self._library = seed_library(list(observed_colors), list(observed_keys))
        # Beta posterior per library id: (α, β)
        self._posterior: dict[str, tuple[float, float]] = {
            pid: (0, 0) for pid, _ in self._library
        }
        # Current in-flight program continuation
        self._current_program: Optional[Program] = None
        self._current_program_id: Optional[str] = None
        self._observed_count: int = 0

    def reset_game(self, game_id: str) -> None:
        self._rep = RepLoop()
        self._action_table = ActionScoreTable()
        # Library stays
        self._posterior = {pid: (0, 0) for pid, _ in self._library}
        self._current_program = None
        self._current_program_id = None
        self._observed_count = 0
        self._rng = random.Random(self._seed)

    def reset_attempt(self) -> None:
        self._current_program = None
        self._current_program_id = None

    def observe(self, transition: Transition) -> None:
        self._observed_count += 1
        self._rep.update(transition)
        # Substrate update
        self._action_table.observe(transition)
        # Score events update the program that was running (if any)
        scored = False
        for e in transition.events:
            if e.kind in (EVENT_SCORE_INCREASED, EVENT_LEVEL_INCREASED):
                scored = True
                break
        if self._current_program_id is not None:
            α, β = self._posterior.get(self._current_program_id, (0, 0))
            if scored:
                self._posterior[self._current_program_id] = (α + 1, β)
            else:
                # Small negative update — many non-scoring steps under a
                # program shouldn't drown out a few scoring ones, but the
                # program should drift if it never scores
                self._posterior[self._current_program_id] = (α, β + 0.1)

    def act(self, state: State, budget: ComputeBudget) -> Action:
        candidates = _candidate_actions(state)
        if not candidates:
            raise ValueError("no candidates")
        # 1. Continue a multi-step program in flight?
        if self._current_program is not None:
            sig, next_program = self._current_program.step(state)
            self._current_program = next_program
            action = _resolve_sig_to_action(sig, state, candidates)
            if action is not None:
                return action
            # Couldn't resolve — abandon program, fall through
            self._current_program = None
            self._current_program_id = None

        # 2. Thompson-sample a program from library + a fallback "substrate"
        best_score = -1.0
        best_id = None
        best_program = None
        for pid, prog in self._library:
            α, β = self._posterior.get(pid, (0, 0))
            v = self._rng.betavariate(α + 1, β + 1)
            if v > best_score:
                best_score = v
                best_id = pid
                best_program = prog

        # Also sample the substrate's best action — if it beats any program,
        # we use it (the substrate is the "always available" fallback)
        substrate_score = -1.0
        substrate_action = None
        for a in candidates:
            sig = ActionScoreTable._signature(a, state)
            n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
            v = self._rng.betavariate(n_goal + 1, n_obs - n_goal + 1)
            if v > substrate_score:
                substrate_score = v
                substrate_action = a

        # Choose between best program and substrate's argmax
        if best_score > substrate_score and best_program is not None:
            sig, next_program = best_program.step(state)
            action = _resolve_sig_to_action(sig, state, candidates)
            if action is not None:
                self._current_program = next_program
                self._current_program_id = best_id
                return action

        return substrate_action if substrate_action is not None else self._rng.choice(candidates)

    def end_of_attempt(self) -> None:
        pass

    def n_visited_states(self) -> int:
        return self._rep.n_distinct_derived_states()
