"""biobrain.ledger — working memory / scientific method.

Brain region: Hippocampus / dorsolateral prefrontal cortex.
ML/RL term: Episodic memory + cross-task transfer.

The Ledger persists ACROSS LEVELS within a game, wipes BETWEEN games.

Mechanism:
  1. Maintain a rolling buffer of (action, before_state) for the last K
     transitions.
  2. On score event: parameterize the buffer into an entity-anchored DSL
     program (replace literal click(x, y) with click_on_color(entity-at-cell)).
  3. Store the program in a Ledger entry with hierarchical Beta per
     (program_id, level).
  4. On entry to a new level, promote_at_level() surfaces programs with
     high confidence from PRIOR levels — these become high-prior
     candidates the action policy can try first.

Parameterization depth: entity-color anchors (Q2 decision). We don't
anchor to grid geometry — that would hallucinate level-specific layout
as part of the rule. Color is the most robust handle.

Hierarchical Beta (Q3 decision): per-(program, level) Beta. Single-level
scorers don't dominate; cross-level scorers earn confidence.

Wipe boundary (Q4 decision): full Ledger wipe on reset_game. Across
levels within a game it persists, since that's where the scientific
method does its hypothesis-testing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from biobrain.types import (
    Action, State, Transition, action_kind,
    EVENT_LEVEL_INCREASED, EVENT_SCORE_INCREASED,
)
from biobrain.motor_cortex.core import (
    Program, SEQ, click_on_color, key, noop,
)


# Default rolling buffer depth — number of recent actions abstracted into
# a single Ledger entry on score event. K=5 covers most observed
# multi-step interactions in ARC-AGI-3 without explosive program length.
DEFAULT_ABSTRACTION_DEPTH = 5

# Threshold above which a program from a prior level is "promoted" as a
# high-confidence candidate for a new level. 0.7 = 70% posterior mean ≈
# "scored at least 2/3 times when tried." Conservative.
DEFAULT_PROMOTION_THRESHOLD = 0.7


@dataclass
class LedgerEntry:
    """One abstracted successful trajectory.

    program_id        — canonical string id (allows equality / lookup)
    program           — the DSL Program (executable via brain library API)
    per_level         — dict[level → (alpha, beta)] hierarchical Beta
    origin_level      — level where the trajectory was first observed scoring
    origin_action_seq — the raw parameterized signatures (for diagnostics)
    """
    program_id: str
    program: Program
    per_level: dict[int, tuple[float, float]] = field(default_factory=dict)
    origin_level: int = 0
    origin_action_seq: tuple = ()

    def confidence_at(self, level: int) -> float:
        """Beta-posterior mean of scoring at `level`. 0.5 if uninformed."""
        alpha, beta = self.per_level.get(level, (0.0, 0.0))
        return (alpha + 1) / (alpha + beta + 2)

    def max_other_level_confidence(self, current_level: int) -> float:
        """Best per-level confidence excluding current_level. 0.0 if none."""
        best = 0.0
        for lvl, (a, b) in self.per_level.items():
            if lvl == current_level:
                continue
            if a + b < 1:
                continue
            conf = (a + 1) / (a + b + 2)
            if conf > best:
                best = conf
        return best


class Ledger:
    """Per-game persistent memory of successful action sequences."""

    def __init__(self,
                 abstraction_depth: int = DEFAULT_ABSTRACTION_DEPTH,
                 promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD,
                 ) -> None:
        self._abstraction_depth = abstraction_depth
        self._promotion_threshold = promotion_threshold
        self._entries: dict[str, LedgerEntry] = {}
        self._recent: deque[tuple[Action, State]] = deque(
            maxlen=abstraction_depth)

    def reset_game(self) -> None:
        self._entries = {}
        self._recent = deque(maxlen=self._abstraction_depth)

    # ------------------------------------------------------------------ public

    def observe(self, transition: Transition) -> Optional[LedgerEntry]:
        """Track action history; on score event, abstract into a program.

        Returns the LedgerEntry that was created/updated, or None.

        IMPORTANT: when a score event fires, the recorded score_level is
        the BEFORE level (the level the program was running at when it
        scored), NOT the AFTER level (the new level the brain just
        entered). This is critical for cross-level promotion: a program
        that "scored at level 0" should be promoted as a candidate for
        levels 1, 2, etc. If we instead recorded score_level=1 (the
        after level), the entry would be at the same level as the new
        attempt and never get promoted as cross-level transfer.
        """
        if transition.before is not None and transition.action is not None:
            self._recent.append((transition.action, transition.before))
        scored = False
        # Use the BEFORE level — the level the program ran at when it scored.
        # Fall back to after if no before (initial-state edge case).
        if transition.before is not None:
            score_level = transition.before.level
        else:
            score_level = (transition.after.level
                            if transition.after is not None else 0)
        for e in transition.events:
            if e.kind in (EVENT_SCORE_INCREASED, EVENT_LEVEL_INCREASED):
                scored = True
                break
        if scored:
            return self._on_score_event(score_level)
        return None

    def confidence(self, program_id: str, level: int) -> float:
        entry = self._entries.get(program_id)
        if entry is None:
            return 0.5
        return entry.confidence_at(level)

    def promote_at_level(self, level: int) -> list[tuple[Program, float, str]]:
        """Return (program, confidence, program_id) tuples for programs
        with ANY positive evidence at a prior level.

        Sorted by confidence descending. No magic threshold — the brain
        consumes this list and Thompson-samples among candidates,
        which naturally weighs by confidence. The previous hard
        threshold (0.7) excluded single-success programs (Beta(1,0) →
        conf=0.67) which is exactly the case the scientific-method
        loop is supposed to surface.
        """
        out: list[tuple[Program, float, str]] = []
        for entry in self._entries.values():
            conf = entry.max_other_level_confidence(level)
            # Any positive evidence (conf > 0.5) means "tried at prior
            # level, at least one positive outcome." Promote everything
            # that has SOME signal; let Thompson at the consumer decide.
            if conf > 0.5:
                out.append((entry.program, conf, entry.program_id))
        out.sort(key=lambda x: -x[1])
        return out

    def register_failure(self, program_id: str, level: int) -> None:
        """Note that this program was tried at this level and didn't score."""
        entry = self._entries.get(program_id)
        if entry is None:
            return
        alpha, beta = entry.per_level.get(level, (0.0, 0.0))
        entry.per_level[level] = (alpha, beta + 1)

    def __len__(self) -> int:
        return len(self._entries)

    def all_entries(self) -> list[LedgerEntry]:
        return list(self._entries.values())

    # ------------------------------------------------------------------ private

    def _on_score_event(self, score_level: int) -> Optional[LedgerEntry]:
        if not self._recent:
            return None
        # Parameterize each recent action against its before-state
        sigs: list[tuple] = []
        for action, before_state in list(self._recent):
            sig = self._parameterize_action(action, before_state)
            if sig is None:
                continue
            sigs.append(sig)
        if not sigs:
            return None
        program = self._build_program(sigs)
        program_id = self._build_program_id(sigs)
        entry = self._entries.get(program_id)
        if entry is None:
            entry = LedgerEntry(
                program_id=program_id, program=program,
                origin_level=score_level,
                origin_action_seq=tuple(sigs),
            )
            self._entries[program_id] = entry
        # Record this score at this level
        alpha, beta = entry.per_level.get(score_level, (0.0, 0.0))
        entry.per_level[score_level] = (alpha + 1, beta)
        return entry

    def _parameterize_action(self, action: Action,
                              state: State) -> Optional[tuple]:
        """Replace literal positions with entity-color anchors.

        click(x, y) → ('click_on_color', c) where c is the color of the
        entity at cell (y, x). Returns None if the click landed outside
        any entity (we can't anchor a click on empty space without
        introducing a geometric prior).

        key(k) → ('key', k) — keys are already parameterized.
        spacebar → ('spacebar',)
        """
        if action_kind(action) == "click" and len(action) >= 3:
            x, y = int(action[1]), int(action[2])
            for e in state.entities:
                if (y, x) in e.region.cells:
                    return ("click_on_color", int(e.color))
            return None
        if action_kind(action) == "key" and len(action) >= 2:
            return ("key", int(action[1]))
        if action_kind(action) == "spacebar":
            return ("spacebar",)
        return None

    def _build_program(self, sigs: list) -> Program:
        progs: list[Program] = []
        for sig in sigs:
            if sig[0] == "click_on_color":
                progs.append(click_on_color(int(sig[1])))
            elif sig[0] == "key":
                progs.append(key(int(sig[1])))
            elif sig[0] == "spacebar":
                from biobrain.motor_cortex.core import spacebar
                progs.append(spacebar())
        if not progs:
            return noop()
        result = progs[0]
        for p in progs[1:]:
            result = SEQ(result, p)
        return result

    def _build_program_id(self, sigs: list) -> str:
        parts: list[str] = []
        for s in sigs:
            if len(s) > 1:
                parts.append(f"{s[0]}({s[1]})")
            else:
                parts.append(s[0])
        return " → ".join(parts)


__all__ = ["Ledger", "LedgerEntry",
           "DEFAULT_ABSTRACTION_DEPTH", "DEFAULT_PROMOTION_THRESHOLD"]
