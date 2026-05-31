"""biobrain.env.brain_interface — the BrainEngine protocol.

The seam between the arena (perception + env + measurement) and any
brain engine. PRISM is one implementation; future engines plug in
through this same interface.

The protocol is small by design (4 methods). Every type the
interface mentions is from `biobrain.types`. No brain implementation
may extend the protocol with additional public methods — internal
helpers are fine; cross-talk through globals is not.

Cross-references:
- Technical design §2 (the BrainEngine interface)
- Technical design §14 (why the split matters)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from biobrain.types import Action, ComputeBudget, State, Transition


@runtime_checkable
class BrainEngine(Protocol):
    """Interface any brain engine must implement.

    The arena handles game mechanics, perception, environment glue,
    and RHAE measurement. The brain is pluggable. PRISM (in `prism/`)
    is the default implementation; baselines and future engines plug
    in here.

    The protocol is enforced at runtime via `@runtime_checkable`; a
    concrete brain class need not subclass BrainEngine explicitly,
    but it MUST provide all four methods with compatible signatures.

    --- OOD covenant ---
    `reset_game(game_id)` MUST clear all within-game state. The arena
    calls this at every game boundary. Implementations that retain
    state across games violate the benchmark's OOD evaluation
    discipline.
    """

    def reset_game(self, game_id: str) -> None:
        """Start of a new game. Clear ALL within-game state.

        Implementations MUST clear:
          - any posterior / hypothesis state
          - any learned schemas
          - any observed-transition history
          - any cached rollouts or plans

        The arena calls this at every game boundary during evaluation.
        """
        ...

    def reset_attempt(self) -> None:
        """Start of a new attempt within a game.

        Within-game state IS preserved (posteriors, schemas, library
        learned this game). Implementations may clear per-attempt
        scratch (e.g., rollout caches, intent commitments).
        """
        ...

    def observe(self, transition: Transition) -> None:
        """A transition was observed.

        The brain updates internal posteriors / schemas as a function
        of `transition`. The arena calls this AFTER each environment
        step, before the next `act` call.
        """
        ...

    def act(self, state: State, budget: ComputeBudget) -> Action:
        """Return an action to take given current state and budget.

        MUST return an `Action` constructed via `biobrain.types.action_*`
        helpers or matching their tuple shape. Side-effect-free with
        respect to environment state.
        """
        ...


# ---------------------------------------------------------------------------
# NullBrainEngine — a stub brain for testing the arena loop in isolation
# ---------------------------------------------------------------------------

class NullBrainEngine:
    """A trivial brain that always presses key 0.

    Implements the BrainEngine protocol without any actual reasoning.
    Useful for:
      - Testing the arena loop without depending on PRISM.
      - Sanity-checking the BrainEngine protocol's type compatibility.
      - Establishing a "random / trivial" baseline RHAE.

    Verifies via Stage 1 gate that the protocol is satisfiable by a
    concrete implementation that does not import `prism/`.
    """

    def __init__(self) -> None:
        self._game_id: str | None = None
        self._observed_count: int = 0

    def reset_game(self, game_id: str) -> None:
        self._game_id = game_id
        self._observed_count = 0

    def reset_attempt(self) -> None:
        # NullBrain has no per-attempt state.
        pass

    def observe(self, transition: Transition) -> None:
        self._observed_count += 1

    def act(self, state: State, budget: ComputeBudget) -> Action:
        # Always press key 0. Trivial baseline.
        from biobrain.types import action_key
        return action_key(0)
