"""biobrain.brain — the composer.

Brings the named cortical regions together into a single agent interface.
A BioBrain has the same public interface as our other brains (observe,
act, reset_game, reset_attempt) so it slots into existing probes.

Lifecycle (House model):
  reset_game(game_id) — full inter-game amnesia
    Wipes: substrate posterior, world model, transition history, ledger.
    Keeps: static perception layer (Spelke priors live outside the brain).
  reset_attempt() — between attempts within a game
    Substrate + world model + ledger all persist across attempts (since
    they encode game-specific physics that survives attempt resets).
"""

from __future__ import annotations

from biobrain.types import Action, ComputeBudget, State, Transition
from biobrain.critic import Critic
from biobrain.curiosity import Curiosity
from biobrain.curiosity.predicates import emit_atomic_facts
from biobrain.latent_inference import LatentInference
from biobrain.ledger.ledger import Ledger
from biobrain.planner.planner_facade import Planner
from biobrain.simulator.simulator import Simulator


class BioBrain:
    """Active-inference agent composed of seven named cortical regions.

    Public interface (matches BrainEngine):
        reset_game(game_id)
        reset_attempt()
        observe(transition)
        act(state, budget) → Action
        end_of_attempt()

    Internal components (exposed for diagnostics):
        critic            — multi-extractor goal-setting (compression motif)
        curiosity         — ICM-style intrinsic reward via Bayesian WM
        ledger            — episodic memory of successful trajectories
        simulator         — forward-dynamics queries via WM
        planner           — combines all signals into action selection
        latent_inference  — hypothesizes hidden state variables when the
                            World Model has persistent prediction error
                            at a context. The 7th component.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self.critic = Critic()
        self.curiosity = Curiosity()
        self.ledger = Ledger()
        self.simulator = Simulator(self.curiosity.world_model)
        self.planner = Planner(seed=seed)
        self.latent_inference = LatentInference()

    def reset_game(self, game_id: str) -> None:
        self.critic.reset_game()
        self.curiosity.reset_game()
        self.ledger.reset_game()
        self.latent_inference.reset_game()
        self.planner.reset_game(game_id)
        # Rebuild simulator with the fresh world model
        self.simulator = Simulator(self.curiosity.world_model)

    def reset_attempt(self) -> None:
        self.curiosity.reset_attempt()
        self.latent_inference.reset_attempt()
        self.planner.reset_attempt()

    def observe(self, transition: Transition) -> None:
        # All components update in parallel from the same transition.
        # The planner's underlying brain stack performs the substrate
        # update + surprise injection internally; we mirror updates for
        # our own diagnostic component handles.
        self.planner.observe(transition)
        self.critic.observe_transition(transition)
        if transition.before is not None:
            self.curiosity.observe(transition)
            # Feed prediction residuals to the Latent Inference module
            predicted = self.curiosity.world_model.predict(
                transition.before, transition.action,
            )
            predicted_high = {f for f, p in predicted.items() if p >= 0.5}
            actual = emit_atomic_facts(transition.before, transition.after)
            residual = (actual - predicted_high) | (predicted_high - actual)
            self.latent_inference.observe(
                transition.before, transition.action, transition.after, residual,
            )
        self.ledger.observe(transition)

    def act(self, state: State, budget: ComputeBudget) -> Action:
        return self.planner.act(state, budget)

    def end_of_attempt(self) -> None:
        self.planner.end_of_attempt()


__all__ = ["BioBrain"]
