"""biobrain.curiosity.residual — Eisen synthesis, agent-agnostic version.

Per Q1 gate failure (cd82/dc22 agent-ID unstable): pivoting the L1
explainer from "predict agent motion" to "predict per-context outcomes"
— the more general residual principle. Uses BayesianWorldModel's
per-(fact, context) Beta predictors, which don't require agent ID.

Architecture (the Eisen synthesis, instantiated minimally):

  SUBSTRATE: ActionScoreTable + Thompson sampling. The proven scorer.

  L1 EXPLAINER: BayesianWorldModel predicts P(fact in next state | context)
  for each observed fact. Trained on every transition (dense signal).

  RESIDUAL = surprise per transition: deviation between observed facts
  and predicted P. Signed so predictable transitions yield negative credit,
  surprising ones yield positive. This is the compression-progress
  approximation: predictable patterns self-extinguish as the model learns,
  surprises stay positive only until the model captures them.

  INJECTION: surprise is injected as fractional credit to ActionScoreTable
  during observe() — NOT as a competing posterior at decision time. This
  is the proven Proxy-style mechanism (commit 580b916) that avoided
  Gate 18's miscalibration trap. Action selection remains pure Thompson
  over the substrate posterior.

  NO LAMBDA. The surprise is in [-1, +1] and contributes to the
  Beta(α, β) directly. Its weight relative to score events emerges from
  data: as score events arrive, they contribute integer increments while
  surprise contributes fractional. The balance is automatic via posterior
  arithmetic.

  FIXATION INSTRUMENTATION: track per-sig the running average of
  surprise contributions. If a sig's surprise stays positive for many
  observations (model can't predict it), it's a noisy-TV trap candidate.

Per the gradient principle (residual only carries signal on
caused-object-manipulation games, not counter/flag cliffs): primary
test game is cd82 ("array equality" → gradient lever; partial matches
yield partial residuals), held-out is dc22 ("iterates entities" →
per-entity residuals).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from biobrain.types import (
    Action, ComputeBudget, State, Transition, action_kind,
    EVENT_LEVEL_INCREASED, EVENT_SCORE_INCREASED,
)
from biobrain.planner.agency import _candidate_actions
from biobrain.planner.representation import RepLoop
from biobrain.planner.posterior import ActionScoreTable
from biobrain.curiosity.world_model import BayesianWorldModel
from biobrain.curiosity.predicates import emit_atomic_facts


# Signed surprise is clipped before injection to bound its per-transition
# influence on the action-table's Beta posterior. Allows score events
# (integer +1 to α) to dominate fractional surprise updates over time.
SURPRISE_CLIP = 0.5


class MemoryBrainResidual:
    """Eisen synthesis brain: substrate + L1 explainer + signed-surprise credit.

    Same Thompson action policy as MemoryBrain V1; the only architectural
    change is in observe(): each transition's signed surprise (relative
    to BayesianWorldModel predictions) is injected as fractional credit
    to the action's signature in ActionScoreTable.

    This makes the substrate posterior dense (every transition adds
    information, not just score events) while keeping decision-time logic
    minimal (pure Thompson, no competing posteriors).
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._seed = seed
        self._rng: random.Random = random.Random(seed)
        self._rep: RepLoop = RepLoop()
        self._action_table: ActionScoreTable = ActionScoreTable()
        self._world: BayesianWorldModel = BayesianWorldModel()
        self._observed_count: int = 0
        # Fixation instrumentation: per-sig running sum + count of
        # surprise contributions (so we can compute the mean and detect
        # sigs that stay positive forever — the noisy-TV trap).
        self._surprise_log: dict[tuple, tuple[float, int]] = defaultdict(
            lambda: (0.0, 0)
        )

    def reset_game(self, game_id: str) -> None:
        self._rep = RepLoop()
        self._action_table = ActionScoreTable()
        self._world = BayesianWorldModel()
        self._observed_count = 0
        self._surprise_log = defaultdict(lambda: (0.0, 0))
        self._rng = random.Random(self._seed)

    def reset_attempt(self) -> None:
        self._world.reset_attempt()

    def _compute_signed_surprise(self, before: State, action: Action,
                                  after: State) -> float:
        """Surprise ∈ [-1, +1] of the observed transition.

        Positive: the model predicted the observed facts poorly → surprising.
        Negative: the model predicted correctly → predictable.
        Zero: no model evidence for this context yet (neutral).

        Implementation:
          predicted_p[fact] = P(fact in next state | context) from world model
          actual_facts = emit_atomic_facts(before, after)

          For each predicted fact f:
            if f ∈ actual: contribution = 1 - p   (low p, high present = surprise)
            else:          contribution = p       (high p, but absent = surprise)
            normalized to [-1, +1] via (contribution - 0.5) * 2

          For each actual fact NOT in predicted set (untracked yet):
            contribution = +1 (fully surprising, new fact for this context)

          Total = mean over contributions.
        """
        predicted = self._world.predict(before, action)
        actual = emit_atomic_facts(before, after)
        if not predicted and not actual:
            return 0.0
        contributions = []
        for fact, p in predicted.items():
            if fact in actual:
                # Lower p ⇒ more surprising that fact is present
                contributions.append((1.0 - p - 0.5) * 2.0)
            else:
                # Higher p ⇒ more surprising that fact is absent
                contributions.append((p - 0.5) * 2.0)
        # Untracked-yet facts (model has no posterior for them at this ctx)
        untracked = actual - set(predicted.keys())
        for _ in untracked:
            contributions.append(1.0)  # fully surprising
        if not contributions:
            return 0.0
        return sum(contributions) / len(contributions)

    def observe(self, transition: Transition) -> None:
        self._observed_count += 1
        self._rep.update(transition)
        # 1. Substrate update: standard action-signature posterior on score events.
        self._action_table.observe(transition)
        # 2. World model learns P(fact | context). Dense signal.
        self._world.observe(transition.before, transition.action,
                            transition.after)
        # 3. Compute signed surprise. Only inject if no score event fired
        #    (score events already contributed integer +1 to α; don't double-credit).
        scored = False
        for e in transition.events:
            if e.kind in (EVENT_SCORE_INCREASED, EVENT_LEVEL_INCREASED):
                scored = True
                break
        if scored or transition.before is None:
            return
        surprise = self._compute_signed_surprise(
            transition.before, transition.action, transition.after
        )
        # Clip to bound per-transition influence; score events still dominate.
        surprise_clipped = max(-SURPRISE_CLIP, min(SURPRISE_CLIP, surprise))
        # Inject as fractional credit to the action's signature.
        sig = ActionScoreTable._signature(transition.action, transition.before)
        n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
        # Positive surprise → α += fraction (encourages more visits).
        # Negative surprise → β += fraction (discourages — predictable).
        if surprise_clipped >= 0:
            self._action_table.counts[sig] = (n_obs, n_goal + surprise_clipped)
        else:
            self._action_table.counts[sig] = (
                n_obs + (-surprise_clipped),
                n_goal,
            )
        # Fixation instrumentation
        s_sum, s_n = self._surprise_log[sig]
        self._surprise_log[sig] = (s_sum + surprise_clipped, s_n + 1)

    def act(self, state: State, budget: ComputeBudget) -> Action:
        """Pure Thompson sampling over the action-signature posterior.

        The surprise-injection has already shaped the posterior at
        observe(). Decision time is clean: sample, argmax. No competing
        posteriors, no lambda, no hand-tuned switch.
        """
        candidates = _candidate_actions(state)
        if not candidates:
            raise ValueError(
                f"MemoryBrainResidual has no candidates from "
                f"available_actions={state.available_actions}"
            )
        best_score = -1.0
        best_action: Optional[Action] = None
        for a in candidates:
            sig = ActionScoreTable._signature(a, state)
            n_obs, n_goal = self._action_table.counts.get(sig, (0, 0))
            alpha = max(0.01, n_goal + 1.0)
            beta = max(0.01, n_obs - n_goal + 1.0)
            v = self._rng.betavariate(alpha, beta)
            if v > best_score:
                best_score = v
                best_action = a
        if best_action is None:
            best_action = self._rng.choice(candidates)
        return best_action

    def end_of_attempt(self) -> None:
        pass

    def n_visited_states(self) -> int:
        return self._rep.n_distinct_derived_states()

    def fixation_report(self) -> list:
        """Return sigs with persistent positive surprise — noisy-TV candidates.

        For each sig with ≥10 surprise observations and mean ≥0.3:
        sig keeps surprising the model, which suggests its outcome
        depends on hidden state the L1 model can't represent. Either
        it's a counter/flag cliff lever (residual drive trap) or a real
        gradient that just hasn't been learned yet.
        """
        out = []
        for sig, (s_sum, s_n) in self._surprise_log.items():
            if s_n < 10:
                continue
            mean = s_sum / s_n
            if mean >= 0.3:
                out.append((sig, mean, s_n))
        out.sort(key=lambda x: -x[1])
        return out
