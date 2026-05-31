# biobrain — Active Inference Architecture for ARC-AGI-3

A bio-RL-grounded agent that wakes up in an unfamiliar grid environment with no instructions, no rewards, no pretraining, and reasons its way to the puzzle's solution through active inference.

---

## The problem ARC-AGI-3 actually poses

Earlier ARC benchmarks were static: given input grids, produce an output grid. ARC-AGI-3 is fundamentally different — it's **interactive**. The agent is dropped into a turn-based grid environment with:

- No instructions ("What should I do?")
- No stated goals ("What does winning look like?")
- No predefined rules ("What do my actions do?")
- A finite action budget per attempt

The agent must poke around, infer the physics, identify a goal that no one stated, and produce action sequences that satisfy it. Then do it again on the next level of the same game (where physics carries over) and again on the next *game* (where physics is wiped).

Frontier LLMs score under 1% on this benchmark. Humans score 100%. The gap is not a gap in pattern recognition. It is a gap in **online active inference** — the capacity to act in order to learn, learn in order to predict, predict in order to plan, and plan in order to act. LLMs are excellent at model-free pattern matching from training data; they have no online prefrontal cortex.

biobrain is an attempt to build that prefrontal cortex from first principles.

## What biobrain is, structurally

The agent is organized into six named cortical regions, each with both a biological identity and a precise ML/RL counterpart:

| Region | ML/RL term | Role |
|---|---|---|
| **Motor Cortex** | Action space / DSL | The agent's "muscles": atomic environment actions plus compositional combinators |
| **Curiosity** (dopamine system) | ICM intrinsic reward via signed Reward Prediction Error | Learns the world model by chasing surprise; "boredom" emerges as the model sharpens |
| **Critic** (limbic / OFC) | Heuristic value function over win-state aesthetics | Identifies what "looks like winning" without being told: compression, symmetry, cohesion, noise-elimination |
| **Ledger** (hippocampus / DLPFC) | Episodic memory + scientific method | Persists across levels within a game (wipes between games); abstracts winning trajectories into reusable parameterized scripts |
| **Simulator** (prefrontal cortex, dorsolateral) | Forward dynamics model | Mental sandbox — predict the consequence of an action before executing |
| **Planner** (PFC executive + basal ganglia) | Actor / policy | Combines exploration, exploitation, intrinsic reward, lookahead, and Ledger scripts into action selection |

The agent's run loop, in plain language: *thrash to learn the physics, build a model that predicts consequences, find the gradient of "win-state likelihood" using compression-based aesthetic primitives, plan via short rollouts in the mental sandbox, execute, and when something works — abstract it into a script for the next level.*

## The principle that holds it together

Every component reduces to one objective: **compression-progress** (after Chollet 2019 and Schmidhuber's classical formulation).

A state is more "win-like" if its representation has lower description length. Symmetry compresses (one half encodes the other). Object cohesion compresses (fewer connected components). Noise elimination compresses (fewer outlier predicates). Pattern recurrence compresses (one region encodes another). Each Critic primitive is a different *projection* of the same compression objective onto a different aspect of grid structure.

Likewise the Curiosity / dopamine drive is compression-progress over time: a state-transition is interesting if it *would* reduce the world model's prediction error. Once the model has captured the rule, surprise (and dopamine) drop to zero automatically. The agent gets "bored" without any explicit timer.

This unification matters because it means biobrain has a single objective function — even though it manifests as multiple Critic extractors and Curiosity signals at the implementation level.

## What separates biobrain from neural ICM / RND

The biological-RL literature operationalizes intrinsic motivation through neural networks: a learned encoder φ maps grids to latent vectors; a forward model predicts next-latent given action; the prediction error is the intrinsic reward. This is **ICM** (Pathak 2017) or **RND** (Burda 2018).

biobrain implements the same architecture **symbolically**, with three deliberate departures:

1. **The encoder φ is hand-engineered, not learned.** `emit_atomic_facts` projects each state onto Spelke-primitive axes (Object × Color, Object × Size, Object × Place, Number × {Color, Size, Place}, plus pairwise joints for object identity). The vocabulary is bounded (~250 possible joints, ~20 active per state) and derived from one principle: *each predicate is a projection of state onto a Spelke axis or a pairwise joint for identity-bound queries.*

2. **The world model is Bayesian and non-parametric.** Per-`(fact, context)` Beta posteriors of `P(fact in next state | context)`, where context is `(action_kind, target_color, level)`. No gradient descent. No backprop. Online updates after each transition. Boredom = posterior sharpening.

3. **No training.** biobrain has zero pretraining and zero offline fitting. It is purely an online algorithm. This is a hard constraint imposed by ARC-AGI-3's out-of-distribution covenant: the test set is held out and there is no learning *across* games.

The trade-off is direct: we lose representational flexibility (we can only see what `emit_atomic_facts` emits) but we gain interpretability, sample efficiency, and the ability to start cold without any weight initialization.

## The build phases (with empirical gating)

biobrain is built in phases. Each phase has a measurement gate that decides whether to proceed.

**Phase 0 — Vocabulary alignment.** ✓ Complete. The Critic and World Model must speak the same language. Both consume the fact alphabet emitted by `emit_atomic_facts`. Critic extractors that operate on raw grid cells (legacy) are still supported but cannot inform forward simulation.

**Phase 1 — World Model accuracy validation.** ✓ Complete. Per-step fact-prediction F1 measured on five public games (warm-up: 50 steps; scored over remaining transitions):

| Game | F1 | Verdict |
|---|---|---|
| lp85 | 0.94 | ✓ simulator viable |
| g50t | 0.86 | ✓ simulator viable |
| vc33 | 0.82 | ✓ simulator viable |
| cd82 | 0.73 | ✓ simulator viable |
| bp35 | 0.58 | ⚠ shallow only |
| r11l | 0.53 | ⚠ shallow only |

Four of six games clear F1 > 0.7. The simulator was approved for build.

**Phase 2 — Simulator + 1-step lookahead.** ✓ Complete. For each candidate action, the Simulator predicts the next-state fact set via the World Model; the Critic evaluates predicted facts; the action posterior gets a `current_distance − predicted_distance` bonus. Result (N=15 × 200 steps):

| Game | Planner alone | + 1-step lookahead |
|---|---|---|
| vc33 | 33% | **60%** (+27pp) |
| lp85 | 80% | **93%** (+13pp) |
| cd82 | 0% | 0% |
| g50t | 0% | 0% |
| bp35 | 0% | 0% |

Lookahead is a real lift on games where the World Model is accurate AND L3 has identified the right structural goal. Cold-start games (cd82 etc.) remain floored — they need either deeper lookahead or richer Critic primitives.

**Phase 3 — Ledger (scientific method).** ✓ Complete (machinery built; transfer mechanism not yet validated). On a score event, the Ledger captures the last 5 actions, parameterizes them via entity-color anchors (`click(x, y) → click_on_color(entity_color_at(x, y))`), and stores them as a DSL Program with hierarchical Beta per `(program, level)`. On entry to a new level, promoted programs from prior levels with confidence ≥ 0.7 are executed first.

Current limitation: the brains rarely reach Level 2 within the action budget on most games, so the cross-level transfer mechanism hasn't yet been exercised. The Ledger fills correctly when scores fire (13 entries on lp85, 10 on vc33, 2 on r11l in 15 attempts × 250 steps).

**Phase 4 — Critic primitive expansion.** Partial. Five primitives built:
- `Compression` — fact-count and color-palette reduction (always-on background)
- `Noise` — tiny / small entity elimination (bp35-class)
- `Symmetry` — mirror-pair alignment using the `entity_color_quadrant` joint
- `ChangeDynamicsFactSpace` — canvas-vs-target via per-quadrant color-set diff
- `StaticPatternRecurrence` and `ChangeDynamics` (raw-cell legacy)

Coverage with the current six extractors: **25/25 public games match at least one extractor.** The "neither" set (where no extractor identifies any goal) is empty. This was the gate for Phase 4: structural coverage of the benchmark.

**Phase 5 — MCTS (full tree search over DSL).** Not yet built. Gated on showing that 1-step lookahead provides robust lift first.

## The discipline principles

biobrain was built on top of, and is now distilled from, ~25 ablation experiments. Six principles emerged that govern further work:

1. **Upstream-first debugging.** When a downstream measurement looks wrong, FIRST verify the inputs. The lp85 lookahead "regression" (80% → 13%) looked like a Critic bug. It was actually a missing extractor upstream; once added, the bug disappeared (80% → 93%).

2. **Principled derivation over hardcoded formulas.** No `0.7 × A + 0.3 × B` blends. Components are combined via their intrinsic weights (e.g., goal DL-savings), or not at all. Magic constants get `# RL-TODO` markers and a comment naming what learnable signal would replace them.

3. **Single source of truth for cross-cutting algorithms.** If a computation appears in two places, it lives in one file and the others import. No copy-paste.

4. **Abstraction-level alignment.** Components that exchange data must agree on the abstraction level. The World Model and Critic must both speak fact-space; mixing fact-space and pixel-space breaks the Simulator.

5. **House-model lifecycle discipline.** `reset_game` wipes everything except static perception. `reset_attempt` preserves the substrate posterior, world model, history, and Ledger. The intra-game / inter-game distinction is non-negotiable (it's required by the covenant AND it's what enables the scientific method).

6. **Generalize-by-protocol, not by inheritance.** New extractor types implement a `GoalExtractor` Protocol; they don't extend a base class. Each component is falsifiable independently.

## Where biobrain currently lives in the validation curve

biobrain scores on 4/25 public games (lp85, vc33, r11l, lf52) at substrate-only competence comparable to the field-leading ~17-level baselines. With 1-step lookahead enabled, scoring rates lift meaningfully on the games where the World Model is accurate.

Cold-start games (cd82, g50t, bp35, ~18 others) remain at 0%. Two distinct subclasses:

- *Sequence-dependent* games (cd82): require multi-step interactions the single-step planner can't represent. The Ledger + MCTS are the architectural responses; they're built / planned respectively.

- *L3-gap* games (the previously-"neither" set): the L3 layer needed richer extractor types. Coverage analysis after adding Compression / Noise / Symmetry / ChangeDynamicsFactSpace shows 25/25 games now have a detected goal. Whether those goals translate to scoring is the next experiment.

This is the honest competence envelope: biobrain has all four ARC-AGI-3 capabilities the benchmark designers named (Exploration, Modeling, Goal-setting, Planning + Execution), each operating in isolation. Their joint behavior on cold-start games is the open question that the remaining build phases address.

## Open questions

These are the questions whose answers will determine whether biobrain reaches human-level performance or hits a structural ceiling.

1. **Is fact-space deep enough?** Our predicate vocabulary is bounded (~30 predicate templates with 256 possible joints). It successfully covers static layouts and Spelke-grounded properties, but it does not represent: shape topology (closed shapes, holes), 2D transformations (rotations, reflections of entities), or fine-grained relational facts (entity A is N cells away from entity B). The Phase 6 question is whether the games we still fail on require extending this vocabulary or whether the existing vocabulary + better planning suffices.

2. **Will MCTS escape the single-step ceiling?** cd82's mechanic is a 2-step modal interaction (select color, then paint). 1-step lookahead can't represent this. MCTS with depth ≥ 2 might — but only if World Model error doesn't compound destructively across rollouts. Phase 5 will measure this directly.

3. **Does the Ledger generalize across levels?** The machinery builds entries on score events. The cross-level promotion mechanism has not yet been exercised because brains rarely reach Level 2 within the action budget. A targeted experiment on level-2-reachable games is needed to validate that scientific-method transfer actually works.

4. **At what point does compression-as-objective break?** Some games may score on *anti-compression* moves (e.g., growing structure from a seed). The compression objective then has the wrong sign. Phase 4's Critic primitives are all compression-aligned; we have no built-in opposite. If we find a game class where the objective sign is wrong, the architecture needs a learnable critic-direction selector.

5. **What is the cleanest learnable replacement for the hand-set thresholds?** The Critic extractors have parameters (`DYNAMIC_THRESHOLD`, `MATCH_THRESHOLD_LOW`, etc.) that were chosen by inspection. Each is a candidate for RL: the threshold that maximizes per-game scoring would be derivable from per-attempt outcomes via a meta-bandit. But this would require an offline learning loop, which conflicts with the no-training constraint. Resolution unclear.

## What "shipped" looks like

biobrain ships when:

- All five build phases pass their measurement gates.
- The competence envelope on public games is documented with confidence intervals.
- The held-out covenant is enforced (no information leakage from `bench/private/` to `biobrain/`).
- The architecture documentation suffices for a peer researcher to reproduce both the methodology and the empirical findings.

The current state corresponds to Phase 4 complete, Phase 5 (MCTS) the immediate next target. The repo at `github.com/dramdass/biobrain` is the canonical artifact.

---

*biobrain is not "an LLM that solves ARC-AGI-3." It is the brain-architecture argument that solving ARC-AGI-3 requires online active inference, that active inference decomposes cleanly into six biological-RL primitives, and that those primitives can be implemented symbolically (no neural nets, no training) with sample efficiency the neural variants do not have. Whether this argument survives contact with the full benchmark is what the remaining build phases will determine.*
