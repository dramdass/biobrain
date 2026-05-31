# Architecture Principles

Disciplined patterns that apply across the biobrain codebase. Each one is
here because we violated it during the design spike and paid for it.
Re-reading these saves future diagnostic time.

---

## 1. Upstream-first debugging

**When a downstream measurement looks wrong, FIRST verify the inputs.**

The lp85 lookahead "regression" (80% → 13%) looked like a Critic bug. The
actual cause was upstream: a missing extractor (`ChangeDynamicsFactSpace`)
that the lookahead Critic depended on. Adding it fixed everything:
lp85 80% → 93%, vc33 33% → 60%.

Checklist when a probe result looks wrong:

- Did the input data (state, transitions, goal stream) change?
- Did an upstream component's contract or signature change?
- Did a recently-added component change the observable behavior of
  something this one depends on?
- Is the measurement repeatable, or is it seed noise?

Only AFTER ruling these out: suspect the downstream component itself.

## 2. Principled derivation over hardcoded formulas

**Don't combine signals with hand-tuned blends. Let components' intrinsic
weights/posteriors do the combination.**

Hack patterns we keep regressing into:

- `0.7 × A + 0.3 × B` — top-level hardcoded blend. *Right form*:
  weighted-mean by the components' own DL-saving weights. If
  game-specific signals exist with high weight, they dominate; if not,
  background contributes by default.
- `GOAL_BIAS_MAX = 0.3` — magic additive constant. *Right form*: bias is
  the signal itself, clipped to a scale-matched bound (e.g., SURPRISE_CLIP).
- `delta * 2.0` — arbitrary multiplier before clipping. *Right form*: clip
  the raw delta.

When a hardcoded number appears in a new build, it gets a `# RL-TODO`
marker and a comment explaining what observed distribution or learned
posterior could replace it.

Active RL-TODO list (these will eventually be data-derived):

- `biobrain/critic/compression.py`: `N_FACTS_CAP=80`, `N_COLORS_CAP=12`
- `biobrain/critic/noise.py`: `0.05` noisy-fraction floor
- `biobrain/critic/change_dynamics.py`: `DYNAMIC_THRESHOLD=0.10`, `STATIC_THRESHOLD=0.02`
- `biobrain/critic/change_dynamics_facts.py`: per-quadrant thresholds
- `biobrain/critic/pattern_recurrence.py`: `MATCH_THRESHOLD_LOW/HIGH`
- `biobrain/ledger/ledger.py`: `DEFAULT_PROMOTION_THRESHOLD=0.7`
- `biobrain/curiosity/residual.py`: `SURPRISE_CLIP=0.5`
- `biobrain/planner/commit_monitor.py`: `VIOLATION_SURPRISE_THRESHOLD=0.35`
- `0.5` fact-presence threshold in WM predictions

## 3. Single source of truth for cross-cutting algorithms

**If a computation appears in two places, it lives in one file and the
others import. No copy-paste.**

Examples we caught during the design spike:

- `_compute_signed_surprise` was defined once in `MemoryBrainResidual`
  and duplicated in `Curiosity` facade. Fixed: Curiosity delegates.
- `_critic_distance_from_facts` lives in `lookahead_planner`; ledger_planner imports.

When in doubt, search for duplicated function bodies before writing
new ones.

## 4. Abstraction-level alignment

**Components exchanging data must agree on the abstraction level.**

The World Model and Critic must both speak fact-space. Mixing
fact-space and pixel-space breaks the Simulator. The lookahead Critic
in v0.1 had this bug — `StaticPatternRecurrence` operated on raw cells
while Curiosity emitted facts. Lookahead's predicted-state Critic
silently dropped to background-only because no fact-space extractor
gave a game-specific signal.

Pattern: every Critic extractor must expose a `distance_fn` that accepts
EITHER a State (for observe-time critique) OR a fact set (for predicted-
state lookahead). See `biobrain/critic/compression.py` for the pattern.

## 5. House-model lifecycle discipline

**Be explicit about what wipes at `reset_game` vs `reset_attempt`.**

| Boundary | Wipes | Persists |
|---|---|---|
| `reset_game` | substrate posterior, world model, transition history, ledger, current goals, salience state, latent schemas | static perception (Spelke object detector — outside the brain library) |
| `reset_attempt` | in-flight Program continuation, salience fine-attention queue | substrate posterior, world model, history, ledger, modeled vars, banked surprises, affordance |
| `on_level_change` (within attempt) | Planner's in-flight Program (force re-engage cold path) | everything else |

Inter-game amnesia is required by the OOD covenant. Intra-game memory
enables the scientific-method (Ledger). Level boundaries trigger Ledger
promotion but don't wipe game-physics knowledge.

## 6. Generalize-by-protocol, not by inheritance

**New components implement a Protocol; they don't inherit from a base class.**

`biobrain.critic.GoalExtractor` is a Protocol (just `.name` and
`.detect()`). The L3 orchestrator runs all registered extractors without
knowing implementation details. Same for `biobrain.protocols.Encoder`,
`biobrain.protocols.Adapter`. New extractor/adapter types = new file,
no base class to extend, no constructor coupling.

The ablation discipline (each component falsifiable independently)
follows from this.

## 7. Predicates as Spelke-axis projections

**The predicate vocabulary is derived from one principle: each predicate
is a projection of state onto a Spelke primitive axis or a pairwise joint
of axes for object identity.**

Single-axis projections (Object × Color, Object × Size, Object × Place;
Number × Color, Number × Size, Number × Place) capture properties. Joint
predicates (Color × Place, Color × Size) capture identity-bound queries
that the Critic's relational primitives (Symmetry, Cohesion) need.

Skip: shape predicates (not Spelke-canonical), pairwise relational
(O(N²) explosion), topology (over-engineered for our sample regime).

When adding a new predicate: name which Spelke axis or joint it projects
from. If it doesn't have one, reconsider whether it's principled or
ad-hoc.

## 8. Commit-and-monitor over per-step evaluation

**Per the phenomenology: humans don't reason every step. They commit a
hypothesis and execute open-loop until violation.**

The Planner's three-path control:

- **Hot path** (default, every step): cheap. Continuation step + WM
  prediction check.
- **Warm path** (every transition, in observe()): learning never sleeps.
  WM updates, Salience banks surprises, Ledger consolidates.
- **Cold path** (violation OR program-end): full reasoning. Thompson +
  Critic + Simulator + Ledger + Affordance.

This unifies execution + learning + budget into one event-driven loop.
A violation IS a prediction error IS the compression-progress signal IS
where learning happens. Don't build separate machinery for each.

## 9. Composer-mediated communication

**Components never call each other directly. The composer (BioBrainV2) is
the only thing that knows the inter-component wiring.**

Each component is testable in pure isolation with synthetic inputs.
Hardening one component doesn't ripple into others. The composer grows
as components multiply, but at 8 components it's manageable and there's
one place to look for "what flows where."

## 10. Salience as central organ, not utility

**Salience curates the modeled-variable set, triggers fine perception at
violations, banks surprises, and proposes new predicates. It is the
representation-state-management organ.**

Per the phenomenology, salience is not a perception detail — it's the
mechanism by which the brain refines its own representation. Subsumes
the prior "Latent Inference" module: observable-tell explanations first,
schema-fallback only when no observable feature explains the residual.

Without Salience as central, the combinatorial fact-space explodes
and the brain can't escape from "wrong representation."

---

## Summary

If you're stuck on a measurement you don't understand:
1. **Upstream first.** Verify inputs.
2. **Principled derivation.** No magic numbers.
3. **Single source of truth.** No copy-paste.
4. **Abstraction alignment.** Both ends fact-space or both ends pixel-space.
5. **Lifecycle discipline.** Game vs attempt vs level — get it right.
6. **Protocols, not inheritance.** Independent falsifiability.
7. **Spelke-axis predicates.** Principled derivation, bounded combinatorics.
8. **Commit-and-monitor.** Don't reason every step.
9. **Composer-mediated.** Components isolated.
10. **Salience central.** Curation + attention + proposing all in one organ.
