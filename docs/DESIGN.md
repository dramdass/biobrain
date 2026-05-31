# biobrain v0.2 — Design

The brainstorm-approved architecture spec. Conceptual / interface level only —
no implementation details. Implementation plans live in separate plan docs.

---

## 1. Architecture overview

biobrain is a generic-typed brain library composed of **8 named components**,
exchanging information through a **composer** rather than direct dependencies,
designed to run against any environment via an **adapter** that supplies
`State` / `Action` types and an `Encoder`.

```
                  ┌─────────────────────────────────────────────┐
                  │              ADAPTER (per env)              │
                  │  supplies: StateT, ActionT, Encoder,        │
                  │  optional affordance prior, env binding     │
                  └─────────────────────────────────────────────┘
                                      │
                                      ▼
                  ┌─────────────────────────────────────────────┐
                  │              BIOBRAIN CORE                  │
                  │                                              │
                  │   1. Motor Cortex   (DSL action vocabulary)  │
                  │   2. Perception     (Encoder + finer attn)   │
                  │   3. Salience       (curator + LI subsumed)  │
                  │   4. Curiosity      (Bayesian WM + ICM)      │
                  │   5. Critic         (library of extractors)  │
                  │   6. Simulator      (forward queries via WM) │
                  │   7. Ledger         (per-game program memory)│
                  │   8. Planner        (commit-and-monitor)     │
                  │                                              │
                  │   Composer routes ALL inter-component flow.  │
                  │   No component knows about any other.        │
                  └─────────────────────────────────────────────┘
```

**Architectural commitments (each locked via the brainstorm interview):**

| # | Commitment |
|---|---|
| 1 | Control loop: commit-and-monitor (hot/warm/cold paths) |
| 2 | 8 components; Salience subsumes Latent Inference; Simulator separate |
| 3 | Generic-typed brain (parameterized over StateT, ActionT) |
| 4 | Encoder is pluggable; brain is fact-set-native; State access mediated |
| 5 | Modal state: attend-finer first, schema fallback |
| 6 | Affordance prior: emergent default + adapter slot |
| 7 | Inter-component communication: composer-mediated |
| 8 | Lifecycle: two-verb (`reset_game` / `reset_attempt`) |
| 9 | Level transitions: implicit-by-context + explicit `on_level_change` hook for Ledger/Planner/Salience |
| 10 | Staging: big-bang v0.2 refactor; push when all 8 work together |

---

## 2. Per-component responsibilities

Each component, conceptual only: what it owns, what it produces, what it
consumes, how it wipes.

### 2.1 Motor Cortex
- **Owns:** atomic action vocabulary + combinators that compose them into Programs
- **Produces:** Programs (composable action sequences) and ActionSigs (abstract action descriptors)
- **Consumes:** nothing (stateless construction kit)
- **Lifecycle:** no state; nothing wipes

### 2.2 Perception
- **Owns:** the Encoder — maps raw env observation to State and from State to fact set; multi-granularity perceptual modes (coarse default, fine on Salience request)
- **Produces:** State and fact set
- **Consumes:** raw env observation (from adapter); Salience's fine-attention requests
- **Lifecycle:** stateless re: state representation. Current fine-attention focus is held by Salience, not Perception itself

### 2.3 Salience
- **Owns:** the curated modeled-variable set; banked-surprises log; fine-attention queue; affordance posterior; any instantiated latent schemas. The central representation-state-management organ.
- **Produces:** curated fact filter; fine-attention requests to Perception; new predicate templates registered into active vocabulary; affordance prior values consumed by Planner
- **Consumes:** Curiosity's signed surprise; Perception's fact set; the Encoder (to ask for finer perception)
- **Backpropagation:** when a new predicate explains banked surprises, retroactively re-interpret them and boost the predicate's confidence (v0: scaffolded but inactive; v1: active)
- **Lifecycle:**
  - `reset_game`: wipes everything (curated vars, banked surprises, affordance posterior, schemas)
  - `reset_attempt`: clears short-term fine-attention queue; retains modeled vars, banked surprises, affordance posterior, schemas
  - `on_level_change`: re-evaluates which variables are *active* at the new level (the full modeled-var set persists; the active subset rotates)

### 2.4 Curiosity
- **Owns:** the Bayesian world model (per-(fact, context) Beta posteriors of `P(fact_next | context)`); signed-surprise computation
- **Produces:** next-state fact predictions (for Simulator); signed surprise per transition (for Salience, and dual-used by Planner as violation signal); WM posterior updates (the warm path)
- **Consumes:** Transitions; Salience-curated facts (only active vars predicted/learned)
- **Lifecycle:**
  - `reset_game`: wipes WM posteriors
  - `reset_attempt`: preserves WM
  - Implicit level partition via the `level` field in context

### 2.5 Critic
- **Owns:** library of goal-extractors (Compression, Noise, Symmetry, ChangeDynamicsFactSpace, StaticPatternRecurrence, …); L3 orchestrator combining by weight
- **Produces:** ProtoGoals — soft state-distance functions
- **Consumes:** State and fact set; TransitionHistory (for dynamics-aware extractors)
- **Lifecycle:**
  - `reset_game`: wipes TransitionHistory
  - Extractors are stateless

### 2.6 Simulator
- **Owns:** forward-query interface to the WM. Stateless wrapper.
- **Produces:** predicted next-state fact sets given (state, action)
- **Consumes:** WM (from Curiosity), action being simulated, current state
- **Lifecycle:** stateless; nothing wipes
- **Deferred:** multi-step rollouts (`rollout(state, sequence, depth)`) — v0 has 1-step only

### 2.7 Ledger
- **Owns:** banked successful trajectories as parameterized DSL Programs; hierarchical Beta posteriors per (program_id, level)
- **Produces:** promoted Programs at level entry (those with prior-level confidence ≥ threshold)
- **Consumes:** Transitions; score events; the Encoder (to parameterize click positions into entity-color anchors)
- **Lifecycle:**
  - `reset_game`: wipes all entries
  - `reset_attempt`: preserves
  - `on_level_change`: surfaces promoted Programs for new level

### 2.8 Planner
- **Owns:** substrate posterior (per-action-signature Beta over goal-distance reduction); in-flight Program continuation; violation threshold; hot/warm/cold control logic
- **Produces:** Action to execute next
- **Consumes:** Critic ProtoGoals; Simulator predictions (per candidate action); Ledger promoted Programs; Salience affordance posterior; Curiosity surprise (violation trigger)
- **Lifecycle:**
  - `reset_game`: wipes substrate posterior
  - `reset_attempt`: preserves substrate; wipes in-flight Program continuation
  - `on_level_change`: abandons in-flight Program; re-engages cold path on next act()

---

## 3. Composer dataflow

The BioBrain composer is the only thing that knows the wiring. Each
component is pure (no other-component references). The composer routes
data flow in three call paths.

### 3.1 observe(transition) — order matters

The composer routes observe in this specific order because each downstream
component depends on the previous one's updated state:

1. **Encoder.encode(transition.after)** → fact_set (and fact_set_before from transition.before)
2. **Curiosity.predict(before, action)** → predicted_facts; compute signed surprise = f(predicted, actual)
3. **Curiosity.update_wm(before, action, after)** → posterior updates
4. **Salience.observe(surprise, predicted, actual)** → bank surprises, request fine attn if persistent error, update affordance posterior, possibly propose new predicates
5. **Critic.update_history(transition)** → TransitionHistory update (consumed by ChangeDynamics-family)
6. **Ledger.observe(transition)** → rolling buffer of (action, before_state); on score event, abstract last K actions into Program
7. **Planner.observe(transition, surprise)** → substrate posterior update (surprise + ΔGoal credit), set violation flag if surprise exceeds threshold
8. **If transition.after.level > transition.before.level**: composer fires `on_level_change(prev_level, new_level)` on Ledger, Planner, Salience

### 3.2 act(state, budget) — three paths

#### Hot path (default, every step)

```
In-flight Program continuation? ──Yes──▶ Step continuation → ActionSig
        │                                          │
        │                                          ▼
        │                                  Encoder.resolve(ActionSig, state)
        │                                          │
        │                                          ▼
        │                                     return Action
        ▼
Violation pending OR program completed?
        │
        Yes ──▶ escalate to COLD PATH
```

#### Cold path (decision time, runs on violation or program-end)

```
1. Encoder.encode(state) → fact_set
2. Critic.evaluate(state, fact_set) → ProtoGoals
3. Ledger.promote_at_level(state.level) → candidate Programs
4. Salience.affordance_posterior() → per-action-class priors
5. For each candidate (Program or atomic Action):
     a. Simulator.simulate_one(state, candidate) → predicted_facts
     b. predicted_distance = state_distance_to_goals(predicted_facts, ProtoGoals)
     c. current_distance = state_distance_to_goals(state, ProtoGoals)
     d. score = Thompson(substrate) + affordance_bonus + (current_d - predicted_d)
                + Ledger_promotion_bonus_if_applicable
   Pick best. Commit to Program (or wrap atomic Action as 1-step Program).
6. Step the chosen Program → ActionSig
7. Encoder.resolve(ActionSig, state) → Action
8. return Action
```

#### Warm path (implicit, runs on every observe())

WM updates, Critic history updates, Salience surprise banking, affordance
posterior updates, Ledger trajectory tracking. Learning never stops, even
while the hot path is cheap.

### 3.3 Lifecycle dataflow

```
reset_game(game_id)
   │
   ├──▶ Salience.reset_game()    [wipe all]
   ├──▶ Curiosity.reset_game()   [wipe WM]
   ├──▶ Critic.reset_game()      [wipe history]
   ├──▶ Ledger.reset_game()      [wipe entries]
   └──▶ Planner.reset_game()     [wipe substrate posterior]

reset_attempt()
   │
   ├──▶ Salience.reset_attempt() [keep modeled vars, clear fine-attn queue]
   ├──▶ Curiosity.reset_attempt()[keep WM]
   ├──▶ Ledger.reset_attempt()   [keep entries]
   └──▶ Planner.reset_attempt()  [keep substrate; clear in-flight Program]

on_level_change(prev_level, new_level)
   │
   ├──▶ Ledger.on_level_change()  [trigger promotion]
   ├──▶ Salience.on_level_change()[re-evaluate active vars]
   └──▶ Planner.on_level_change() [abandon in-flight Program]
```

### 3.4 Dual role of signals

Some signals serve multiple roles intentionally:

- **Curiosity's signed surprise**: simultaneously (a) substrate credit injection (the existing residual mechanism), (b) Salience's banking signal, (c) Planner's violation trigger. Computed once per transition; consumed three ways.
- **Salience's curated variables**: simultaneously (a) filter on Curiosity's predicted facts (only active vars learned), (b) constraint on Critic extractors (only active vars goal-relevant), (c) Planner's reasoning scope.
- **Encoder**: simultaneously (a) raw observation → State, (b) State → fact set, (c) ActionSig → Action resolution. One protocol, three call sites.

---

## 4. Adapter pattern

### 4.1 Adapter responsibilities

The adapter is **per-environment** code that sits outside the brain library
and connects it to a specific env (e.g., ARC-AGI-3).

**Required:**
- Concrete `StateT`, `ActionT`, `TransitionT` types satisfying the `biobrain.protocols`
- An `Encoder` implementation (default `DefaultSpelkeEncoder` works if StateT matches the reference type)
- Environment-binding code: how to reset, step, parse observations

**Optional:**
- `initial_affordance_priors() → dict[action_kind, prior]` — per-action-class
  Beta priors the brain seeds its affordance posterior with. Default: no
  priors (uniform start, emergent shaping). Populated only when
  covenant-relaxation is desired.

### 4.2 Brain library's view of the adapter

The brain library is parameterized over StateT, ActionT. It never imports
adapter-specific code. The composer accepts an adapter instance at
construction and reads:
- the Encoder (for predicate emission and ActionSig resolution)
- the affordance priors (if provided; otherwise uses uniform)

### 4.3 The ARC-AGI-3 adapter (the one we ship)

Lives in `biobrain/adapters/arc/`. Wraps the `arc_agi` SDK. Provides:
- Concrete State (with entities, raw_grid, level, available_actions, score)
- Concrete Action (tuple-based)
- The `DefaultSpelkeEncoder` (or its arc-specific subclass if extensions needed)
- Env binding (`ArenaEnv` class)
- No affordance priors by default (strict covenant)

---

## 5. Validation strategy

End-to-end is the headline. Per-component diagnostics are how we know
*where* lift comes from.

### 5.1 Per-component diagnostics

| Component | Diagnostic | Pass criterion |
|---|---|---|
| Motor Cortex | DSL synthesis recovers known mechanics from positive trajectories | succeeds on ≥3 known-scoring games (lp85, vc33, r11l) within budget |
| Perception | multi-granularity probe: fine attention emits additional predicates | >0 new predicates emitted on synthetic test |
| Salience | banked surprise count tracks actual surprise count | rank correlation ≥ 0.8 |
| Curiosity | per-step fact-prediction F1 (Phase 1 complete) | F1 ≥ 0.7 on 4/6 games (achieved) |
| Critic | extractor coverage + goal-correctness | coverage 25/25 (achieved); correctness verified on 4 scoring games |
| Simulator | predicted vs actual fact set after 1 step | matches WM F1 |
| Ledger | trajectory abstraction rate; entries fill correctly on score events | entries accumulate per attempt (already verified) |
| Planner | hot path significantly cheaper than cold; scoring lift preserved | hot path 5-10× cheaper; lp85/vc33 lookahead lift survives refactor |

### 5.2 End-to-end measurement

- **Headline:** scoring rate per game with Wilson CI; max-level reached
- **Critical comparisons:**
  - biobrain v0.2 vs current substrate-only baseline (lp85, vc33, r11l)
  - biobrain v0.2 vs biobrain v0.1 lookahead (refactor regression check)
- **Pending experiments:**
  - Phase 5a: cd82 with hand-coded armed_color latent. Prediction: WM F1 jumps 0.73 → 0.85+. Decides whether Salience schema-fallback is the right v0 build.
  - Phase 7: Ledger transfer on vc33 (the one game reaching Level 2). Tests cross-level promotion.

### 5.3 "No bugs" criteria

- All existing tests pass after refactor (535+ tests)
- New per-component diagnostic tests pass
- End-to-end smoke test: biobrain v0.2 on lp85 produces ≥80% scoring (preserves current capability)

---

## 6. Deferred — explicit honesty about what's not yet built

- **MCTS over DSL.** Gated on Phase 5a outcome.
- **Multi-step Simulator rollouts.** Stub. 1-step only in v0.
- **Full schema library for latent fallback.** v0 has minimal templates; v1 adds learned residual clustering.
- **Backpropagation of confirmations to priors.** Phenomenology mechanism; v0 scaffolds the hook; v1 activates.
- **Adapter-provided affordance prior.** Slot built; no values populated. Covenant decision deferred.
- **Full 25-game competence envelope measurement.** Run after v0.2 push.

---

## 7. ARC-AGI-3 capability mapping

The benchmark designers identify four capabilities. Each maps to a primary
biobrain component plus supporting ones:

| Capability | Primary | Supporting |
|---|---|---|
| Exploration | Curiosity | Salience (banked surprises drive priority), Planner (Thompson over substrate) |
| Modeling | Curiosity (WM) | Perception (encoder + finer attention), Salience (representation evolution) |
| Goal-setting | Critic | Salience (curates vars feeding Critic) |
| Planning + Execution | Planner | Simulator (lookahead), Ledger (Programs), Motor Cortex (action space) |

The mapping is reasonably honest at the *primary-component* level. The
*supporting* roles are real: removing the supporting component degrades but
doesn't eliminate the capability.

---

## 8. Discipline principles (carried over)

These govern further work in the v0.2 repo:

1. **Upstream-first debugging.** When a downstream measurement looks wrong, FIRST verify inputs.
2. **Principled derivation over hardcoded formulas.** No magic blends; magic constants get `# RL-TODO`.
3. **Single source of truth for cross-cutting algorithms.** No copy-paste.
4. **Abstraction-level alignment.** Components exchanging data must agree on level (fact-space vs raw-cell).
5. **House-model lifecycle discipline.** `reset_game` wipes everything game-specific; `reset_attempt` preserves intra-game state.
6. **Generalize-by-protocol, not by inheritance.** New extractor types implement `GoalExtractor`; new adapters implement the adapter protocol.
7. **Predicates as Spelke-axis projections.** Each predicate is a projection of state onto a Spelke axis or a pairwise joint.

---

## 9. What's novel vs the field (honest scoping)

- Salience as central organ (subsumes Latent Inference); not seen in ARC-AGI-3 attempts we know of
- Bayesian symbolic WM (vs neural ICM); unusual choice
- Fact-space Critic library (vs single objective function); honest as disjunction-of-motifs
- Commit-and-monitor with hot/warm/cold paths in an online no-training setting; not seen
- Cross-level Ledger via hierarchical Beta in online no-training; not seen
- Multi-granularity attention-modulated perception; novel here

What's well-trodden:
- Motor cortex / DSL composition (ARC-1/2 solved this; we use environment-action DSL not grid-transformation)
- Object-centric perception (Slot Attention etc, but we use symbolic Spelke segmentation)
- Options framework / commit-and-monitor in HRL (Sutton 1999); we apply to online no-training

What's honestly absent:
- Affordance prior (humans bring it; no-training agents lack it). Slot exists for optional adapter-supplied prior.
- Full Latent Inference machinery for genuinely-invisible state (v0 schemas are minimal).

---

*biobrain is the brain-architecture argument that solving ARC-AGI-3 requires
online active inference, that active inference decomposes cleanly into eight
biological-RL-grounded components, and that those components can be
implemented symbolically (no neural nets, no training) with sample efficiency
the neural variants do not have. Whether this argument survives contact with
the full benchmark is what the remaining build phases will determine.*
