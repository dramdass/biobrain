# Components — deep dive

Per-component breakdown of biobrain's 8 cortical regions. For each:
**role + responsibilities** (what it does) → **bio analog** (with honesty
about literal vs metaphorical) → **why it's needed** (what fails without)
→ **ML research analogs** (the lineage) → **ARC-AGI attempt history**
(what's been tried, what's novel here).

---

## 1. Motor Cortex (DSL)

**Role.** Action vocabulary + composition. The agent's "muscles" plus the
grammar for chaining them.

**Responsibilities.**
- Atomic actions: `click_on_color(c)`, `key(k)`, `spacebar()`, `noop()`. These map to ARC-AGI-3's exposed action space — *not* grid transformations.
- Combinators: `SEQ`, `IF`, `REPEAT`, `WHILE_NOT`. Compose atoms into Programs.
- Execution semantics: a Program is a `State → (Action, Continuation)` function.

**Bio analogy.** Primary motor cortex (M1) converts intent to motor commands; premotor cortex sequences them; supplementary motor area handles internally-generated action selection. The mapping is honest at the *interface* level (vocabulary + sequencing) but artificial at the *implementation* level — biology uses population codes and central pattern generators, not IF/SEQ combinators. Functional analogy.

**Why needed.** Without composition, every action is atomic and Programs aren't first-class. The Ledger can't store "click then paint" as a unit. The Planner can't commit to multi-step Programs. The phenomenology's "hypotheses-as-programs-over-curated-variables" requires Programs as the unit of commitment.

**ML research analogs.**
- **DSL-based program synthesis**: DreamCoder (Ellis et al 2020), Stitch (Bowers et al 2023)
- **Hierarchical RL / Options framework**: Sutton 1999, Option-Critic (Bacon et al 2017)
- **Compositional generalization**: Lake 2023 (MLC)

**ARC-AGI attempt history.**
- ARC-AGI-1/2: DSL is *the* dominant approach (Hodel, MindsAI, ICEMonkey — all DSLs over grid transformations).
- ARC-AGI-3: less clear. Our DSL is at the *environment-action* level (click/key), not the grid-transformation level. The game implements grid transformations internally and the agent reverse-engineers them.

What's novel: environment-action DSL with combinators that the Ledger stores and the Planner commits to.

---

## 2. Perception

**Role.** Extract Spelke-grounded entities from raw grid. Multi-granularity — coarse default, finer on Salience trigger.

**Responsibilities.**
- Connected-component segmentation by color (one contiguous same-color blob = one entity)
- Multi-granularity: default coarse; when Salience requests finer attention at a cell/region, emit sub-entity features (tile highlights, edge patterns)
- Salience-mask-aware: incorporate salience grid into entity prioritization
- Predicate emission: from entities + their attributes, emit the fact alphabet (Spelke-axis projections + joints)

**Bio analogy.** Visual cortex (V1→V4) does feature extraction at progressively higher levels. Ventral "what" stream is object recognition. Multi-granularity = the parvo/magno pathway split. Attention modulation from frontal eye fields and pulvinar. Honest at *function* level — biological vision IS multi-granular and IS attention-modulated.

**Why needed.** Raw pixel grids aren't usable units. The Spelke prior (objects, contiguous blobs) is innate perceptual scaffolding humans bring. Multi-granularity matters because modal state has *observable tells* that coarse perception throws away (per the phenomenology) — without finer attention triggered by salience, cd82's armed-color tell is invisible.

**ML research analogs.**
- **Object-centric representation**: Slot Attention (Locatello et al 2020), Neural-Symbolic Concept Learner (Mao 2019)
- **Spelke priors**: Spelke 1990 (core knowledge of object), Spelke + Kinzler 2007
- **Visual attention**: Mnih et al 2014, Bahdanau 2015, the Transformer line
- **Multi-scale vision**: Feature Pyramid Networks (Lin 2017)

**ARC-AGI attempt history.**
- ARC-1/2: Connected-component object segmentation is standard.
- ARC-3: Object extraction copied from ARC-1/2 approaches, usually coarser than needed because dynamics expose sub-entity features. We're not aware of an ARC-3 entry doing multi-granularity perception.

What's novel: salience-triggered fine-grained perception, where granularity adapts to prediction-error feedback.

---

## 3. Salience (the central organ — subsumes Latent Inference)

**Role.** Curate the small variable set the brain models; trigger fine perception at prediction failures; bank unexplained surprises; propose new predicate templates.

**Responsibilities.**
- **Curate**: maintain the active modeled-variable set; filter `emit_atomic_facts` output to active variables
- **Attend**: on coarse-prediction failure, flag the cell/context for finer perception
- **Bank**: log salient-but-unexplained observations awaiting hypothesis
- **Propose**: when a banked surprise gets a candidate explanation, register a new predicate template
- **Affordance posterior**: track per-action-class informativeness; consumed by Planner

**Bio analogy.** Multi-region. Salience network (anterior insula + dorsal ACC) does cognitive salience. Pulvinar gates visual attention. Hippocampal dentate gyrus does novelty detection + pattern separation (the "banking" function). Lateral PFC manages working memory selection (the "curation" function). Honest at *function* level — salience IS multiple coordinated subsystems — but obscures that biology distributes these across regions while our component centralizes them.

**Why needed.** Without curation, the combinatorial fact space explodes (every Spelke joint always in scope; the conjunction-synth wall is real). Without attention-triggering, we can't find observable modal-state tells (cd82's armed-color highlight). Without surprise-banking, we can't backpropagate "the lit tiles meant active all along" when an explanation eventually arrives. Without proposing new predicates, the predicate vocabulary stays fixed.

**ML research analogs.**
- **Attention mechanisms**: Transformer attention (Vaswani 2017); learned attention masks (Mnih et al)
- **Active learning / acquisition functions**: Settles 2009
- **Predictive State Representations (PSRs)**: Singh + James 2004 — explicit construction of latent state to make predictions sufficient. Closest formal analog.
- **Schemas + curriculum**: Lake et al 2017 (Building Machines That Learn Like People)
- **Bayesian surprise**: Itti + Baldi 2009

**ARC-AGI attempt history.**
- ARC-1/2: little need for salience because puzzles are static.
- ARC-3: this is the architectural gap. Most agents have implicit salience (Bayesian surprise drives count-based exploration) but not as a first-class organ. Cross-level program promotion exists in some submissions, but the curation + finer-attention + predicate-proposal combination is novel here.

What's novel: Salience as central organ, not a perception utility. The phenomenology insight (humans don't separately "do salience" and "infer latents" — they do one integrated thing) is the strongest evidence for this architectural shape.

---

## 4. Curiosity (Bayesian World Model + ICM)

**Role.** Maintain a Bayesian world model; provide intrinsic reward signal via signed prediction error; drive exploration via surprise.

**Responsibilities.**
- Per-`(fact, context)` Beta posteriors of `P(fact_next | context)`. Context = `(action_kind, target_color, level)`, optionally extended by Salience-proposed latents.
- Predict next-state fact set given `(state, action)`.
- Compute signed surprise per transition. Positive = unexpected (interesting). Negative = predictable (boring).
- Update posteriors every transition (the warm path).
- Boredom mechanic: as posteriors sharpen, surprise → 0 automatically. Schmidhuber's compression-progress in its self-extinguishing form.

**Bio analogy.** Dopaminergic system (VTA, substantia nigra pars compacta) signals reward prediction error — the literal RPE signal. Hippocampus does episodic learning of state transitions. Honest at *function* level (dopamine literally signals prediction error in mammals). The biology is more sophisticated (timing, eligibility traces) but the core insight — *intrinsic reward = prediction error* — is the same.

**Why needed.** Without a world model, no forward simulation, no MCTS, no Ledger validation. Without surprise, no exploration drive (no extrinsic reward in cold-start games). Without boredom, exploration never converges. Phase 1 measurement confirmed F1 ≥ 0.7 on 4/6 games — WM is accurate enough to anchor everything else.

**ML research analogs.**
- **ICM (Intrinsic Curiosity Module)**: Pathak et al 2017 — canonical formulation
- **RND**: Burda et al 2018 — alternative novelty signal
- **Active inference**: Friston's free energy principle
- **Bayesian model learning**: Goodman + Tenenbaum's probabilistic programming
- **Curiosity-driven exploration**: Schmidhuber 1991 (the original compression-progress paper)

**ARC-AGI attempt history.**
- ARC-1/2: World model not strictly needed for static puzzles.
- ARC-3: World models matter because the env is interactive. Most agents use *implicit* world models (RL on transitions, no explicit predictor). Some explicitly model dynamics (DeepMind). Our Bayesian *symbolic* WM is unusual.

What's novel: symbolic ICM with per-`(fact, context)` Beta posteriors. No neural networks, no training, just online Bayesian updates with self-extinguishing surprise.

---

## 5. Critic (multi-extractor goal-setting library)

**Role.** Identify "win-state aesthetics" without supervision. Emit ProtoGoals.

**Responsibilities.**
- Run multiple extractors on current state:
  - **Compression** — fewer facts is more win-like (dominant motif)
  - **Noise** — fewer tiny/small entities (entropy reduction)
  - **Symmetry** — mirror-pair alignment via `entity_color_quadrant`
  - **ChangeDynamicsFactSpace** — dynamic-canvas-matches-static-target (reference-matching motif, NOT compression)
  - **StaticPatternRecurrence** — same-shape region match
- Weight goals by DL-saving estimate
- Return goal list for Planner consumption

**Bio analogy.** Orbitofrontal cortex (OFC) — value computation, reward prediction. Ventromedial PFC — abstract value. Amygdala — affective tagging. The Critic *is* the brain's value-prediction system. Honest at *function* level.

**Why needed.** ARC-AGI-3 has no extrinsic reward until winning. Without a Critic, exploration is undirected. Per the critique: this is a *library* of goal-detectors with compression as dominant motif, not a single objective function.

**ML research analogs.**
- **Empowerment**: Klyubin + Polani + Nehaniv 2005
- **Compression-progress**: Schmidhuber 1991/2010
- **Successor features**: Dayan 1993, Barreto 2017
- **CompressARC**: Liao 2025 — solves ARC-AGI offline puzzles via compression
- **Universal Value Function Approximators**: Schaul 2015

**ARC-AGI attempt history.**
- ARC-1/2: Critic trivial — match the target output.
- ARC-3: Hardest open problem. Various intrinsic motivation works (count-based exploration, surprise bonuses). CompressARC proves compression-objective works on offline ARC.

What's novel: library of pluggable Critic extractors with explicit acknowledgment that compression is dominant motif, not universal.

---

## 6. Simulator (forward queries via WM)

**Role.** Mental sandbox. Predict next-state facts given `(state, action)` without executing.

**Responsibilities.**
- `simulate_one(state, action) → set[fact]` — deterministic projection
- `simulate_one_sampled(state, action, rng) → set[fact]` — Bernoulli draw
- Multi-step rollout (deferred to MCTS builds)

**Bio analogy.** Dorsolateral PFC handles working memory simulation. Hippocampal-PFC dialog supports episodic future-thinking (Schacter + Addis 2007). Cerebellum implements forward models for motor prediction (Wolpert + Miall + Kawato 1998). Honest — biology DOES separate "learning the model" (Curiosity / hippocampus) from "using it for simulation" (Simulator / PFC).

**Why needed.** Lookahead bonus in Planner. Validating Ledger programs before commit. Salience uses it to check whether a proposed new predicate would improve WM prediction.

**ML research analogs.**
- **Model-based RL**: Sutton + Barto, Levine's deep MBRL line
- **World Models**: Ha + Schmidhuber 2018
- **Dreamer**: Hafner et al
- **MuZero**: Schrittwieser 2020
- **MCTS**: Coulom 2006, UCT (Kocsis + Szepesvári 2006), AlphaGo (Silver 2016)

**ARC-AGI attempt history.**
- ARC-1/2: Some program-synthesis approaches do simulate (apply candidate transforms to inputs).
- ARC-3: Most agents lack a reliable enough world model to simulate. Phase 1 measurement is what unlocks this for biobrain.

What's novel: fact-space simulation rather than pixel-space. Simulator inherits WM's bounded-error structure; Critic consumes its output directly without rendering.

---

## 7. Ledger (per-game program memory)

**Role.** Bank successful action sequences as parameterized DSL Programs. Cross-level promotion via hierarchical Beta.

**Responsibilities.**
- On score event: abstract last K actions into a Program (entity-color-anchored parameterization)
- Hierarchical Beta posterior per `(program_id, level)`
- `promote_at_level(L)`: return programs whose prior-level confidence ≥ threshold
- `register_failure(pid, L)`: note a Program tried at this level that didn't score
- Wipe at `reset_game`; persist across `reset_attempt`

**Bio analogy.** Hippocampus + medial temporal lobe — episodic memory consolidation. Procedural memory (basal ganglia, cerebellum) for compiled skills. Medial PFC (schema cortex per Tse et al 2007) for skill abstraction. Honest at *function* level — biology DOES consolidate one-shot experiences into reusable skills via overnight replay.

**Why needed.** Without it, the brain re-learns each level from scratch. The phenomenology: "background reason if my hypothesis should be generalized" — that IS the Ledger.

**ML research analogs.**
- **Skill libraries / Options framework**: Sutton + extensions (Bacon, Konidaris)
- **DreamCoder**: Ellis et al 2020 — closest to what the Ledger does
- **Episodic control**: Botvinick + Pritzel 2017
- **Hindsight experience replay**: Andrychowicz et al 2017
- **Hierarchical Bayesian models**: Tenenbaum + Kemp et al

**ARC-AGI attempt history.**
- ARC-1/2: Program-synthesis accumulates solutions within a puzzle.
- ARC-3: Cross-level transfer barely explored. Most agents treat levels independently.

What's novel: trajectory abstraction + hierarchical Beta + per-(program, level) posterior in an online no-training setting.

---

## 8. Planner (commit-and-monitor)

**Role.** Pick the next Program. Execute open-loop. Monitor for violations. Re-engage on violation.

**Responsibilities.**
- **Hot path** (every step): step Program continuation + WM prediction check
- **Cold path** (violation OR program-end): full reasoning — Thompson + Critic + Lookahead via Simulator + Ledger promoted programs + Salience affordance
- **Warm path** (every transition, in observe()): substrate posterior updates from surprise

**Bio analogy.** Dorsolateral PFC (executive function), anterior cingulate cortex (ACC) for conflict monitoring + violation detection — *exact* biological correlate of "prediction error triggers re-planning." Basal ganglia gates action selection. Honest at *function* level (ACC literally signals expectation violation).

**Why needed.** The hot/warm/cold path separation is the budget-efficiency win the phenomenology promises. Without commit-and-monitor, every step costs full reasoning.

**ML research analogs.**
- **Options framework / Semi-MDPs**: Sutton 1999
- **Option-Critic**: Bacon et al 2017
- **Predictive coding control**: Clark 2013, Friston's active inference
- **Hierarchical RL**: Dayan + Hinton 1993 (Feudal RL), Vezhnevets 2017 (FuN)
- **MCTS with policy networks**: AlphaGo lineage

**ARC-AGI attempt history.**
- ARC-1/2: Planning is "search for the right transform program." Not directly analogous.
- ARC-3: Most agents are per-step. Event-driven commit-and-monitor is novel.

What's novel: three-path separation (hot/warm/cold) where each path does the work appropriate at its temporal scale, justified by the phenomenology and the budget constraint.

---

## Cross-component summary

**What's load-bearing across all eight.** Compression-progress drive (Curiosity), symbolic representation (Perception + Spelke-axis predicates), event-driven control loop (Planner). Drop any of them and the architecture falls apart.

**What's novel vs the ARC-AGI field.**
- Salience as central organ (subsumes Latent Inference)
- Bayesian symbolic WM (vs neural ICM)
- Fact-space Critic library (vs single objective)
- Commit-and-monitor with hot/warm/cold paths in online no-training setting
- Cross-level Ledger via hierarchical Beta in online no-training
- Multi-granularity attention-modulated perception

**What's well-trodden.**
- Motor cortex / DSL composition (ARC-1/2 solved this; we use environment-action DSL)
- Object-centric perception (Slot Attention etc, but we use symbolic Spelke segmentation)
- Options framework / commit-and-monitor in HRL (Sutton 1999); we apply to online no-training

**The honest residual gap: affordance prior.** Humans bring "buttons are clickable, spacebars are special" from outside the game. A strictly no-training agent does not have this. We name the gap honestly and provide an optional adapter slot for it if/when we relax covenant.
