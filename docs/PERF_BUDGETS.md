# Performance Budgets — Architectural Math

What each biobrain component should be doing within ARC-AGI-3's
constraints. Numbers are order-of-magnitude estimates derived from
the architecture, not measured (yet). Real numbers will replace
estimates as we instrument.

---

## ARC-AGI-3 constraints (the env's side of the deal)

| Constraint | Value | Source |
|---|---|---|
| Grid size | 64 × 64 = 4096 cells | spec |
| Color palette | 16 colors | spec |
| Action space | 7 IDs (5 keys, 1 click, 1 undo) | spec |
| Click granularity | 64 × 64 (any cell) | spec |
| Per-attempt action budget | ~100-500 actions typical | empirical |
| Per-action wall budget | unbounded in OFFLINE; ~30s in ONLINE | spec |
| Levels per game | 1-10+ typical | empirical |
| Scoring | RHAE — fewer actions to score = higher | spec |
| Public game count | 25 | spec |
| Held-out count | ~hundreds | spec |

The agent does NOT have:
- Persistent memory across games (covenant)
- Pretraining / training data of any kind (covenant)
- Affordance prior brought from outside the game (covenant — relaxable via adapter slot)

---

## Per-component time/space budgets

For each component, we compute: typical per-call cost, per-attempt
total, and memory footprint. "Typical" = mid-attempt state with
~20 entities, ~50 active facts, ~100 substrate signatures.

### 1. Motor Cortex (DSL)

**Per-call:** O(1). Program construction is dict-of-functions. Step
function evaluates `state → (sig, continuation)` in constant time
for atoms and combinators.

**Memory:** Library of templates is small (~10s of programs). Each
program is a small closure. <1KB total.

**Per-attempt:** N programs constructed × O(1) = negligible.

### 2. Perception

**Per-call (perceive):** O(grid × n_entities) ≈ 4096 × 20 = 80K cell
operations for flood-fill. ~5-10ms in Python.

**With multi-granularity:** Coarse default same cost. Fine attention
adds O(|attention_cells| × 4) for neighbor checks. Bounded to ~32
attended cells → 128 extra ops. Negligible overhead.

**Memory:** State + entities per observation. ~10-50KB per state
depending on entity count. Not kept long-term.

**Per-attempt:** N steps × 10ms = 200 actions × 10ms = 2 seconds total.

### 3. Salience

**Per-observe call:** O(n_recent_actions) for residual signature
compute. RESIDUAL_WINDOW=30 actions × ~5 ops = 150 ops. Schema
matching: 3 schemas × ~5 features = 15 ops. Total: ~microseconds.

**Memory:**
- Banked surprises: capped at 200 entries × ~200 bytes = ~40KB
- Modeled variables: ~30 template names × bytes = ~1KB
- Fine-attention queue: ~32 cells × bytes = ~256B
- Affordance posterior: ~10 action kinds × 32 bytes = ~320B
- Latent schemas instantiated: ≤4 × small = <1KB
- Total: ~45KB

**Per-attempt:** N observes × microseconds = negligible time cost.

### 4. Curiosity (Bayesian WM)

**Per-call (predict):** O(facts_at_context). Per-context fact count
~10-50 (Spelke-axis vocabulary). Dict lookup is O(1) per fact.
~1-2ms.

**Per-call (observe):** Same — update Beta for each fact in
union(before, after). Predicate emission is O(n_entities). ~5-10ms.

**Per-call (compute_surprise):** Same as predict + comparison.
~2-3ms.

**Memory:** Per-(fact, context) Beta posterior.
- Contexts: ~100-200 unique `(action_kind, target_color, level)` tuples.
- Facts per context: ~20-50.
- Total entries: ~5000.
- Each entry: 2 floats = 16 bytes.
- Total: ~80KB.

**Per-attempt:** 200 actions × ~5ms observe = 1 second.

### 5. Critic

**Per-call (L3.detect):** Sums across all 6 extractors.
- StaticPatternRecurrence: O(R²) region pairs, R ≤ 5. ~25 ops × cell-compare. ~5ms.
- ChangeDynamics: O(R²) with histogram compute. ~5-10ms.
- ChangeDynamicsFactSpace: O(Q² = 16²=256) quadrant pairs. Cheaper since fact-set ops. ~3ms.
- Compression: O(1) — count facts. <1ms.
- Noise: O(1). <1ms.
- Symmetry: O(8 mirror pairs × set ops). <1ms.
- Total: ~10-20ms per evaluate.

**Memory:** TransitionHistory per-cell change rate grid: 64×64 × float = 16KB.

**Per-attempt:** Critic runs only on cold-path decisions. With
commit-and-monitor saving 90% of steps as hot, Critic runs ~20-50
times per attempt × 15ms = 300-750ms.

### 6. Simulator

**Per-call (simulate_one):** Same as WM.predict. ~1-2ms.

**Per-attempt:** Cold-path candidates × simulate_one. 18 candidates ×
2ms × 50 cold calls = 1.8 seconds.

**Memory:** None — stateless wrapper over WM.

### 7. Ledger

**Per-observe:** O(K=5) rolling buffer update. O(1) dict insert on
score event. Per-(program, level) update O(1). <1ms.

**Per-call (promote_at_level):** O(n_entries) ≈ 5-20 entries. <1ms.

**Memory:**
- Per entry: ~100-500 bytes (program closure + Beta dicts).
- Total entries: bounded by score events per game, typically <20.
- Total: ~5-10KB.

### 8. Planner (commit-and-monitor)

**HOT path per-call:**
- Step continuation: O(1).
- WM prediction check: ~2ms.
- Total: ~3-5ms.

**COLD path per-call:**
- L3 detect: ~15ms (Critic).
- promote_at_level: <1ms (Ledger).
- For each candidate (~18):
  - Substrate Thompson sample: ~10μs (rng.betavariate).
  - Simulate_one: ~2ms (WM.predict).
  - Critic distance compute: ~1ms.
- 18 × 3ms = ~55ms.
- Total: ~70-80ms per cold call.

**WARM path per-observe:** ~10-15ms total across all components.

**Memory:**
- Substrate posterior: ~100 signatures × 16 bytes = 1.6KB.
- In-flight Program: tiny.

---

## Aggregate budget — per attempt

**With commit-and-monitor (the v0.2 ideal):**

| Phase | Per call | Count | Total |
|---|---|---|---|
| Hot path | 5ms | ~180 (90% of steps) | 900ms |
| Cold path | 80ms | ~20 (10% of steps) | 1600ms |
| Warm path (observe) | 15ms | 200 | 3000ms |
| **Total wall** | | | **~5.5 sec/attempt** |

**Without commit-and-monitor (cold every step):**

| Phase | Per call | Count | Total |
|---|---|---|---|
| Cold every step | 80ms | 200 | 16,000ms |
| Warm (observe) | 15ms | 200 | 3000ms |
| **Total wall** | | | **~19 sec/attempt** |

**Speedup from commit-and-monitor:** ~3.5x. Real if violation detection
is calibrated — over-trigger and you're back to per-step.

---

## Aggregate budget — per game

Typical game: 10-25 attempts × 5.5 sec/attempt = **1-2 minutes per game**.
25 public games × 2 min = ~50 minutes for a full sweep.

---

## Aggregate memory — per game brain instance

| Component | Size |
|---|---|
| Curiosity WM | 80KB |
| Salience | 45KB |
| Critic TransitionHistory | 16KB |
| Ledger | 10KB |
| Substrate (Planner) | 2KB |
| Misc state | 5KB |
| **Total** | **~160KB/game** |

Across 25 games (sequential, not concurrent): 160KB peak. Memory is
trivially cheap; this is a compute-bound architecture.

---

## Performance optimization priorities

Order by where time is actually spent:

1. **WM predict / observe** (40% of warm-path time). Optimization
   targets:
   - Cache `emit_atomic_facts` output for the same state (states repeat
     within attempts).
   - Vectorize Beta updates if multi-fact updates become bottleneck.

2. **Cold-path candidate ranking** (30% of time when not hot). 18
   candidates × per-candidate work. Optimization targets:
   - Filter candidates by Salience priority before full evaluation
     (top-K candidates instead of all).
   - Memoize Critic distance for repeated state visits.

3. **Critic L3.detect** (20% of cold time). Optimization targets:
   - Skip extractors that returned empty last call if state hasn't
     materially changed (state.grid_hash matches).
   - Parallelize extractors (they're independent).

4. **Perception flood-fill** (10% of warm-path time). Already cheap.
   Optimize only if profiling shows hot.

---

## What the math tells us about architectural choices

- **Commit-and-monitor is structurally worth it.** ~3.5x speedup if
  violation detection is calibrated. Not a marginal optimization — it's
  why the agent can afford to run lookahead/Critic at decision time
  without exploding.

- **The brain is compute-bound, not memory-bound.** 160KB per game is
  trivial. The optimization surface is CPU time per action, not RAM.

- **The fact-space representation is the right scale.** ~50 facts per
  state, ~5000 (fact, context) entries total. Larger predicate
  vocabularies (e.g., relational pairs) would explode this — keeping
  the alphabet bounded was the right architectural commitment.

- **Cold path is the optimization target.** Hot path is already cheap
  (5ms). Cold path is 80ms × 20 calls = 1.6 sec. Cutting cold by 2x
  (e.g., via top-K candidate filtering) saves ~800ms per attempt = 15%
  reduction. Cutting warm by 2x saves ~1.5 sec but warm is structurally
  required learning.

- **MCTS would multiply cold-path cost.** Depth-2 rollouts × 18
  candidates × per-candidate simulator call = ~30ms per candidate × 18
  = 540ms per cold call. 7x more expensive than current 1-step
  lookahead. Phase 5+ build requires either better simulator caching or
  more aggressive Salience filtering.

- **Salience curation matters more than it might look.** If Salience
  filters facts down by 50%, WM observe is 50% faster (because per-fact
  Beta updates are the inner loop). Salience pays for itself in WM cost
  alone, before you count its other contributions.

---

## Open questions (math-side)

1. **What's the actual ONLINE mode per-step time budget?** OFFLINE is
   unbounded; ONLINE might enforce ~30s but I haven't verified. If
   ONLINE enforces a tight budget, MCTS depth >2 becomes problematic.

2. **How does the Encoder's fine-attention contribute to per-step
   cost?** Currently we estimate it as negligible because attention
   cells are bounded. But if Salience over-triggers, the attention queue
   could grow and each observe pays for finer perception over many cells.

3. **What's the real Ledger growth rate?** We estimate ~20 entries per
   game; if games have many score events and each creates a new entry
   (no dedup beyond program_id), this could grow. Need to instrument.

4. **At what scale does WM context-partition lose efficiency?** If a
   game introduces many novel `(action_kind, target_color, level)`
   contexts (e.g., one per level × per color), the WM's per-context
   posterior count grows. At what count does cache miss / dict slowdown
   start mattering?

---

These numbers are starting points. As we run probes, the real numbers
go into `docs/RESULTS.md` and the estimates here get refined or
falsified. The architecture-level math is the planning tool; the
measurements are the ground truth.
