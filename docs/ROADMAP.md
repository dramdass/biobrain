# Roadmap

Build phases with measurement gates. Each phase has a go/no-go criterion
before the next phase fires.

---

## Status (current)

**v0.2 — published.** 8 components present and integrated. Commit-and-monitor
Planner. Generic-typed brain library. Pluggable Encoder. Cross-level
Ledger. All unit tests pass. Real-scenario validation pending (one-by-one
sweep planned).

---

## Phase 0 — Vocabulary alignment ✓ COMPLETE

The Critic and World Model must speak the same fact-space.
`emit_atomic_facts` emits the Spelke-axis predicate vocabulary; the WM
consumes it; fact-space Critic extractors (Compression, Noise, Symmetry,
ChangeDynamicsFactSpace) consume the same alphabet. Raw-cell legacy
extractors (StaticPatternRecurrence, ChangeDynamics) still exist but
can't participate in Simulator-mediated lookahead.

---

## Phase 1 — World Model accuracy validation ✓ COMPLETE

Per-step fact-prediction F1 measured on six games:

| Game | F1 | Verdict |
|---|---|---|
| lp85 | 0.94 | ✓ simulator viable |
| g50t | 0.86 | ✓ simulator viable |
| vc33 | 0.82 | ✓ simulator viable |
| cd82 | 0.73 | ✓ simulator viable |
| bp35 | 0.58 | ⚠ shallow only |
| r11l | 0.53 | ⚠ shallow only |

4/6 clear the F1 > 0.7 bar. The simulator was approved for build.

---

## Phase 2 — Simulator + 1-step lookahead ✓ COMPLETE (in v0.1)

For each candidate action: predict next-state fact set via WM; Critic
evaluates predicted facts; action posterior gets `current_distance −
predicted_distance` bonus.

Result (v0.1, N=15 × 200 steps):

| Game | Planner alone | + 1-step lookahead |
|---|---|---|
| vc33 | 33% | **60%** (+27pp) |
| lp85 | 80% | **93%** (+13pp) |
| cd82 | 0% | 0% |
| g50t | 0% | 0% |
| bp35 | 0% | 0% |

Lookahead is a real lift on games where the WM is accurate AND L3
identifies the right structural goal. Cold-start games (cd82) remain
floored — they need either deeper lookahead or richer representations.

---

## Phase 3 — Ledger machinery ✓ COMPLETE (transfer NOT YET validated)

Trajectory abstraction + hierarchical Beta per (program, level) +
cross-level promotion all built. The Ledger fills correctly on score
events (13 entries on lp85, 10 on vc33, 2 on r11l in 15 attempts ×
250 steps).

**The cross-level transfer mechanism is NOT YET exercised** because
brains rarely reach Level 2 within the action budget. vc33 is the one
public game that reaches Level 2 reliably — Phase 7 will validate
transfer there.

---

## Phase 4 — Critic primitive expansion ✓ COMPLETE

Six extractors built:
- **Compression** (fact-count + color-palette; dominant motif)
- **Noise** (size-bucket reduction)
- **Symmetry** (mirror-pair via `entity_color_quadrant`)
- **ChangeDynamicsFactSpace** (canvas-vs-target via per-quadrant color-set)
- **StaticPatternRecurrence** (legacy, same-shape match)
- **ChangeDynamics** (legacy, raw-cell histogram)

Coverage: 25/25 public games match at least one extractor. **Coverage ≠
correctness** — only 4/25 have verified goal-correctness via scoring;
the other 21 have detected goal candidates whose correctness is
unverified.

---

## Phase 5 — In progress

### Phase 5a — cd82 representational experiment (NEXT)

The critique predicts MCTS will not crack cd82 because the bottleneck is
representational (armed_color is invisible to coarse fact-space), not
search-depth. Decisive experiment: **add a hand-coded `armed_color`
latent to cd82's fact-space, re-measure World Model F1.**

Predicted outcome:
- If F1 jumps from 0.73 → 0.85+: bottleneck is representational. Build
  Latent Inference machinery (Salience schema-fallback).
- If F1 doesn't jump: bottleneck is deeper. Diagnose further before
  committing to MCTS or LI.

### Phase 5b — Outcome-dependent

If 5a confirms representational bottleneck:
- Build full Salience schema-fallback machinery (residual-clustering
  hypothesizer, schema library, automatic instantiation)
- Validate by automatically rediscovering the `armed_color` latent the
  hand-coded experiment proved was the missing piece

If 5a doesn't confirm: pause and re-think.

---

## Phase 6 — MCTS over DSL (deferred)

Multi-step rollouts via Simulator. UCB or Thompson at tree nodes.
Gated on Phase 5a outcome (no point committing to deeper search if the
bottleneck is representational).

---

## Phase 7 — Ledger transfer validation on vc33

vc33 reaches Level 2 under residual/lookahead configurations. This is
the one game where cross-level program promotion can be exercised.

Run BioBrainV2 on vc33 with Ledger enabled vs Ledger disabled. Measure:
- Does Level-1-scored Program get promoted on Level 2 entry?
- Does the promoted Program score on Level 2 (cross-level transfer)?
- Does max-level-reached increase?

Cheapest probe with the highest information value about the Ledger
phase being real.

---

## Phase 8 — Full 25-game competence envelope

Run BioBrainV2 on all 25 public games. Document with Wilson confidence
intervals. This is the v0.2 measurement headline.

Compares against:
- Substrate-only baseline (4/25 expected: lp85, vc33, r11l, lf52)
- v0.1 Planner without commit-and-monitor (to confirm refactor preserves
  lookahead lift)

---

## Open questions

These determine whether biobrain reaches human-level or hits a structural ceiling.

1. **Is fact-space deep enough?** Predicate vocabulary covers ~30
   templates with 256 possible joints. It doesn't represent: shape
   topology, 2D transformations of entities, fine-grained relational
   facts. Whether failed games need vocabulary extension or just better
   planning is open.

2. **Will MCTS escape the single-step ceiling?** Depth-2 rollouts
   compound to ≈ F1² joint accuracy. cd82 at F1=0.73 → 0.53 at depth 2.
   Phase 5a will likely show cd82's bottleneck is representational
   (missing modal-state predicate), not depth-related.

3. **Does the Ledger generalize across levels?** Untested. Phase 7.

4. **Where does compression-as-objective break?** Some games may score
   on *anti-compression* moves (grow-from-seed). The Critic has no
   built-in opposite. If we find a class where the objective sign is
   wrong, we need a learnable critic-direction selector.

5. **Affordance prior — accept gap or relax covenant?** Slot exists.
   Strict default. Adapter-supplied priors available if we ever decide
   covenant-relaxation is worth it. Decision deferred.
