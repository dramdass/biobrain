# Within-Game Search — biobrain v0.3 Design Spec

**Date:** 2026-06-01
**Status:** approved via brainstorm; ready for implementation planning
**Scope:** single implementation plan (~800-1200 LOC)

---

## 0. Why this build

biobrain v0.2 is at the edge of what symbolic Bayesian + 1-step lookahead reach.
Empirical evidence:

- Probe **falsified** the cd82 modal-state hypothesis: coarse encoder already
  distinguishes armed-color states. Perception is NOT the bottleneck.
- Persistent-brain Ledger probe on vc33: 100% scoring at Level 1, never reached
  Level 2 across 20 attempts. Architecture correct; transfer mechanism wasn't
  unlocked.
- Per-game data math: Level 2+ has near-zero training data. A learned WM
  cannot be the level-transfer mechanism. Macros + within-game search must do
  that work.

The God Mode architectural vision identified within-game search over the env's
deterministic + resettable + hash-emitting structure as the depth keystone.
This spec converts that vision into a buildable design.

---

## 1. Locked architectural decisions

From the brainstorm interview, in order:

| # | Decision | Choice |
|---|---|---|
| 1 | Node equivalence | (d) **multi-level**: `grid_hash` for navigation, **role-fingerprint** for cross-level transfer |
| 2 | Roles | 10-role Spelke-grounded catalogue (incl. `unknown`), discovered via causal signature in Salience |
| 3 | Search strategy | (b) **curiosity-guided** frontier expansion |
| 4 | Action-budget management | (c) **phase-by-evidence**: maximize `epistemic + pragmatic + empowerment` EV per cold-path call. Equal weights for v0; (d) UCB is v1 backlog; weights are RL-TODO |
| 5 | Architecture integration | (c) **extend Planner** with `SearchGraph` submodule; `RoleFingerprintIndex` lives in Salience |
| 6 | Transferable units | (c) **subgoals as units**, NOT full trajectories |
| 7 | Subgoal detection | (α) **fingerprint delta** as primary detector; (β) **Critic-distance delta** as confirmation channel for promotion confidence |

---

## 2. Component changes — diff against v0.2

Component count stays at 8. Two components gain submodules.

### 2.1 Planner — adds `SearchGraph` submodule

The cold path becomes search-augmented (not search-replaced). The existing
Thompson + Critic + Simulator + Ledger machinery becomes inputs to the EV
computation rather than competing decision rules.

**`SearchGraph` data structure** (lives inside CommitMonitorPlanner):

- `nodes: dict[grid_hash → NodeMetadata]` — NodeMetadata: visit count, last attempt seen, scoring flag, dead-end flag
- `edges: dict[(parent_hash, action_key) → child_hash]` — adjacency list
- `unexpanded_frontier: set[(node_hash, candidate_action)]` — actions not yet tried from this node
- `scoring_paths: list[ActionSequence]` — recorded sequences that produced score events

**Lifecycle:** wipes on `reset_game`; **persists across reset_attempt** (this is
the keystone — the graph accumulates over attempts).

### 2.2 Salience — adds role machinery + fingerprint index

Salience already curates variables, banks surprises, manages fine-attention,
maintains the affordance posterior. New additions:

**Per-entity causal counters** (online, updated in observe()):
- `clicked_caused_self_change` — entity changed under click on itself
- `clicked_caused_other_change` — clicking entity changed something elsewhere
- `clicked_caused_global_change` — clicking entity changed >K cells (mode flip)
- `translated_under_key_count` — entity centroid shifted under key actions
- `persistence` — fraction of observed transitions where entity remains present
- `referenced_by_distance_goals` — appears in active goal's `relevant_cells`

**Per-entity role posterior** over the 10-role catalogue (see §3). After K
observations (start K=5), assign the highest-likelihood role. Below K, tag is
`unknown`. Roles are refinable as more data arrives.

**Fingerprint computation** — given a state, produce three fingerprints:
- `F_tight`: multiset of `(role, color, quadrant)` tuples over the state's entities
- `F_mid`: multiset of `(role, color)` (color-anchored identity)
- `F_loose`: multiset of `role` only (most permissive)

**`RoleFingerprintIndex`** data structure:
- `dict[fingerprint → list[Subgoal]]` — three sub-indexes, one per granularity
- Subgoal record: `(start_fingerprint, action_subsequence, end_fingerprint, critic_validated: bool, source_level, source_attempt_id)`

**Lifecycle:** wipes on `reset_game`; persists across `reset_attempt` and
`on_level_change`.

---

## 3. The 10-role catalogue

Each role is a Spelke-grounded functional characterization, identified by
distinctive causal signatures observable through transitions.

| # | Role | Signature (likelihood maximized when…) |
|---|---|---|
| 1 | **Selector** | `clicked_caused_other_change` ≥ θ AND `clicked_caused_self_change` low |
| 2 | **Cursor / Agent** | `translated_under_key_count` is the dominant change observed |
| 3 | **Painter** | clicking E + subsequent transition produces new visible cells elsewhere (deferred causality) |
| 4 | **Target / Reference** | high `persistence` AND `referenced_by_distance_goals` |
| 5 | **Toggle** | clicking E flips E between exactly two distinguishable states |
| 6 | **Counter / Indicator** | E's appearance changes monotonically under some other action class |
| 7 | **Removable / Barrier** | E disappears after click; blocks cursor motion when present |
| 8 | **Container** | other entities accumulate inside E's region OR pass through |
| 9 | **Static / Decorative** | low change rate across all action classes; not referenced by other entities |
| 10 | **Unknown** | default tag until ≥K observations confirm a specific role |

Role assignment is a per-entity Bayesian likelihood comparison. Heuristics are
Spelke-grounded, not learned, not per-game-tuned. # RL-TODO: likelihood
parameters could be learned from per-game outcome correlations.

---

## 4. Data flow

### 4.1 observe(transition) — additions to v0.2 flow

After the existing v0.2 observe (Curiosity, Critic, Ledger, etc.):

1. **Salience.causal_counters.update(transition)** — increment counters for entities involved
2. **Salience.role_posteriors.refresh()** — for entities with `n_observations ≥ K`, update role assignments
3. **Compute `fingerprint(transition.after)`** at all 3 granularities via current role assignments
4. **Planner.SearchGraph.add_edge** `(transition.before.grid_hash, transition.action_key) → transition.after.grid_hash`. Add node if new.
5. **Subgoal detection:**
   - If `fingerprint_after.F_mid ≠ fingerprint_before.F_mid`: a subgoal was achieved.
   - Build `Subgoal(start_fp, action_subseq_since_last_subgoal, end_fp, validated=False, source_level, source_attempt_id)`.
   - If `Critic_distance(state_after) < Critic_distance(state_before)` for any active goal: set `validated=True`.
   - Index the subgoal under `start_fp` AND `end_fp` in `Salience.fingerprint_index` (at all 3 granularities).
6. **On score event**: mark the path's terminal node as scoring in SearchGraph; also propose the full trajectory as a Ledger macro (existing pipeline).

### 4.2 act(state, budget) — cold-path decision rule

```
candidates = encoder.candidate_actions(state)
fact_set, fingerprint = encoder.encode(state), Salience.fingerprint(state)
critic_goals = Critic.evaluate(state)
promoted_macros = Ledger.promote_at_level(state.level)
transferred_subgoals = Salience.fingerprint_index.lookup(fingerprint)  # all granularities

for each candidate action a:
    # Epistemic — expected information gain
    epistemic(a) =
        if (state.grid_hash, a) is unexpanded in SearchGraph: +HIGH_PRIOR
        else: curiosity from symbolic WM (expected surprise on predicted child)

    # Pragmatic — progress toward known goals/scoring
    pragmatic(a) =
        critic_distance_reduction(simulate_one(state, a))  # symbolic WM as pre-filter
        + transferred_subgoal_first_action_bonus(a, transferred_subgoals)
        + promoted_macro_first_action_bonus(a, promoted_macros)

    # Empowerment — control over reachable future, REAL graph
    empowerment(a) =
        |reachable_nodes_from(SearchGraph.children_of(state.grid_hash, a))| within depth K
        (counted over the actual graph; NO WM-imagined rollouts)

    EV(a) = epistemic(a) + pragmatic(a) + empowerment(a)   # equal weights for v0

choose argmax EV(a)
commit candidate as in-flight Program (1-step, OR extended if part of a macro)
```

**Hot path:** unchanged from v0.2 (step Program continuation).

**Warm path:** same as observe (above) — Salience does extra work each transition.

### 4.3 Lifecycle additions

| Boundary | Wipes | Persists |
|---|---|---|
| `reset_game` | **all** SearchGraph state, **all** fingerprint index, role counters/posteriors | nothing new |
| `reset_attempt` | nothing in SearchGraph or fingerprint index | both persist across attempts |
| `on_level_change` | nothing in SearchGraph or fingerprint index | both persist (these ARE the transfer machinery) |

---

## 5. Honest scope

### Unlocked by this build

- **Within-game graph mapping** — the brain accumulates a real reachable-state graph across attempts
- **Cross-level transfer via subgoals** — the user's "same color, different shape, same role" pattern works
- **Empowerment over REAL state space** — no WM imagination compounding errors
- **Curiosity-guided exploration** — principled frontier expansion ordering
- **Macro abstraction at subgoal granularity** — less fragile than full-trajectory macros
- **Role-tagged entity identity** — cursor / selector / target as functional roles, not visual recognition

### Not addressed (deferred)

- Differentiable / learned World Model — paused per math (insufficient data budget for Level 2+)
- Multi-step rollouts via WM — replaced by real-env graph expansion
- Affordance prior — still no cross-game knowledge (covenant respected)
- Adapter-supplied priors — slot remains; unused in v0.3
- Critic abstention pre-score — deferred; not strictly required for this build

---

## 6. Validation strategy

### 6.1 Per-component diagnostics (before end-to-end)

| Component | Diagnostic | Pass criterion |
|---|---|---|
| Causal counters | Click on cd82's dark selector triggers `clicked_caused_other_change` | After 1 observation |
| Role posterior | After 20 observations on cd82, dark-selector tags as `painter`/`selector`; target tags as `target`; framing as `static` | All correct after K observations |
| Fingerprint stability | Same state encoded twice → identical fingerprints (all 3 granularities) | Bit-exact equality |
| SearchGraph correctness | After N transitions, graph has ≤N edges (dedup); replay from root reaches recorded child | 100% replay accuracy |
| Subgoal detection | On a scored trajectory, ≥1 subgoal detected; each has start ≠ end fingerprint | ≥1 per scoring trajectory |
| Subgoal validation | Scored-trajectory subgoals have `validated=True` more often than random-walk subgoals | Statistical separation |
| Cross-level transfer | On Level 1 entry, fingerprint lookup surfaces ≥1 candidate subgoal from Level 0 | When applicable |

### 6.2 End-to-end (after build complete)

- **Headline:** does v0.3 score MORE OFTEN at Level 2+ than v0.2 on vc33, lp85?
- **Cold-start:** does v0.3 score on any of the 21 currently-floored games (especially cd82, bp35)?
- **Compute cost:** what fraction of cold-path is spent on search vs Thompson? Acceptable ranges TBD.

---

## 7. Implementation questions deferred to the plan

Not blocking the design; resolved in the writing-plans step:

- SearchGraph memory bound — K-node cap with LRU eviction, or per-game total cap?
- Empowerment depth K — start K=2; benchmark cost-vs-discrimination
- Role-discovery K threshold — start K=5
- Subgoal storage — all 3 granularities by default; profile cost
- First-attempt bootstrap — no special logic; empty graph → all `epistemic(a)` high uniformly → choice is random; emergent
- Terminal-edge handling — children-of-terminal tagged dead-end; `empowerment(a) = 0`
- Replay vs forward-step — within an attempt, no replay (env doesn't support state-injection); only `reset()` between attempts

---

## 8. Discipline anchors (carried from PRINCIPLES.md)

1. **Upstream-first debugging.** When a search measurement looks wrong, verify the graph extends correctly before suspecting EV.
2. **Principled derivation over hardcoded formulas.** No magic blends on EV terms. Equal weights are explicit; learning is RL-TODO.
3. **Single source of truth.** SearchGraph in Planner; fingerprint index in Salience; subgoals reference state-fingerprints (not duplicated state objects).
4. **Abstraction-level alignment.** Roles + fingerprints are the cross-level vocabulary; hash is within-level navigation. Don't mix.
5. **House-model lifecycle discipline.** SearchGraph + fingerprint index follow "persist across attempts, wipe on reset_game" alongside Ledger.
6. **Generalize-by-protocol.** New role types added by appending the catalogue + providing a likelihood heuristic. No subclassing.
7. **Subgoals as the abstraction unit.** Trajectories are not transferable; subgoals are. Always index at start AND end fingerprints.

---

## 9. Cross-references

- biobrain v0.2 design: `docs/DESIGN.md`
- Discipline principles: `docs/PRINCIPLES.md`
- Performance budgets: `docs/PERF_BUDGETS.md`
- Roadmap: `docs/ROADMAP.md`

*Within-game search adds the depth keystone the God Mode vision identified,
integrated cleanly into the existing 8-component spine, respecting the
covenant, with honest per-component diagnostics gating the build.*
