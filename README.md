# biobrain

**Active Inference Architecture for ARC-AGI-3.**

A bio-RL-grounded agent that wakes up in unfamiliar grid environments with no instructions, no rewards, no pretraining, and reasons its way to the puzzle's solution through active inference.

```
┌─────────────────────────────────────────────────┐
│              BIOBRAIN CORE (8 components)       │
│                                                  │
│   1. Motor Cortex   (DSL action vocabulary)      │
│   2. Perception     (Encoder + finer attention)  │
│   3. Salience       (central curating organ)     │
│   4. Curiosity      (Bayesian WM + ICM)          │
│   5. Critic         (library of goal-detectors)  │
│   6. Simulator      (forward queries via WM)     │
│   7. Ledger         (per-game program memory)    │
│   8. Planner        (commit-and-monitor)         │
└─────────────────────────────────────────────────┘
```

## What this is

ARC-AGI-3 is an interactive benchmark where agents are dropped into turn-based grid environments with no instructions, no stated goals, and no predefined rules. Frontier LLMs score under 1%. Humans score 100%. The gap is not pattern recognition; it's **online active inference** — the capacity to act in order to learn, learn in order to predict, predict in order to plan, and plan in order to act.

biobrain is an attempt to build that capacity from first principles, symbolically, with no pretraining and no neural networks. The architecture decomposes active inference into eight named components, each with both a biological-RL analog and a precise ML/RL term.

See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture spec.

## Status

- **v0.2** — current. 8 components present and integrated; commit-and-monitor Planner; generic-typed brain library; pluggable Encoder; per-game scientific-method Ledger.
- All unit tests pass.
- End-to-end probes on the public 25-game set: TBD (next).

## Install

biobrain requires Python 3.12+ and the official ARC-AGI-3 SDK.

```bash
pip install -e .
pip install git+https://github.com/arcprize/ARC-AGI-3.git  # for env binding
```

Set `BIOBRAIN_ENV_DIR` to the directory containing per-game environment files:

```bash
export BIOBRAIN_ENV_DIR=/path/to/arc-agi-3/environment_files
```

## Use

```python
from biobrain import BioBrainV2
from biobrain.adapters.arc import ArenaEnv, ArcAdapter
from biobrain.perception.perceive import perceive, detect_events
from biobrain.perception.salience import Salience
from biobrain.types import ComputeBudget, Transition

# Build the brain with the ARC adapter
brain = BioBrainV2(seed=0, adapter=ArcAdapter())
brain.reset_game("vc33")

# Run one attempt
env = ArenaEnv("vc33", mode="OFFLINE")
sal = Salience()
brain.reset_attempt()
obs = env.reset()
prev = None; last_a = None

for step in range(200):
    if env.is_terminal(obs):
        break
    parsed = env.parse(obs)
    if parsed["grid"] is None:
        break
    sal.observe(parsed["grid"])
    state = perceive(parsed["grid"], prev,
                     score=parsed["score"],
                     level=parsed["levels_completed"],
                     available_actions=tuple(parsed.get("available_actions") or ()),
                     salience_mask=sal.mask())
    if prev is not None:
        events = detect_events(prev, state)
        brain.observe(Transition(before=prev, action=last_a, after=state, events=events))
    action = brain.act(state, ComputeBudget(actions_remaining=200-step,
                                              time_remaining_ms=10000,
                                              attempts_remaining=1))
    obs = env.step(action)
    prev = state
    last_a = action

env.close()

# Diagnostics
print(f"Hot calls: {brain.n_hot_calls}, Cold calls: {brain.n_cold_calls}")
print(f"Banked surprises: {brain.salience.n_banked}")
print(f"Ledger entries: {len(brain.ledger)}")
```

## Architecture in one paragraph

biobrain is a **generic-typed brain library** parameterized over `StateT` and `ActionT`. The brain operates entirely on **fact sets** (Spelke-axis-grounded predicates), with a pluggable **Encoder** mediating State-to-Facts conversion. Components communicate exclusively through a **composer** — no component knows about any other. The **control loop** is event-driven commit-and-monitor: the brain commits to a Program, executes it open-loop predicting each step, and re-engages expensive reasoning only on violations (prediction errors). The **lifecycle** is two-verb (`reset_game` for inter-game amnesia, `reset_attempt` for intra-game preservation) with an explicit `on_level_change` hook for the components that act at level transitions (Ledger, Planner, Salience).

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — full v0.2 architecture spec (the canonical doc)
- [`docs/COMPONENTS.md`](docs/COMPONENTS.md) — deep dive on each of the 8 components (bio analog, ML analog, ARC-AGI history)
- [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) — discipline patterns
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — build phases + open questions

## Layout

```
biobrain/
├── biobrain/
│   ├── types.py              — reference State/Action/Transition types
│   ├── protocols.py          — generic-type contracts (Encoder, Adapter, ...)
│   ├── brain_v2.py           — the canonical v0.2 composer
│   ├── motor_cortex/         — DSL action vocabulary
│   ├── perception/           — Encoder + multi-granularity perception
│   ├── salience/             — central curating organ
│   ├── curiosity/            — Bayesian WM + ICM
│   ├── critic/               — multi-extractor goal-setting library
│   ├── simulator/            — forward-queries via WM
│   ├── ledger/               — per-game program memory
│   ├── planner/              — commit-and-monitor control loop
│   ├── latent_inference/     — schema library (subsumed by Salience)
│   └── adapters/
│       └── arc/              — ARC-AGI-3 adapter (env binding + Encoder)
├── tests/
└── docs/
```

## License

MIT — see [LICENSE](LICENSE).

## Notes

This is a research repository. It is not (yet) a competition submission, not (yet) a paper, and not (yet) a benchmark winner. It is an architectural argument that solving ARC-AGI-3 requires online active inference, and that active inference decomposes cleanly into eight biological-RL components implementable symbolically without training. Whether the argument survives contact with the full benchmark is what subsequent measurements will determine.
