"""Probe — cd82 selector observability.

The decisive pre-build experiment per God Mode §7. The question:
*does cd82's armed_color have a visible tell that fine-attention perception
can recover, or is it genuinely modal (no visible state at any granularity)?*

Pre-registered outcomes:
  PASS — fine-attention emits a fact that distinguishes "armed white" from
         "armed dark" (or any other arming). → Perception is the fix; no
         LCI machinery is needed for cd82.
  FAIL — even with finer perception over the selector region, no fact
         distinguishes the armed states. → cd82 is genuinely modal; LCI
         backprop in Salience is justified.

Methodology:
  1. Reset cd82. Capture baseline grid.
  2. Click the WHITE color selector. Capture grid 1.
  3. Reset. Click the DARK color selector. Capture grid 2.
  4. Diff grid 0 vs grid 1, grid 0 vs grid 2, grid 1 vs grid 2.
     The diffs identify WHERE the armed-state is encoded visually.
  5. Apply the encoder with fine attention at those cells.
     Check whether the emitted fact set differs between the two armings.

If grid 1 and grid 2 differ visually, the armed_color IS visible —
the question is just whether OUR perception sees it. Pass condition:
encoder emits at least one differing fact between armed_white and
armed_dark states under fine attention.

Usage:
    BIOBRAIN_ENV_DIR=... python bench/probe_cd82_selector_observability.py
"""

import logging
import sys

logging.disable(logging.CRITICAL)

import numpy as np

from biobrain.adapters.arc import ArenaEnv
from biobrain.perception.perceive import perceive
from biobrain.perception.salience import Salience
from biobrain.perception.encoder import DefaultSpelkeEncoder
from biobrain.types import action_click


# cd82 layout from earlier inspection:
#   Top-left target pattern: rows 3-12, cols 3-12 (color 0 + color 15)
#   Top-right color swatches:
#     DARK (color 0) swatch:  rows 3-5, cols 36-38
#     WHITE (color 15) swatch: rows 3-5, cols 42-44
SELECTOR_WHITE = (43, 4)  # (col, row) center of white selector
SELECTOR_DARK  = (37, 4)  # (col, row) center of dark selector


def grab_state(env, click=None, prev=None, sal=None):
    """Reset env, optionally apply one click, return parsed grid + state."""
    obs = env.reset()
    parsed = env.parse(obs)
    grid0 = np.array(parsed["grid"], dtype=np.int8).copy()
    if click is not None:
        obs = env.step(action_click(*click))
        parsed = env.parse(obs)
    grid = np.array(parsed["grid"], dtype=np.int8)
    sal_local = sal or Salience()
    sal_local.observe(grid)
    state = perceive(grid, prev,
                     score=parsed["score"],
                     level=parsed["levels_completed"],
                     available_actions=tuple(int(a) for a in
                                              parsed.get("available_actions") or ()),
                     salience_mask=sal_local.mask())
    return grid, state, grid0


def diff_cells(g_a, g_b):
    """Cells where g_a and g_b differ (as set of (row, col, val_a, val_b))."""
    if g_a.shape != g_b.shape:
        return None
    differing = []
    for r in range(g_a.shape[0]):
        for c in range(g_a.shape[1]):
            if g_a[r, c] != g_b[r, c]:
                differing.append((r, c, int(g_a[r, c]), int(g_b[r, c])))
    return differing


def main():
    print("=" * 70)
    print("cd82 selector observability probe")
    print("=" * 70)

    env = ArenaEnv("cd82", mode="OFFLINE")
    encoder = DefaultSpelkeEncoder()

    # 1. Baseline (no click yet)
    print("\n[Step 1] Baseline state — no selector clicked")
    grid_baseline, state_baseline, _ = grab_state(env, click=None)
    print(f"  grid shape: {grid_baseline.shape}, "
          f"entities: {len(state_baseline.entities)}, "
          f"score: {state_baseline.score}, level: {state_baseline.level}")

    # 2. After clicking WHITE selector
    env_w = ArenaEnv("cd82", mode="OFFLINE")
    print(f"\n[Step 2] After click on WHITE selector at {SELECTOR_WHITE}")
    grid_white, state_white, _ = grab_state(env_w, click=SELECTOR_WHITE)
    diff_w = diff_cells(grid_baseline, grid_white)
    print(f"  cells changed from baseline: {len(diff_w)}")
    if 0 < len(diff_w) <= 30:
        print(f"  changed cells: {diff_w[:15]}")

    # 3. After clicking DARK selector (fresh env)
    env_d = ArenaEnv("cd82", mode="OFFLINE")
    print(f"\n[Step 3] After click on DARK selector at {SELECTOR_DARK}")
    grid_dark, state_dark, _ = grab_state(env_d, click=SELECTOR_DARK)
    diff_d = diff_cells(grid_baseline, grid_dark)
    print(f"  cells changed from baseline: {len(diff_d)}")
    if 0 < len(diff_d) <= 30:
        print(f"  changed cells: {diff_d[:15]}")

    # 4. The diagnostic: do WHITE-armed and DARK-armed differ visually?
    diff_wd = diff_cells(grid_white, grid_dark)
    print(f"\n[Step 4] WHITE-armed grid vs DARK-armed grid:")
    print(f"  cells differing between the two armings: "
          f"{len(diff_wd) if diff_wd is not None else 'shape-mismatch'}")
    if diff_wd is not None and 0 < len(diff_wd) <= 50:
        print(f"  differing cells (the armed-state tell): {diff_wd}")

    # 5. Decision tree
    print(f"\n[Step 5] Decision:")
    if not diff_wd:
        print("  ✗ NO VISIBLE DIFFERENCE between armings.")
        print("    cd82's armed_color is genuinely not encoded in the grid.")
        print("    LCI backprop in Salience IS JUSTIFIED.")
        env.close(); env_w.close(); env_d.close()
        return 2

    # We have a visible tell. Test whether the encoder sees it.
    print(f"  ✓ Armed-state IS visible ({len(diff_wd)} cells differ).")

    # Identify the cells where the tell lives
    tell_cells = frozenset((r, c) for r, c, _, _ in diff_wd)

    # 6. Does the encoder, WITHOUT fine attention, see the difference?
    coarse_white = encoder.encode(state_white)
    coarse_dark = encoder.encode(state_dark)
    coarse_diff = coarse_white.symmetric_difference(coarse_dark)
    print(f"\n[Step 6] Coarse encoding: facts differing = {len(coarse_diff)}")
    if coarse_diff:
        sample = list(coarse_diff)[:8]
        print(f"  sample differing facts: {sample}")

    # 7. With fine attention at the tell cells
    fine_white = encoder.encode(state_white, attention_hint=tell_cells)
    fine_dark = encoder.encode(state_dark, attention_hint=tell_cells)
    fine_diff = fine_white.symmetric_difference(fine_dark)
    print(f"\n[Step 7] Fine-attention encoding ({len(tell_cells)} cells):")
    print(f"  facts differing between armings: {len(fine_diff)}")
    if fine_diff:
        sample = list(fine_diff)[:8]
        print(f"  sample differing facts: {sample}")

    # 8. Verdict
    print(f"\n[Step 8] Verdict:")
    coarse_sees = len(coarse_diff) > 0
    fine_sees = len(fine_diff) > 0

    if coarse_sees:
        print("  ✓✓ COARSE ENCODING ALREADY DISTINGUISHES armings.")
        print("     Perception was NOT the bottleneck. Look elsewhere for")
        print("     why cd82 didn't score (likely Planner/Critic).")
        verdict = "coarse_sufficient"
    elif fine_sees:
        print("  ✓ FINE ATTENTION DISTINGUISHES armings; coarse does not.")
        print("    Perception is the fix. Wire fine attention to Salience's")
        print("    request mechanism — LCI machinery NOT needed for cd82.")
        verdict = "fine_attention_suffices"
    else:
        print("  ✗ NEITHER coarse NOR fine recovers the distinction.")
        print("    The encoder's predicate vocabulary is insufficient even at")
        print("    cell granularity. Either extend the vocabulary OR LCI is")
        print("    needed to posit the invisible-to-encoder latent.")
        verdict = "encoder_insufficient"

    env.close(); env_w.close(); env_d.close()
    print()
    print(f"VERDICT: {verdict}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
