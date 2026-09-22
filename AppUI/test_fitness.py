"""
Test the fitness function:
1. Evaluate the original red/green/blue/yellow palette (baseline)
2. Manually shift green → cyan-ish to reduce protan/deutan confusion
3. Verify fitness improves and fidelity constraint is tracked
"""

import numpy as np
from cvd_module import srgb_to_lab, lab_to_srgb
from fitness import fitness, evaluate_baseline

# --- Original palette ---
original_rgb = np.array([
    [1.0, 0.0, 0.0],  # red
    [0.0, 1.0, 0.0],  # green
    [0.0, 0.0, 1.0],  # blue
    [1.0, 1.0, 0.0],  # yellow
])
original_lab = srgb_to_lab(original_rgb)

print("STEP 1: Baseline (no changes)")
print()
evaluate_baseline(original_lab)

# Fitness of the original = treating it as its own "candidate"
score_orig, details_orig = fitness(
    original_lab, original_lab, return_details=True
)
print(f"\nOriginal fitness score: {score_orig:.2f}")
print(f"  objective (min ΔE across CVDs): {details_orig['objective']:.2f}")
print(f"  penalty: {details_orig['penalty']:.2f}")
print(f"  feasible: {details_orig['feasible']}")

# --- Manual improvement ---
# The problem: green and yellow collapse under protan/deutan.
# Strategy: shift green toward cyan (increase b* negativity, lower a*)
# to separate it from yellow in the simulated space.
print("\n" + "=" * 55)
print("STEP 2: Manual tweak — shift green toward teal/cyan")
print("=" * 55)

candidate_lab = original_lab.copy()
# Shift green: L* stays similar, a* goes more negative, b* drops
candidate_lab[1] = [80.0, -60.0, 20.0]  # was [87.7, -86.2, 83.2]

candidate_rgb = lab_to_srgb(candidate_lab)
print(f"\nAdjusted green in sRGB: {np.round(candidate_rgb[1], 3)}")

score_adj, details_adj = fitness(
    candidate_lab, original_lab, tau=15.0, return_details=True
)

print(f"\nAdjusted fitness score: {score_adj:.2f}")
print(f"  objective (min ΔE across CVDs): {details_adj['objective']:.2f}")
print(f"  penalty: {details_adj['penalty']:.2f}")
print(f"  feasible: {details_adj['feasible']}")
print(f"  per-color drifts from original: {np.round(details_adj['drifts'], 2)}")
print(f"  min ΔE per CVD: {({k: round(v, 2) for k, v in details_adj['min_de_per_cvd'].items()})}")

print(f"\n--- Summary ---")
print(f"  Original score:  {score_orig:.2f}")
print(f"  Adjusted score:  {score_adj:.2f}")
print(f"  Improvement:     {score_adj - score_orig:+.2f}")

# --- Test with a VERY aggressive shift that violates fidelity ---
print("\n" + "=" * 55)
print("STEP 3: Over-aggressive shift (should trigger penalty)")
print("=" * 55)

wild_lab = original_lab.copy()
wild_lab[1] = [50.0, 30.0, -60.0]  # completely different color

score_wild, details_wild = fitness(
    wild_lab, original_lab, tau=15.0, return_details=True
)
print(f"\nWild fitness score: {score_wild:.2f}")
print(f"  objective: {details_wild['objective']:.2f}")
print(f"  penalty: {details_wild['penalty']:.2f}")
print(f"  feasible: {details_wild['feasible']}")
print(f"  per-color drifts: {np.round(details_wild['drifts'], 2)}")
print(f"\n  -> Even though objective is high, penalty kills the score.")