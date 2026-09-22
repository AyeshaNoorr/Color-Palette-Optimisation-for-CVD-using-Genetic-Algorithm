"""Quick sanity tests for cvd_module."""
import numpy as np
from cvd_module import (
    simulate_palette, simulate_palette_lab,
    srgb_to_lab, lab_to_srgb, delta_e,
)

# A classic problem palette: pure red and pure green
palette_rgb = np.array([
    [1.0, 0.0, 0.0],  # red
    [0.0, 1.0, 0.0],  # green
    [0.0, 0.0, 1.0],  # blue
    [1.0, 1.0, 0.0],  # yellow
])

print("=" * 60)
print("Original palette (sRGB):")
print(palette_rgb)

# Convert to LAB
palette_lab = srgb_to_lab(palette_rgb)
print("\nOriginal palette (CIELAB):")
print(np.round(palette_lab, 2))

# Pairwise ΔE in original (normal vision)
print("\nPairwise ΔE (CIEDE2000) — normal vision:")
n = len(palette_lab)
for i in range(n):
    for j in range(i + 1, n):
        de = delta_e(palette_lab[i], palette_lab[j])
        print(f"  color {i} vs {j}: ΔE = {de:.2f}")

# Now simulate each CVD type
for cvd in ["protan", "deutan", "tritan"]:
    print(f"\n--- {cvd.upper()} simulation ---")
    sim_lab = simulate_palette_lab(palette_lab, cvd_type=cvd, severity=1.0)
    print(f"Simulated palette (CIELAB):\n{np.round(sim_lab, 2)}")

    # Find minimum pairwise ΔE — this is the metric we optimize
    min_de = np.inf
    worst_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            de = delta_e(sim_lab[i], sim_lab[j])
            if de < min_de:
                min_de = de
                worst_pair = (i, j)
    print(f"Min pairwise ΔE = {min_de:.2f} (worst pair: {worst_pair})")

print("\n" + "=" * 60)
print("Sanity check complete.")