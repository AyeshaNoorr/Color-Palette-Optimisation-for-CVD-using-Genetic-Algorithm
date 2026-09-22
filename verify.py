
import numpy as np
import colour
from daltonlens import simulate
import colour_math as cm

rng = np.random.default_rng(0)
rgb = rng.random((5000, 3))


lab_ref = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))
lab = cm.srgb_to_lab(rgb)
print(f"sRGB->LAB   max |diff| vs colour-science : {np.abs(lab - lab_ref).max():.2e}")
print(f"LAB->sRGB   round-trip max |diff|         : {np.abs(cm.lab_to_srgb(lab) - rgb).max():.2e}")

# 2. CIEDE2000 on random pairs and on the Sharma et al. (2005) test set
lab2 = cm.srgb_to_lab(rng.random((5000, 3)))
de_ref = colour.delta_E(lab, lab2, method="CIE 2000")
print(f"CIEDE2000   max |diff| vs colour-science : {np.abs(cm.ciede2000(lab, lab2) - de_ref).max():.2e}")
sharma = np.array([  # (L1,a1,b1,L2,a2,b2,dE) selected rows from Sharma et al. Table 1
    [50, 2.6772, -79.7751, 50, 0, -82.7485, 2.0425],
    [50, 3.1571, -77.2803, 50, 0, -82.7485, 2.8615],
    [50, -1.3802, -84.2814, 50, 0, -82.7485, 1.0000],
    [50, 0, 0, 50, -1, 2, 2.3669],
    [50, 2.49, -0.001, 50, -2.49, 0.0009, 7.1792],
    [50, 2.5, 0, 73, 25, -18, 27.1492],
    [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
    [22.7233, 20.0904, -46.694, 23.0331, 14.973, -42.5619, 2.0373],
    [2.0776, 0.0795, -1.135, 0.9033, -0.0636, -0.5514, 0.9082],
])
err = np.abs(cm.ciede2000(sharma[:, :3], sharma[:, 3:6]) - sharma[:, 6]).max()
print(f"CIEDE2000   max |diff| vs Sharma table   : {err:.2e}")

# 3. Brettel 1997 vs DaltonLens 
sim = simulate.Simulator_Brettel1997()
rgb8 = (rng.random((1, 5000, 3)) * 255).round().astype(np.uint8)
names = {"protan": simulate.Deficiency.PROTAN, "deutan": simulate.Deficiency.DEUTAN,
         "tritan": simulate.Deficiency.TRITAN}
for obs, d in names.items():
    ref = sim.simulate_cvd(rgb8, d, severity=1.0).astype(float)
    ours = cm.linear_to_srgb(cm.simulate_linear(cm.srgb_to_linear(rgb8 / 255.0), obs)) * 255
    diff = np.abs(ours - ref)
    print(f"Brettel {obs}: max |diff| = {diff.max():.2f} code values, "
          f"{(diff > 1).mean() * 100:.2f}% of channels differ by >1")
