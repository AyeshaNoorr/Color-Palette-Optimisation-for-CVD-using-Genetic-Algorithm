"""
CVD Simulation Module
=====================
Wraps DaltonLens-Python to provide simulation of color vision deficiencies
(protanopia, deuteranopia, tritanopia) for palettes and images.

Design choices:
- Brettel 1997 is the default model because it is the only one of the three
  major models that handles tritanopia in a principled way.
- All palette operations work in sRGB [0,1] floats internally, but DaltonLens
  expects uint8 [0,255], so conversions happen at the boundary.
- A separate helper handles CIELAB conversions for the fitness function,
  using colour-science for accuracy.
"""

import numpy as np
from daltonlens import simulate
import colour


# ---------------------------------------------------------------------------
# CVD type registry
# ---------------------------------------------------------------------------
CVD_TYPES = {
    "protan": simulate.Deficiency.PROTAN,   # red-blind
    "deutan": simulate.Deficiency.DEUTAN,   # green-blind
    "tritan": simulate.Deficiency.TRITAN,   # blue-blind
}

# Single shared simulator instance — Brettel handles all three deficiencies
_SIMULATOR = simulate.Simulator_Brettel1997()


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def simulate_palette(palette_rgb, cvd_type="deutan", severity=1.0):
    """
    Simulate how a palette is perceived under a given CVD.

    Parameters
    ----------
    palette_rgb : ndarray, shape (k, 3), float in [0, 1]
        Palette of k colors in sRGB.
    cvd_type : str
        One of "protan", "deutan", "tritan".
    severity : float in [0, 1]
        1.0 = full dichromacy, <1.0 = anomalous trichromacy.

    Returns
    -------
    ndarray, shape (k, 3), float in [0, 1]
        Simulated palette in sRGB.
    """
    if cvd_type not in CVD_TYPES:
        raise ValueError(f"cvd_type must be one of {list(CVD_TYPES.keys())}")

    # DaltonLens expects an image-like uint8 array, so reshape (k,3) -> (1,k,3)
    palette_uint8 = (np.clip(palette_rgb, 0, 1) * 255).astype(np.uint8)
    image_like = palette_uint8.reshape(1, -1, 3)

    simulated = _SIMULATOR.simulate_cvd(
        image_like,
        deficiency=CVD_TYPES[cvd_type],
        severity=severity,
    )

    return simulated.reshape(-1, 3).astype(np.float64) / 255.0


def simulate_image(image_rgb, cvd_type="deutan", severity=1.0):
    """
    Simulate a full image. Accepts uint8 (H,W,3) or float [0,1].
    Returns the same dtype as the input.
    """
    if cvd_type not in CVD_TYPES:
        raise ValueError(f"cvd_type must be one of {list(CVD_TYPES.keys())}")

    is_float = image_rgb.dtype != np.uint8
    img = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8) if is_float else image_rgb

    simulated = _SIMULATOR.simulate_cvd(
        img, deficiency=CVD_TYPES[cvd_type], severity=severity
    )

    return simulated.astype(np.float64) / 255.0 if is_float else simulated


# ---------------------------------------------------------------------------
# Color space conversions (sRGB <-> CIELAB) for the fitness function
# ---------------------------------------------------------------------------
def srgb_to_lab(rgb):
    """sRGB [0,1] -> CIELAB. Accepts (..., 3) array."""
    xyz = colour.sRGB_to_XYZ(rgb)
    return colour.XYZ_to_Lab(xyz)


def lab_to_srgb(lab):
    """CIELAB -> sRGB [0,1], clipped to gamut. Accepts (..., 3) array."""
    xyz = colour.Lab_to_XYZ(lab)
    rgb = colour.XYZ_to_sRGB(xyz)
    return np.clip(rgb, 0, 1)


def delta_e(lab1, lab2):
    """CIEDE2000 ΔE between two CIELAB colors or arrays of colors."""
    return colour.delta_E(lab1, lab2, method="CIE 2000")


# ---------------------------------------------------------------------------
# Convenience: simulate a palette and return it directly in CIELAB
# (this is what the fitness function will call most often)
# ---------------------------------------------------------------------------
def simulate_palette_lab(palette_lab, cvd_type="deutan", severity=1.0):
    """
    Take a palette in CIELAB, simulate the CVD perception, and return
    the simulated palette back in CIELAB. This is the main entry point
    for the fitness function.
    """
    rgb = lab_to_srgb(palette_lab)
    simulated_rgb = simulate_palette(rgb, cvd_type, severity)
    return srgb_to_lab(simulated_rgb)