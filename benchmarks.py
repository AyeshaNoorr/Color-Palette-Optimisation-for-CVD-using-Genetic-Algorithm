"""
benchmarks.py
=============
Test palettes. Replaces TEST_SUITE in validate.py.

ILLUSTRATIVE : the five scenarios from the draft, now all evaluated under
               all four observers, with the FULL Okabe-Ito palette.
REAL         : widely used categorical palettes.
SYNTHETIC    : seeded random sRGB palettes (stored as hex so the
               original is exactly representable).
"""
import numpy as np

ILLUSTRATIVE = {
    "Red-green-brown": ["#B22222", "#228B22", "#8B4513", "#D2691E", "#808000"],
    "RdYlGn-5 (ordered)": ["#D73027", "#FC8D59", "#FEE08B", "#D9EF8B", "#91CF60"],
    "Pastel UI": ["#87CEEB", "#98FB98", "#FFB6C1", "#FFFFE0", "#E6E6FA"],
    "RGB+CMY primaries": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"],
}

REAL = {
    "Okabe-Ito": ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                  "#0072B2", "#D55E00", "#CC79A7"],
    "IBM": ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"],
    "Tol Bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
                   "#AA3377", "#BBBBBB"],
    "Tol Vibrant": ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311",
                    "#009988", "#BBBBBB"],
    "Tableau 10": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                   "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
    "Set1": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
             "#FFFF33", "#A65628", "#F781BF", "#999999"],
    "Set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
             "#FFD92F", "#E5C494", "#B3B3B3"],
    "Dark2": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
              "#E6AB02", "#A6761D", "#666666"],
}

ORDERED = {"RdYlGn-5 (ordered)"}


def synthetic(ks=(4, 6, 8), per_k=8, seed=2027):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        for n in range(per_k):
            rgb = rng.integers(0, 256, size=(k, 3))
            out[f"Rand-k{k}-{n + 1}"] = ["#{:02X}{:02X}{:02X}".format(*c) for c in rgb]
    return out


def all_instances(include_synthetic=True):
    inst = {**ILLUSTRATIVE, **REAL}
    if include_synthetic:
        inst.update(synthetic())
    return inst
