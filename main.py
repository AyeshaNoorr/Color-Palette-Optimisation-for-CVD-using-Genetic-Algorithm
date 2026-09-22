
import argparse

import numpy as np
import pandas as pd

import colour_math as cm
from problem import Problem
from optimisers import OPTIMISERS, repair_palette

DEFAULT = ["#D73027", "#FC8D59", "#FEE08B", "#D9EF8B", "#91CF60", "#1A9850"]  # RdYlGn-6


def main():
    ap = argparse.ArgumentParser(description="Minimal-change CVD palette repair")
    ap.add_argument("--hex", nargs="+", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--colours", type=int, default=6)
    ap.add_argument("--delta", type=float, default=10.0, help="accessibility target (dE00)")
    ap.add_argument("--tau", type=float, default=10.0, help="max drift per colour (dE00)")
    ap.add_argument("--ordered", action="store_true", help="preserve lightness order (ramps)")
    ap.add_argument("--optimiser", default="GA", choices=list(OPTIMISERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="save a before/after figure (pdf/png)")
    a = ap.parse_args()

    if a.image:
        from palette_extraction import extract_palette
        hexes = extract_palette(a.image, k=a.colours)
    else:
        hexes = [h.upper() for h in (a.hex or DEFAULT)]

    p = Problem(cm.hex_to_lab(hexes), delta=a.delta, tau=a.tau, ordered=a.ordered)
    ev = p.evaluate(p.original, margin=0.0)
    print(f"\nPalette ({p.k} colours): {' '.join(hexes)}")
    print("Worst pair per observer: " + ", ".join(
        f"{o} {v:.2f}" for o, v in zip(p.observers, ev["min_obs"])))
    conf = sorted(p.conflicts(), key=lambda c: c[3])
    if not conf:
        print(f"Already accessible (every pair >= {a.delta} dE00). Nothing changed.")
        return
    print(f"\n{len(conf)} conflicts (dE00 < {a.delta}):")
    for i, j, o, d in conf:
        print(f"  colours {i}-{j}  {o:7s} {d:6.2f}")
    print(f"Minimum vertex cover (mutable colours): {p.exact_cover()}")

    r = repair_palette(p, a.optimiser, seed=a.seed)
    print(f"\nRepaired in {r['runtime']:.2f}s, {r['evals']} evaluations, "
          f"{r['escalations']} escalation(s)")
    for i, (h0, h1, d) in enumerate(zip(hexes, r["hex"], p.evaluate(r["final_lab"], margin=0)["drift"])):
        tag = "changed" if h0 != h1 else "kept"
        print(f"  [{i}] {h0} -> {h1}  drift {d:5.2f}  {tag}")
    print("Worst pair per observer: " + ", ".join(
        f"{o} {v:.2f}" for o, v in zip(("normal", "protan", "deutan", "tritan"), r["min_obs"])))
    print(f"Accessible: {r['accessible']}  |  within tau: {r['faithful']}  |  "
          f"total drift {r['total_drift']:.2f}  |  colours changed {r['n_modified']}/{p.k}")

    if a.save:
        import os
        from report import fig_qualitative
        row = pd.DataFrame([dict(instance="Palette", method="GA", success=r["success"],
                                 total_drift=r["total_drift"], original_hex=hexes,
                                 hex=r["hex"])])
        out, name = os.path.split(os.path.abspath(a.save))
        fig_qualitative(row, out, ["Palette"], name=name)


if __name__ == "__main__":
    main()
