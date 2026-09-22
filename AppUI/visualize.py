"""
visualize.py
============
Shows a before/after comparison of the palette optimization.
 
Produces a 2-row grid:
    ROW 1 — original palette: normal vision + each CVD simulation
    ROW 2 — optimised palette: normal vision + each CVD simulation
 
Also plots the GA convergence curve (fitness over generations).
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cvd_module import lab_to_srgb, simulate_palette_lab, srgb_to_lab
 
 
def draw_palette_row(ax, palette_rgb, title, subtitle=""):
    """Draw a row of colour swatches on a matplotlib axis."""
    k = len(palette_rgb)
    ax.set_xlim(0, k)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=8)
    ax.axis("off")
 
    for i, colour in enumerate(palette_rgb):
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.1), 0.88, 0.75,
            boxstyle="round,pad=0.02",
            facecolor=np.clip(colour, 0, 1),
            edgecolor="white",
            linewidth=1.5
        )
        ax.add_patch(rect)
 
        # Print hex code below each swatch
        hex_col = "#{:02x}{:02x}{:02x}".format(
            int(colour[0]*255), int(colour[1]*255), int(colour[2]*255)
        )
        ax.text(i + 0.49, 0.04, hex_col, ha="center", va="bottom",
                fontsize=7, color="#555555", fontfamily="monospace")
 
 
def show_palette_comparison(original_lab, optimized_lab,
                             cvd_type="deutan", save_path=None):
    """
    Show the before/after comparison grid for all CVD types.
 
    Parameters
    ----------
    original_lab  : ndarray (k, 3) — original palette in CIELAB
    optimized_lab : ndarray (k, 3) — optimized palette in CIELAB
    cvd_type      : str or list of str — CVD type(s) to simulate
    save_path     : str or None — if given, saves the figure to this path
    """
    # Normalise to a list so the rest of the code is uniform
    if isinstance(cvd_type, str):
        cvd_types = [cvd_type]
    else:
        cvd_types = list(cvd_type)
 
    cvd_labels = {"deutan": "Deuteranopia", "protan": "Protanopia",
                  "tritan": "Tritanopia"}
 
    n_cvd = len(cvd_types)
    n_cols = 1 + n_cvd          # first column = normal vision
 
    orig_rgb = lab_to_srgb(original_lab)
    opt_rgb  = lab_to_srgb(optimized_lab)
 
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 5))
 
    # If only 1 CVD type, axes shape is (2, 2) — fine.
    # If n_cols == 1 somehow, force 2D — but that can't happen (min 1 CVD + normal = 2 cols)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
 
    fig.suptitle("Colour Palette Optimisation for CVD Accessibility",
                 fontsize=14, fontweight="bold", y=1.02)
 
    # Column 0: normal vision
    draw_palette_row(axes[0, 0], orig_rgb,
                     "Original Palette", "Normal vision")
    draw_palette_row(axes[1, 0], opt_rgb,
                     "Optimised Palette", "Normal vision")
 
    # Columns 1..n_cvd: one per CVD type
    for col, cvd in enumerate(cvd_types, start=1):
        label = cvd_labels.get(cvd, cvd)
 
        orig_sim_rgb = lab_to_srgb(simulate_palette_lab(original_lab,  cvd_type=cvd))
        opt_sim_rgb  = lab_to_srgb(simulate_palette_lab(optimized_lab, cvd_type=cvd))
 
        draw_palette_row(axes[0, col], orig_sim_rgb,
                         "Original Palette", f"As seen with {label}")
        draw_palette_row(axes[1, col], opt_sim_rgb,
                         "Optimised Palette", f"As seen with {label}")
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
 
 
def show_convergence(history, save_path=None):
    """
    Plot the GA fitness over generations.
 
    Parameters
    ----------
    history   : list of float — best fitness per generation
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=(8, 4))
 
    ax.plot(history, color="#4C72B0", linewidth=2)
    ax.fill_between(range(len(history)), history, alpha=0.15, color="#4C72B0")
 
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (min ΔE)", fontsize=12)
    ax.set_title("GA Convergence — Fitness Over Generations", fontsize=13)
    ax.grid(True, alpha=0.3)
 
    # Mark the final value
    ax.axhline(history[-1], color="red", linestyle="--", alpha=0.5,
               label=f"Final: {history[-1]:.2f}")
    ax.legend()
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()