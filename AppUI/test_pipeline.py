"""
End-to-end test: generate a sample chart image, extract its palette,
and evaluate it under CVD simulation.

This demonstrates the full pipeline that leads into the optimizer.
"""

import numpy as np
from PIL import Image, ImageDraw

from palette_extraction import extract_palette, print_palette, load_image
from cvd_module import srgb_to_lab, simulate_image
from fitness import evaluate_baseline, fitness


# -----------------------------------------------------------------------
# Step 1: Create a synthetic bar chart image with problematic colors
# -----------------------------------------------------------------------
def create_test_chart(path="test_chart.png", width=400, height=300):
    """
    Generate a simple bar chart using colors known to be confusing
    for red-green CVD: red, green, orange, lime, brown.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Bar colors chosen to be deliberately problematic
    bar_colors = [
        (220, 50, 50),    # red
        (50, 180, 50),    # green
        (230, 150, 30),   # orange
        (120, 200, 60),   # lime green
        (160, 80, 40),    # brown
        (50, 50, 180),    # blue (control — should remain distinguishable)
    ]

    n_bars = len(bar_colors)
    margin = 40
    bar_width = (width - 2 * margin) // n_bars
    max_bar_h = height - 2 * margin

    # Random-ish bar heights
    heights = [0.7, 0.9, 0.5, 0.8, 0.6, 0.85]

    for i, (color, h) in enumerate(zip(bar_colors, heights)):
        x0 = margin + i * bar_width + 5
        x1 = x0 + bar_width - 10
        y1 = height - margin
        y0 = y1 - int(h * max_bar_h)
        draw.rectangle([x0, y0, x1, y1], fill=color)

    img.save(path)
    print(f"Test chart saved to {path}")
    return path


# -----------------------------------------------------------------------
# Step 2: Run the pipeline
# -----------------------------------------------------------------------
chart_path = create_test_chart()

# Load and extract palette
image_rgb = load_image(chart_path)
palette_lab, palette_rgb, labels, counts = extract_palette(image_rgb, k=6)

print("\n--- Extracted Palette ---")
print_palette(palette_lab, palette_rgb, counts)

# Evaluate baseline under CVD
print("\n--- CVD Baseline ---")
evaluate_baseline(palette_lab)

# Full fitness score
score, details = fitness(palette_lab, palette_lab, return_details=True)
print(f"\nBaseline fitness score: {score:.2f}")
print(f"Worst CVD: min ΔE per type = {({k: round(v, 2) for k, v in details['min_de_per_cvd'].items()})}")
print(f"Worst pairs: {details['worst_pair_per_cvd']}")

# -----------------------------------------------------------------------
# Step 3: Show what the chart looks like under CVD simulation
# -----------------------------------------------------------------------
print("\n--- Generating CVD-simulated versions of the chart ---")
full_image = np.asarray(Image.open(chart_path).convert("RGB"))
for cvd in ["protan", "deutan", "tritan"]:
    sim = simulate_image(full_image, cvd_type=cvd, severity=1.0)
    sim_uint8 = (sim * 255).astype(np.uint8)
    out_path = f"test_chart_{cvd}.png"
    Image.fromarray(sim_uint8).save(out_path)
    print(f"  Saved {out_path}")

print("\nPipeline complete. These simulated images show exactly the")
print("problem your optimizer will solve.")