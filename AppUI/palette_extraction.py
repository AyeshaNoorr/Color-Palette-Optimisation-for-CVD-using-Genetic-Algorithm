"""
Palette Extraction Module
=========================
Extracts the k dominant colors from an image using k-means clustering
in CIELAB space.

Why CIELAB?
    K-means minimizes Euclidean distance between points and centroids.
    In sRGB, Euclidean distance ≠ perceptual distance (e.g. equal steps
    in green look bigger than equal steps in blue). CIELAB is designed
    so that Euclidean distance ≈ perceptual difference, making k-means
    produce perceptually meaningful clusters.

Pipeline:
    1. Load image → sRGB float [0,1]
    2. (Optional) downsample for speed
    3. Convert every pixel to CIELAB
    4. Run k-means in CIELAB
    5. Return centroids (the palette) in both CIELAB and sRGB
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from cvd_module import srgb_to_lab, lab_to_srgb


def load_image(path, max_pixels=200_000):
    """
    Load an image and optionally downsample it.

    Parameters
    ----------
    path : str
        Path to image file (PNG, JPG, etc.).
    max_pixels : int
        If the image has more pixels than this, it is resized
        proportionally. This speeds up k-means without affecting
        the dominant colors significantly.

    Returns
    -------
    image_rgb : ndarray, shape (H, W, 3), float in [0, 1]
        The image in sRGB.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    total = w * h

    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return np.asarray(img).astype(np.float64) / 255.0


def extract_palette(image_rgb, k=6, random_state=42):
    """
    Extract k dominant colors from an image using k-means in CIELAB.

    Parameters
    ----------
    image_rgb : ndarray, shape (H, W, 3), float in [0, 1]
        Input image in sRGB.
    k : int
        Number of palette colors to extract.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    palette_lab : ndarray, shape (k, 3)
        Palette centroids in CIELAB.
    palette_rgb : ndarray, shape (k, 3)
        Same centroids converted back to sRGB [0, 1].
    labels : ndarray, shape (N,)
        Cluster assignment for each pixel (flattened).
        Useful for recoloring later.
    pixel_counts : ndarray, shape (k,)
        Number of pixels assigned to each cluster.
        Useful for weighting or visualization.
    """
    H, W, _ = image_rgb.shape

    # Convert entire image to CIELAB
    pixels_lab = srgb_to_lab(image_rgb).reshape(-1, 3)  # (N, 3)

    # Run k-means in CIELAB space
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels_lab)
    centroids_lab = kmeans.cluster_centers_  # (k, 3)

    # Convert centroids back to sRGB for display
    centroids_rgb = lab_to_srgb(centroids_lab)

    # Count pixels per cluster (for swatch sizing / importance)
    pixel_counts = np.bincount(labels, minlength=k)

    # Sort by pixel count (most dominant color first)
    order = np.argsort(-pixel_counts)
    centroids_lab = centroids_lab[order]
    centroids_rgb = centroids_rgb[order]
    pixel_counts = pixel_counts[order]

    # Remap labels to match the new order
    remap = np.zeros(k, dtype=int)
    remap[order] = np.arange(k)
    labels = remap[labels]

    return centroids_lab, centroids_rgb, labels, pixel_counts


def extract_palette_from_path(image_path, k=6, max_pixels=200_000,
                              random_state=42):
    """
    Convenience wrapper: path in, palette out.
    """
    image_rgb = load_image(image_path, max_pixels=max_pixels)
    return extract_palette(image_rgb, k=k, random_state=random_state)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------
def print_palette(palette_lab, palette_rgb, pixel_counts=None):
    """Pretty-print a palette with both CIELAB and sRGB values."""
    k = len(palette_lab)
    print(f"Extracted palette ({k} colors)")
    print("-" * 60)
    for i in range(k):
        L, a, b = palette_lab[i]
        r, g, bl = palette_rgb[i]
        count_str = f"  ({pixel_counts[i]:,} px)" if pixel_counts is not None else ""
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(bl * 255)
        )
        print(f"  [{i}] LAB=({L:6.1f}, {a:6.1f}, {b:6.1f})  "
              f"RGB=({r:.3f}, {g:.3f}, {bl:.3f})  "
              f"{hex_color}{count_str}")
    print("-" * 60)