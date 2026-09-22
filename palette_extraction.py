
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import colour_math as cm


def extract_palette(path, k=6, max_pixels=200_000, seed=42):
    """Return the k dominant colours as hex strings, most frequent first."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    lab = cm.srgb_to_lab(np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(lab)
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    return cm.lab_to_hex(km.cluster_centers_[order])
