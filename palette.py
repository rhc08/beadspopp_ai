import json
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 1) Load your bead definitions once
BEADS = json.load(open("beads.json"))

def _closest_bead(rgb):
    arr = np.array(rgb, dtype=int)
    return min(
        BEADS,
        key=lambda bead: np.linalg.norm(arr - np.array(bead["rgb"], dtype=int))
    )

def extract_palette(img_bytes, k=8, top_n=5, crop_frac=0.6):
    """
    1. Decode image bytes → RGB, resize to 224×224.
    2. Center‐crop to crop_frac (e.g. 0.6) so subject dominates.
    3. Run KMeans(k clusters) on the cropped region.
    4. For each cluster: compute its pixel_count and average saturation.
    5. Compute a score = 0.6 * saturation + 0.4 * (count / max_count).
    6. Sort clusters by that score and take top_n.
    7. Map each centroid to the nearest bead shade.
    """
    # --- decode & resize ---
    arr     = np.frombuffer(img_bytes, np.uint8)
    bgr     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb     = cv2.resize(rgb, (224, 224))

    # --- center‐crop ---
    h, w, _ = rgb.shape
    ch      = int(h * crop_frac)
    cw      = int(w * crop_frac)
    top     = (h - ch) // 2
    left    = (w - cw) // 2
    crop    = rgb[top:top+ch, left:left+cw]

    # --- prepare for clustering & saturation ---
    pixels_rgb = crop.reshape(-1, 3)
    pixels_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # --- KMeans ---
    km       = KMeans(n_clusters=k, n_init="auto").fit(pixels_rgb)
    centers  = km.cluster_centers_
    labels   = km.labels_

    # --- gather stats per cluster ---
    counts   = np.bincount(labels, minlength=k)
    max_cnt  = counts.max() or 1
    clusters = []
    for i, cen in enumerate(centers):
        mask = (labels == i)
        # average saturation [0..1]
        sat = float(np.mean(pixels_hsv[mask,1])) / 255.0 if mask.any() else 0.0
        cnt = counts[i]
        # combined score (tune weights as you like)
        score = 0.6 * sat + 0.4 * (cnt / max_cnt)
        clusters.append((i, score))

    # --- pick top_n clusters by score ---
    top_idxs = [i for i,_ in sorted(clusters, key=lambda x: x[1], reverse=True)[:top_n]]

    # --- map each centroid to the nearest bead shade ---
    suggestions = []
    for idx in top_idxs:
        rgb_centroid = tuple(centers[idx].astype(int))
        suggestions.append(_closest_bead(rgb_centroid))

    return suggestions
