import json
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys

# ✅ Load bead definitions with safe path handling
beads_path = os.path.join(os.path.dirname(__file__), "beads.json")
BEADS = json.load(open(beads_path))

def _closest_bead(rgb):
    arr = np.array(rgb, dtype=int)
    return min(
        BEADS,
        key=lambda bead: np.linalg.norm(arr - np.array(bead["rgb"], dtype=int))
    )

def _rgb_to_hsl(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hls(r, g, b)  # returns (hue, lightness, saturation)

def _color_distance_hue(h1, h2):
    return min(abs(h1 - h2), 1 - abs(h1 - h2))

def _find_harmonious_beads(base_rgb, harmony='analogous', top_n=5):
    h_base, l_base, s_base = _rgb_to_hsl(base_rgb)
    matched_beads = []

    for bead in BEADS:
        h, l, s = _rgb_to_hsl(bead["rgb"])

        if harmony == 'complementary':
            if _color_distance_hue(h, (h_base + 0.5) % 1.0) < 0.1:
                matched_beads.append(bead)

        elif harmony == 'analogous':
            if _color_distance_hue(h, h_base) < 0.15:
                matched_beads.append(bead)

    return matched_beads[:top_n]

def extract_palette(img_bytes, k=8, top_n=5, crop_frac=1.0):  # full image
    # --- decode & resize ---
    arr     = np.frombuffer(img_bytes, np.uint8)
    bgr     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb     = cv2.resize(rgb, (224, 224))

    # --- full image instead of crop ---
    crop = rgb

    # --- prepare for clustering & saturation ---
    pixels_rgb = crop.reshape(-1, 3)
    pixels_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # --- remove likely skin tones ---
    hue   = pixels_hsv[:, 0] * 2
    sat   = pixels_hsv[:, 1]
    value = pixels_hsv[:, 2]
    non_skin_mask = ~((hue >= 0) & (hue <= 50) & (sat > 50) & (value > 80))

    filtered_rgb = pixels_rgb[non_skin_mask]
    if len(filtered_rgb) < k:
        filtered_rgb = pixels_rgb  # fallback

    # --- KMeans ---
    km       = KMeans(n_clusters=k, n_init="auto").fit(filtered_rgb)
    centers  = km.cluster_centers_
    labels   = km.labels_

    # --- gather stats per cluster ---
    counts   = np.bincount(labels, minlength=k)
    max_cnt  = counts.max() or 1
    clusters = []
    for i, cen in enumerate(centers):
        mask = (labels == i)
        sat = float(np.mean(pixels_hsv[non_skin_mask][mask, 1])) / 255.0 if mask.any() else 0.0
        cnt = counts[i]
        score = 0.6 * sat + 0.4 * (cnt / max_cnt)
        clusters.append((i, score))

    # --- pick top_n clusters by score ---
    top_idxs = [i for i, _ in sorted(clusters, key=lambda x: x[1], reverse=True)[:top_n]]

    # --- pick harmonious beads based on most dominant ---
    main_idx = top_idxs[0]
    main_rgb = tuple(centers[main_idx].astype(int))
    suggestions = _find_harmonious_beads(main_rgb, harmony='analogous', top_n=top_n)

    return suggestions
