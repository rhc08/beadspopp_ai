def extract_palette(img_bytes, k=8, top_n=5, crop_frac=1.0):  # Step 1: use full image
    # --- decode & resize ---
    arr     = np.frombuffer(img_bytes, np.uint8)
    bgr     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb     = cv2.resize(rgb, (224, 224))

    # --- full image instead of crop ---
    crop = rgb  # using entire resized image

    # --- prepare for clustering & saturation ---
    pixels_rgb = crop.reshape(-1, 3)
    pixels_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Optional: filter out likely skin pixels (hue approx. [0–50 deg] and high saturation)
    hue = pixels_hsv[:, 0] * 2  # OpenCV HSV hue range: [0–179] → [0–360]
    sat = pixels_hsv[:, 1]
    value = pixels_hsv[:, 2]
    non_skin_mask = ~((hue >= 0) & (hue <= 50) & (sat > 50) & (value > 80))

    filtered_rgb = pixels_rgb[non_skin_mask]
    if len(filtered_rgb) < k:
        filtered_rgb = pixels_rgb  # fallback if too few pixels

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
    top_idxs = [i for i,_ in sorted(clusters, key=lambda x: x[1], reverse=True)[:top_n]]

    # --- get the dominant color and find harmonious bead matches ---
    main_idx = top_idxs[0]
    main_rgb = tuple(centers[main_idx].astype(int))

    # harmony can be 'analogous' or 'complementary'
    suggestions = _find_harmonious_beads(main_rgb, harmony='analogous', top_n=top_n)

    return suggestions
