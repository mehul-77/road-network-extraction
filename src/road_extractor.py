"""
Road Network Feature Extraction from Satellite Imagery
ITS Project - Classical Image Processing Approach
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
import os


# ─────────────────────────────────────────────
# 1. IMAGE LOADING
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# 2. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(img: np.ndarray) -> dict:
    """Convert to grayscale, denoise, enhance contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Bilateral filter — preserves edges while smoothing
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE for contrast enhancement (helps in low-contrast satellite images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return {"gray": gray, "denoised": denoised, "enhanced": enhanced}


# ─────────────────────────────────────────────
# 3. ROAD DETECTION
# ─────────────────────────────────────────────

def detect_roads(preprocessed: dict, original_rgb: np.ndarray = None) -> dict:
    """
    Improved road detection pipeline:
    1. HSV colour masking  — keeps only asphalt/concrete tones (grey, low-saturation)
    2. Canny edge detection + adaptive thresholding
    3. Directional morphological path-opening — suppresses square building blobs,
       keeps elongated linear road structures
    4. Gap-filling closing + noise removal
    """
    enhanced = preprocessed["enhanced"]

    # ── 1. HSV Colour Mask (asphalt = low saturation, mid-to-high value) ──────
    if original_rgb is not None:
        # Use the original RGB image for colour analysis
        hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    else:
        # Fallback: reconstruct a fake HSV-compatible image from grayscale
        gray_3ch = cv2.cvtColor(preprocessed["gray"], cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(gray_3ch, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0,   0,   40])
    upper_gray = np.array([179, 60, 220])   # low saturation → road-like surfaces
    color_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # ── 2. Edge Detection ─────────────────────────────────────────────────────
    edges = cv2.Canny(enhanced, threshold1=50, threshold2=150, apertureSize=3)

    # ── 3. Adaptive Thresholding ──────────────────────────────────────────────
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )

    # Apply colour mask to both detections to suppress vegetation / coloured roofs
    road_edges  = cv2.bitwise_and(edges,          edges,          mask=color_mask)
    road_thresh = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=color_mask)

    # Dilate edges to thicken thin road boundaries before combining
    kernel_line  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(road_edges, kernel_line, iterations=2)

    combined = cv2.bitwise_or(dilated_edges, road_thresh)

    # ── 4. Directional Path Opening — favour long linear structures ───────────
    # Multi-angle line kernels; the OR of all openings keeps elongated structures
    DIRECTION_LEN = 15
    directional_max = np.zeros_like(combined)
    for angle in range(0, 180, 22):
        k = np.zeros((DIRECTION_LEN, DIRECTION_LEN), dtype=np.uint8)
        cx = cy = DIRECTION_LEN // 2
        x1 = int(cx + (DIRECTION_LEN / 2) * np.cos(np.radians(angle)))
        y1 = int(cy - (DIRECTION_LEN / 2) * np.sin(np.radians(angle)))
        x2 = int(cx - (DIRECTION_LEN / 2) * np.cos(np.radians(angle)))
        y2 = int(cy + (DIRECTION_LEN / 2) * np.sin(np.radians(angle)))
        cv2.line(k, (x1, y1), (x2, y2), 1, 1)
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
        directional_max = cv2.bitwise_or(directional_max, opened)

    # ── 5. Gap Filling & Noise Removal ────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(directional_max, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Increased min_area to 400 to remove more noise blobs (validated in testing)
    binary_mask = remove_small_components(closed, min_area=400)

    # Skeletonize to get thin road centerlines
    skeleton = skeletonize(binary_mask)

    return {
        "edges": edges,
        "adaptive_thresh": adaptive_thresh,
        "combined": combined,
        "binary_mask": binary_mask,
        "skeleton": skeleton,
    }


def remove_small_components(binary: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to produce road centerlines."""
    skel = np.zeros_like(binary)
    img = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# ─────────────────────────────────────────────
# 4. METRICS COMPUTATION
# ─────────────────────────────────────────────

def compute_metrics(original: np.ndarray, results: dict) -> dict:
    h, w = original.shape[:2]
    total_pixels = h * w

    road_pixels = cv2.countNonZero(results["binary_mask"])
    road_area_pct = (road_pixels / total_pixels) * 100

    skeleton_pixels = cv2.countNonZero(results["skeleton"])

    # Approximate road length in pixels (skeleton pixel count ≈ centerline length)
    road_length_px = skeleton_pixels

    # Road density = road length / total area
    road_density = road_length_px / total_pixels

    # Connected road segments
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        results["binary_mask"], connectivity=8
    )
    num_segments = num_labels - 1  # exclude background

    metrics = {
        "image_size": f"{w} x {h} px",
        "total_pixels": total_pixels,
        "road_pixels": int(road_pixels),
        "road_area_percent": round(road_area_pct, 2),
        "road_length_pixels": int(road_length_px),
        "road_density_per_px": round(road_density, 6),
        "num_road_segments": int(num_segments),
    }
    return metrics


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

def visualize(original: np.ndarray, preprocessed: dict, results: dict, metrics: dict, save_path: str = None):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    panels = [
        (original,                      "Original Satellite Image",   "gray",    False),
        (preprocessed["enhanced"],      "Preprocessed (CLAHE)",       "gray",    True),
        (results["edges"],              "Canny Edge Detection",        "magma",   True),
        (results["adaptive_thresh"],    "Adaptive Threshold",          "gray",    True),
        (results["combined"],           "Combined (Edges + Thresh)",   "gray",    True),
        (results["binary_mask"],        "Road Binary Mask",            "gray",    True),
        (results["skeleton"],           "Road Centerlines (Skeleton)", "hot",     True),
    ]

    for idx, (data, title, cmap, is_gray) in enumerate(panels):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#0d1117")
        if is_gray:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.axis("off")

    # Overlay panel: skeleton on original
    ax_overlay = fig.add_subplot(gs[1, 3])
    ax_overlay.set_facecolor("#0d1117")
    overlay = original.copy()
    overlay[results["skeleton"] > 0] = [255, 80, 80]
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay: Roads on Image", color="white", fontsize=9, pad=6)
    ax_overlay.axis("off")

    # Metrics text box
    metric_text = (
        f"Image Size : {metrics['image_size']}\n"
        f"Road Area  : {metrics['road_area_percent']}%\n"
        f"Road Length: {metrics['road_length_pixels']} px\n"
        f"Density    : {metrics['road_density_per_px']}\n"
        f"Segments   : {metrics['num_road_segments']}"
    )
    fig.text(
        0.5, 0.01, metric_text,
        ha="center", va="bottom", color="#58a6ff",
        fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d")
    )

    fig.suptitle("Road Network Feature Extraction — ITS Project",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✅ Visualization saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(image_path: str, output_dir: str = "output"):
    Path(output_dir).mkdir(exist_ok=True)

    print(f"📡 Loading image: {image_path}")
    img = load_image(image_path)

    print("🔧 Preprocessing...")
    preprocessed = preprocess(img)

    print("🛣️  Detecting roads...")
    results = detect_roads(preprocessed, original_rgb=img)

    print("📊 Computing metrics...")
    metrics = compute_metrics(img, results)

    print("\n── ROAD NETWORK METRICS ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("──────────────────────────────────────────\n")

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {metrics_path}")

    # Save binary road mask
    mask_path = os.path.join(output_dir, "road_mask.png")
    cv2.imwrite(mask_path, results["binary_mask"])
    print(f"✅ Road mask saved → {mask_path}")

    # Visualize
    vis_path = os.path.join(output_dir, "visualization.png")
    visualize(img, preprocessed, results, metrics, save_path=vis_path)

    return metrics, results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_satellite.jpg"
    run_pipeline(image_path)
