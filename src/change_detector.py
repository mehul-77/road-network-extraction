"""
Before / After Road Change Detection
Compares two satellite images of the same area taken at different times.
Highlights:
  - New roads     (green)  — present in AFTER, absent in BEFORE
  - Removed roads (red)    — present in BEFORE, absent in AFTER
  - Unchanged     (white)  — present in both

Usage:
    python change_detector.py before.jpg after.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from road_extractor import load_image, preprocess, detect_roads, compute_metrics


# ─────────────────────────────────────────────
# ALIGNMENT  (in case images are slightly shifted)
# ─────────────────────────────────────────────

def align_images(img_before: np.ndarray, img_after: np.ndarray) -> np.ndarray:
    """
    Align img_after to img_before using ORB feature matching + homography.
    Returns aligned version of img_after.
    """
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
    gray_after  = cv2.cvtColor(img_after,  cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after,  None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  ⚠  Not enough features for alignment — using images as-is")
        return img_after

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:50]

    if len(matches) < 4:
        print("  ⚠  Too few matches for alignment — using images as-is")
        return img_after

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return img_after

    h, w = img_before.shape[:2]
    aligned = cv2.warpPerspective(img_after, H, (w, h))
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB) if aligned.ndim == 3 else aligned


# ─────────────────────────────────────────────
# CHANGE DETECTION
# ─────────────────────────────────────────────

def detect_changes(mask_before: np.ndarray, mask_after: np.ndarray) -> dict:
    """
    Compare two binary road masks and return change maps.
    Both masks must be same size (uint8, 0/255).
    """
    # Resize after to match before if needed
    if mask_before.shape != mask_after.shape:
        mask_after = cv2.resize(mask_after, (mask_before.shape[1], mask_before.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    # Binarise to 0/1
    b = (mask_before > 127).astype(np.uint8)
    a = (mask_after  > 127).astype(np.uint8)

    new_roads     = ((a == 1) & (b == 0)).astype(np.uint8) * 255   # added
    removed_roads = ((a == 0) & (b == 1)).astype(np.uint8) * 255   # removed
    unchanged     = ((a == 1) & (b == 1)).astype(np.uint8) * 255   # same

    return {
        "new_roads":     new_roads,
        "removed_roads": removed_roads,
        "unchanged":     unchanged,
    }


def change_metrics(changes: dict, total_pixels: int) -> dict:
    new_px     = int(cv2.countNonZero(changes["new_roads"]))
    removed_px = int(cv2.countNonZero(changes["removed_roads"]))
    unchanged_px = int(cv2.countNonZero(changes["unchanged"]))

    return {
        "new_road_pixels":     new_px,
        "removed_road_pixels": removed_px,
        "unchanged_road_pixels": unchanged_px,
        "new_road_area_pct":     round(new_px / total_pixels * 100, 3),
        "removed_road_area_pct": round(removed_px / total_pixels * 100, 3),
        "net_change_pct":        round((new_px - removed_px) / total_pixels * 100, 3),
    }


# ─────────────────────────────────────────────
# COLOUR OVERLAY
# ─────────────────────────────────────────────

def build_change_overlay(img_before: np.ndarray, changes: dict) -> np.ndarray:
    """Overlay change map on the before image with colour coding."""
    overlay = img_before.copy()

    # New roads  → bright green
    overlay[changes["new_roads"] > 0] = [0, 220, 80]

    # Removed roads → bright red
    overlay[changes["removed_roads"] > 0] = [220, 40, 40]

    # Unchanged roads → white
    overlay[changes["unchanged"] > 0] = [240, 240, 240]

    return overlay


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def visualize_changes(img_before, img_after, mask_before, mask_after,
                      changes, metrics, save_path=None):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)

    overlay = build_change_overlay(img_before, changes)

    panels = [
        (img_before,               "Before Image",          False),
        (img_after,                "After Image",           False),
        (mask_before,              "Road Mask — Before",    True),
        (mask_after,               "Road Mask — After",     True),
        (changes["new_roads"],     "New Roads (Added)",     True),
        (changes["removed_roads"], "Removed Roads",         True),
        (changes["unchanged"],     "Unchanged Roads",       True),
        (overlay,                  "Change Overlay\n🟢 New  🔴 Removed  ⬜ Same", False),
    ]

    for idx, (data, title, is_gray) in enumerate(panels):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#0d1117")
        ax.imshow(data, cmap="gray" if is_gray else None)
        ax.set_title(title, color="white", fontsize=8.5, pad=5)
        ax.axis("off")

    metric_text = (
        f"New Roads      : {metrics['new_road_area_pct']}%  ({metrics['new_road_pixels']} px)\n"
        f"Removed Roads  : {metrics['removed_road_area_pct']}%  ({metrics['removed_road_pixels']} px)\n"
        f"Net Change     : {metrics['net_change_pct']:+.3f}%\n"
        f"Unchanged Roads: {metrics['unchanged_road_pixels']} px"
    )
    fig.text(
        0.5, 0.01, metric_text,
        ha="center", va="bottom", color="#58a6ff",
        fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d")
    )

    fig.suptitle("Road Change Detection — Before vs After",
                 color="white", fontsize=13, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✅ Change map saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_change_detection(before_path: str, after_path: str, output_dir: str = "output/change"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("📥 Loading images...")
    img_before = load_image(before_path)
    img_after  = load_image(after_path)

    print("🔧 Aligning images...")
    img_after_aligned = align_images(img_before, img_after)

    print("🛣️  Detecting roads in both images...")
    pre_b = preprocess(img_before)
    pre_a = preprocess(img_after_aligned)
    res_b = detect_roads(pre_b)
    res_a = detect_roads(pre_a)

    print("🔍 Computing changes...")
    changes = detect_changes(res_b["binary_mask"], res_a["binary_mask"])
    metrics = change_metrics(changes, img_before.shape[0] * img_before.shape[1])

    print("\n── CHANGE METRICS ───────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("──────────────────────────────────────────────\n")

    # Save outputs
    cv2.imwrite(os.path.join(output_dir, "new_roads.png"),     changes["new_roads"])
    cv2.imwrite(os.path.join(output_dir, "removed_roads.png"), changes["removed_roads"])
    cv2.imwrite(os.path.join(output_dir, "unchanged.png"),     changes["unchanged"])

    with open(os.path.join(output_dir, "change_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    vis_path = os.path.join(output_dir, "change_visualization.png")
    visualize_changes(img_before, img_after_aligned,
                      res_b["binary_mask"], res_a["binary_mask"],
                      changes, metrics, save_path=vis_path)

    return changes, metrics


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python change_detector.py <before.jpg> <after.jpg>")
        sys.exit(1)
    run_change_detection(sys.argv[1], sys.argv[2])
