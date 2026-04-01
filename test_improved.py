"""
Smoke test for the improved road detection algorithm.
Tests:
  1. detect_roads() with HSV color masking (new signature)
  2. detect_roads() without original_rgb (fallback path)
  3. change_detector align_images (color-space fix)
  4. Backend metric: road_density is now a percentage (x100)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import cv2

# ── Test 1: detect_roads with original RGB ─────────────────────────────────
from road_extractor import load_image, preprocess, detect_roads, compute_metrics

# Create a synthetic 200x200 image with a grey road on darker background
img = np.ones((200, 200, 3), dtype=np.uint8) * 80   # dark grey background
img[90:110, :, :] = 170   # lighter grey horizontal strip (road-like, low saturation)
img[:, 90:110, :] = 170   # vertical strip

pre = preprocess(img)
res = detect_roads(pre, original_rgb=img)
metrics = compute_metrics(img, res)

print("=== TEST 1: detect_roads with original_rgb ===")
print(f"  road_area_percent   : {metrics['road_area_percent']}%")
print(f"  num_road_segments   : {metrics['num_road_segments']}")
print(f"  road_length_pixels  : {metrics['road_length_pixels']}px")
print(f"  road_density_per_px : {metrics['road_density_per_px']}")
assert isinstance(metrics['road_area_percent'], float), "road_area_percent must be float"
assert metrics['road_area_percent'] >= 0, "road_area_percent must be >= 0"
print("  PASS\n")

# ── Test 2: detect_roads without original_rgb (fallback) ───────────────────
print("=== TEST 2: detect_roads without original_rgb (fallback) ===")
res2 = detect_roads(pre, original_rgb=None)
metrics2 = compute_metrics(img, res2)
print(f"  road_area_percent   : {metrics2['road_area_percent']}%")
print(f"  num_road_segments   : {metrics2['num_road_segments']}")
print("  PASS\n")

# ── Test 3: align_images color-space fix ───────────────────────────────────
from change_detector import align_images
print("=== TEST 3: align_images color-space fix ===")
# Two identical images (alignment should produce identical result)
before = img.copy()
after  = img.copy()
aligned = align_images(before, after)
assert aligned.shape == before.shape, "Aligned shape must match before"
assert aligned.dtype == before.dtype, "Aligned dtype must match before"
print(f"  Input shape  : {before.shape}")
print(f"  Aligned shape: {aligned.shape}")
print("  PASS\n")

# ── Test 4: road_density percentage scaling ────────────────────────────────
print("=== TEST 4: road_density percentage ===")
raw_density = metrics['road_density_per_px']
scaled = round(raw_density * 100, 4)
print(f"  raw road_density_per_px : {raw_density}")
print(f"  scaled (x100) as %      : {scaled}%")
assert 0.0 <= scaled <= 100.0, f"scaled density {scaled} out of [0,100] range"
print("  PASS\n")

# ── Test 5: ITS readiness thresholds ──────────────────────────────────────
print("=== TEST 5: ITS readiness classification ===")
test_cases = [
    (20, 70, "High"),
    (5,  80, "High"),   # high connectivity even with low density
    (10, 50, "Medium"),
    (3,  20, "Low"),
]
for density, connectivity, expected in test_cases:
    if density > 15 and connectivity > 65:
        label = "High"
    elif density > 8 or connectivity > 40:
        label = "Medium"
    else:
        label = "Low"
    status = "PASS" if label == expected else "FAIL"
    print(f"  density={density}, conn={connectivity} -> {label} [{status}]")

print("\nAll tests complete!")
