"""
Basic unit tests for the road extraction pipeline.
Run with: pytest tests/
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from road_extractor import preprocess, detect_roads, compute_metrics, remove_small_components, skeletonize


def make_dummy_image(h=256, w=256):
    """Create a dummy grayscale-ish RGB satellite image with a white road stripe."""
    img = np.random.randint(60, 120, (h, w, 3), dtype=np.uint8)
    # Horizontal road stripe
    img[h//2 - 5 : h//2 + 5, :] = 200
    return img


def test_preprocess_keys():
    img = make_dummy_image()
    result = preprocess(img)
    assert "gray" in result
    assert "denoised" in result
    assert "enhanced" in result


def test_preprocess_shapes():
    img = make_dummy_image(128, 128)
    result = preprocess(img)
    assert result["gray"].shape == (128, 128)
    assert result["enhanced"].shape == (128, 128)


def test_detect_roads_keys():
    img = make_dummy_image()
    pre = preprocess(img)
    result = detect_roads(pre)
    for key in ["edges", "binary_mask", "skeleton"]:
        assert key in result, f"Missing key: {key}"


def test_binary_mask_dtype():
    img = make_dummy_image()
    pre = preprocess(img)
    result = detect_roads(pre)
    assert result["binary_mask"].dtype == np.uint8
    assert set(np.unique(result["binary_mask"])).issubset({0, 255})


def test_metrics_keys():
    img = make_dummy_image()
    pre = preprocess(img)
    result = detect_roads(pre)
    metrics = compute_metrics(img, result)
    for key in ["road_area_percent", "road_length_pixels", "road_density_per_px", "num_road_segments"]:
        assert key in metrics


def test_road_area_range():
    img = make_dummy_image()
    pre = preprocess(img)
    result = detect_roads(pre)
    metrics = compute_metrics(img, result)
    assert 0 <= metrics["road_area_percent"] <= 100


def test_remove_small_components():
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[10:12, 10:12] = 255   # tiny blob — should be removed
    binary[40:70, 40:70] = 255   # large blob — should survive
    cleaned = remove_small_components(binary, min_area=200)
    assert cleaned[10, 10] == 0     # small removed
    assert cleaned[50, 50] == 255   # large kept
