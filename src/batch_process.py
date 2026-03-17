"""
Batch Processor
Runs the road extraction pipeline on all tiles in a folder
and aggregates metrics across tiles.

Usage:
    python batch_process.py data/tiles/surat_india/
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from road_extractor import load_image, preprocess, detect_roads, compute_metrics

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def batch_run(tile_dir: str, output_dir: str = "output/batch"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tile_paths = [
        p for p in glob.glob(os.path.join(tile_dir, "*"))
        if os.path.splitext(p)[1].lower() in SUPPORTED_EXT
    ]

    if not tile_paths:
        print(f"❌ No image files found in {tile_dir}")
        return

    print(f"📂 Found {len(tile_paths)} tile(s) in {tile_dir}\n")
    all_metrics = []

    for idx, path in enumerate(tile_paths, 1):
        name = Path(path).stem
        print(f"[{idx}/{len(tile_paths)}] Processing: {name}")
        try:
            img = load_image(path)
            pre = preprocess(img)
            res = detect_roads(pre)
            m = compute_metrics(img, res)
            m["tile"] = name
            all_metrics.append(m)

            # Save individual mask
            import cv2
            mask_out = os.path.join(output_dir, f"{name}_mask.png")
            cv2.imwrite(mask_out, res["binary_mask"])
        except Exception as e:
            print(f"  ⚠ Skipped {name}: {e}")

    if not all_metrics:
        print("No tiles processed successfully.")
        return

    # Aggregate
    avg_density = np.mean([m["road_density_per_px"] for m in all_metrics])
    avg_area    = np.mean([m["road_area_percent"] for m in all_metrics])
    total_segs  = sum(m["num_road_segments"] for m in all_metrics)

    summary = {
        "tiles_processed": len(all_metrics),
        "avg_road_area_percent": round(float(avg_area), 2),
        "avg_road_density": round(float(avg_density), 6),
        "total_road_segments": int(total_segs),
        "per_tile": all_metrics,
    }

    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── BATCH SUMMARY ────────────────────────────")
    print(f"  Tiles processed      : {summary['tiles_processed']}")
    print(f"  Avg Road Area        : {summary['avg_road_area_percent']}%")
    print(f"  Avg Road Density     : {summary['avg_road_density']}")
    print(f"  Total Road Segments  : {summary['total_road_segments']}")
    print(f"  Summary saved → {summary_path}")
    print("──────────────────────────────────────────────")


if __name__ == "__main__":
    tile_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tiles/surat_india"
    batch_run(tile_dir)
