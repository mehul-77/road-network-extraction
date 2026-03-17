"""
Dataset Acquisition Script
Downloads satellite tiles from OpenStreetMap tile servers
and road vector data using the Overpass API.

Usage:
    python dataset_downloader.py
"""

import os
import math
import json
import time
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────
# TILE MATH  (Web Mercator / Slippy Map)
# ─────────────────────────────────────────────

def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile_bbox(x: int, y: int, zoom: int) -> dict:
    """Return bounding box (lat/lon) for a tile."""
    n = 2 ** zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return {"lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max}


# ─────────────────────────────────────────────
# SATELLITE TILE DOWNLOAD  (ESRI World Imagery)
# ─────────────────────────────────────────────

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def download_tile(x: int, y: int, zoom: int, save_dir: str) -> str:
    url = ESRI_URL.format(z=zoom, y=y, x=x)
    filename = os.path.join(save_dir, f"tile_{zoom}_{x}_{y}.jpg")
    if os.path.exists(filename):
        print(f"  ⏭  Already exists: {filename}")
        return filename
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ITS-RoadExtraction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            with open(filename, "wb") as f:
                f.write(resp.read())
        print(f"  ✅ Downloaded: {filename}")
        time.sleep(0.5)  # be polite to tile server
        return filename
    except Exception as e:
        print(f"  ❌ Failed {url}: {e}")
        return None


def download_area(lat: float, lon: float, zoom: int = 17, grid: int = 2, save_dir: str = "data/tiles"):
    """Download a grid of satellite tiles around a lat/lon centre."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cx, cy = lat_lon_to_tile(lat, lon, zoom)
    paths = []
    half = grid // 2
    print(f"\n📡 Downloading {grid*grid} tile(s) at zoom={zoom} around ({lat}, {lon})")
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            tx, ty = cx + dx, cy + dy
            p = download_tile(tx, ty, zoom, save_dir)
            if p:
                paths.append({"path": p, "bbox": tile_bbox(tx, ty, zoom)})
    # Save index
    index_path = os.path.join(save_dir, "tile_index.json")
    with open(index_path, "w") as f:
        json.dump(paths, f, indent=2)
    print(f"\n✅ Tile index saved → {index_path}")
    return paths


# ─────────────────────────────────────────────
# OSM ROAD VECTOR DATA  (Overpass API)
# ─────────────────────────────────────────────

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def download_osm_roads(lat_min, lat_max, lon_min, lon_max, save_path: str = "data/osm_roads.json"):
    """Fetch road geometries from OpenStreetMap Overpass API."""
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out geom;
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        data = query.encode("utf-8")
        req = urllib.request.Request(
            OVERPASS_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "ITS-RoadExtraction/1.0"}
        )
        print(f"\n🗺  Fetching OSM road data for bbox ({lat_min:.4f},{lon_min:.4f}) → ({lat_max:.4f},{lon_max:.4f})")
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        road_count = len([e for e in result.get("elements", []) if e["type"] == "way"])
        print(f"✅ {road_count} road segments saved → {save_path}")
        return result
    except Exception as e:
        print(f"❌ OSM download failed: {e}")
        return None


# ─────────────────────────────────────────────
# EXAMPLE LOCATIONS  (swap to your city)
# ─────────────────────────────────────────────

LOCATIONS = {
    "surat_india":    (21.1702, 72.8311),
    "mumbai_india":   (19.0760, 72.8777),
    "new_delhi":      (28.6139, 77.2090),
    "london_uk":      (51.5074, -0.1278),
    "new_york_us":    (40.7128, -74.0060),
}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Pick a location
    location = "surat_india"
    lat, lon = LOCATIONS[location]
    zoom = 17           # zoom 17 = ~1.2m/px  (good for road detection)
    grid_size = 2       # downloads a 2×2 tile grid

    # Download satellite tiles
    tiles = download_area(lat, lon, zoom=zoom, grid=grid_size,
                          save_dir=f"data/tiles/{location}")

    # Download OSM road data for the same area
    if tiles:
        all_bboxes = [t["bbox"] for t in tiles]
        lat_min = min(b["lat_min"] for b in all_bboxes)
        lat_max = max(b["lat_max"] for b in all_bboxes)
        lon_min = min(b["lon_min"] for b in all_bboxes)
        lon_max = max(b["lon_max"] for b in all_bboxes)
        download_osm_roads(lat_min, lat_max, lon_min, lon_max,
                           save_path=f"data/osm/{location}_roads.json")

    print("\n🎉 Dataset ready. Run road_extractor.py on any downloaded tile.")
    print("   Example:")
    print(f"   python road_extractor.py data/tiles/{location}/tile_17_*.jpg")
