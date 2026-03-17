"""
GeoJSON Exporter
Converts the binary road mask into real geographic coordinates
and exports as GeoJSON — usable in QGIS, Google Maps, Leaflet, etc.

Usage:
    from geojson_exporter import export_roads_to_geojson
    export_roads_to_geojson(mask, tile_bbox, output_path="roads.geojson")
"""

import json
import numpy as np
import cv2
from pathlib import Path


# ─────────────────────────────────────────────
# PIXEL → LAT/LON CONVERSION
# ─────────────────────────────────────────────

def pixel_to_latlon(px: int, py: int, img_w: int, img_h: int, bbox: dict) -> tuple[float, float]:
    """
    Convert a pixel coordinate to lat/lon using the tile bounding box.
    bbox = {"lat_min", "lat_max", "lon_min", "lon_max"}
    """
    lon = bbox["lon_min"] + (px / img_w) * (bbox["lon_max"] - bbox["lon_min"])
    lat = bbox["lat_max"] - (py / img_h) * (bbox["lat_max"] - bbox["lat_min"])
    return round(lat, 7), round(lon, 7)


# ─────────────────────────────────────────────
# CONTOUR → POLYGON FEATURE
# ─────────────────────────────────────────────

def contour_to_geojson_feature(contour: np.ndarray, img_w: int, img_h: int,
                                bbox: dict, segment_id: int) -> dict:
    """Convert an OpenCV contour to a GeoJSON Polygon feature."""
    coords = []
    for point in contour:
        px, py = int(point[0][0]), int(point[0][1])
        lat, lon = pixel_to_latlon(px, py, img_w, img_h, bbox)
        coords.append([lon, lat])  # GeoJSON uses [lon, lat] order

    # Close the ring
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords]
        },
        "properties": {
            "segment_id": segment_id,
            "type": "road",
            "source": "automated_extraction"
        }
    }


# ─────────────────────────────────────────────
# SKELETON → LINESTRING FEATURES
# ─────────────────────────────────────────────

def skeleton_to_linestrings(skeleton: np.ndarray, img_w: int, img_h: int,
                             bbox: dict, min_points: int = 5) -> list[dict]:
    """
    Extract road centerlines from skeleton as GeoJSON LineString features.
    Uses connected component analysis to trace individual road lines.
    """
    features = []
    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) < min_points:
            continue

        # Sort points to form a coherent line (simple heuristic: sort by x then y)
        pts = sorted(zip(xs.tolist(), ys.tolist()), key=lambda p: (p[0], p[1]))

        coords = []
        for px, py in pts:
            lat, lon = pixel_to_latlon(px, py, img_w, img_h, bbox)
            coords.append([lon, lat])

        if len(coords) < 2:
            continue

        # Approximate length in km
        length_km = estimate_length_km(coords)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "road_id": label_id,
                "type": "road_centerline",
                "length_km": round(length_km, 4),
                "source": "skeleton_extraction"
            }
        })

    return features


def estimate_length_km(coords: list) -> float:
    """Approximate road length using Haversine formula between consecutive points."""
    import math
    total = 0.0
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i-1]
        lon2, lat2 = coords[i]
        # Haversine
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        total += R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return total


# ─────────────────────────────────────────────
# MAIN EXPORT FUNCTION
# ─────────────────────────────────────────────

def export_roads_to_geojson(
    mask: np.ndarray,
    skeleton: np.ndarray,
    bbox: dict,
    output_path: str = "output/roads.geojson",
    min_contour_area: int = 100
) -> dict:
    """
    Export detected roads to GeoJSON with both Polygon (mask) and
    LineString (centerline) features.

    Parameters
    ----------
    mask      : binary road mask (uint8, 0/255)
    skeleton  : road centerlines (uint8, 0/255)
    bbox      : {"lat_min", "lat_max", "lon_min", "lon_max"}
    output_path : where to save the .geojson file
    """
    img_h, img_w = mask.shape

    features = []

    # --- Road polygons from mask contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        feat = contour_to_geojson_feature(cnt, img_w, img_h, bbox, segment_id=i)
        features.append(feat)

    # --- Road centerlines from skeleton ---
    line_features = skeleton_to_linestrings(skeleton, img_w, img_h, bbox)
    features.extend(line_features)

    # --- Compute summary stats ---
    total_length_km = sum(
        f["properties"]["length_km"]
        for f in line_features
        if "length_km" in f["properties"]
    )

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "metadata": {
            "bbox": bbox,
            "total_road_polygons": len(contours),
            "total_centerline_segments": len(line_features),
            "total_road_length_km": round(total_length_km, 3),
            "generated_by": "road-network-extraction"
        },
        "features": features
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"✅ GeoJSON saved → {output_path}")
    print(f"   Road polygons     : {len(contours)}")
    print(f"   Centerline segments: {len(line_features)}")
    print(f"   Total road length  : {total_length_km:.3f} km")

    return geojson


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json as _json
    sys.path.insert(0, ".")
    from road_extractor import load_image, preprocess, detect_roads

    if len(sys.argv) < 3:
        print("Usage: python geojson_exporter.py <image_path> <bbox_json_path>")
        print("  bbox_json: {\"lat_min\":21.16, \"lat_max\":21.18, \"lon_min\":72.82, \"lon_max\":72.84}")
        sys.exit(1)

    img_path = sys.argv[1]
    bbox_path = sys.argv[2]

    with open(bbox_path) as f:
        bbox = _json.load(f)

    img = load_image(img_path)
    pre = preprocess(img)
    res = detect_roads(pre)

    export_roads_to_geojson(
        mask=res["binary_mask"],
        skeleton=res["skeleton"],
        bbox=bbox,
        output_path="output/roads.geojson"
    )
