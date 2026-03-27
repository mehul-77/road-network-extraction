"""
FastAPI Backend for Road Network Extraction
Combines OpenStreetMap analysis + Classical CV road detection
"""

import os
import sys
import json
import math
import base64
import tempfile
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import numpy as np
import cv2

# ── Import local CV modules ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__) + "/../src")
from road_extractor import load_image, preprocess, detect_roads, compute_metrics
from geojson_exporter import export_roads_to_geojson
from report_generator import generate_report
from change_detector import align_images, detect_changes, change_metrics, build_change_overlay


app = FastAPI(
    title="Road Network Extraction API",
    description="ITS Project — Road network analysis using OSM + satellite imagery CV",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(tempfile.gettempdir()) / "road_extraction"
TEMP_DIR.mkdir(exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def save_upload(file: UploadFile) -> str:
    path = TEMP_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(path)


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bbox_from_center(lat, lon, radius_m):
    """Approximate bounding box from center + radius in meters."""
    d_lat = radius_m / 111320
    d_lon = radius_m / (111320 * math.cos(math.radians(lat)))
    return {
        "lat_min": lat - d_lat,
        "lat_max": lat + d_lat,
        "lon_min": lon - d_lon,
        "lon_max": lon + d_lon,
    }


# ═════════════════════════════════════════════════════════════════════════════
# OSM / OVERPASS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

ROAD_HIERARCHY = {
    "motorway":       {"level": 1, "color": "#e74c3c", "label": "Motorway"},
    "motorway_link":  {"level": 1, "color": "#e74c3c", "label": "Motorway Link"},
    "trunk":          {"level": 2, "color": "#e67e22", "label": "Trunk Road"},
    "trunk_link":     {"level": 2, "color": "#e67e22", "label": "Trunk Link"},
    "primary":        {"level": 3, "color": "#f1c40f", "label": "Primary Road"},
    "primary_link":   {"level": 3, "color": "#f1c40f", "label": "Primary Link"},
    "secondary":      {"level": 4, "color": "#2ecc71", "label": "Secondary Road"},
    "secondary_link": {"level": 4, "color": "#2ecc71", "label": "Secondary Link"},
    "tertiary":       {"level": 5, "color": "#00d4aa", "label": "Tertiary Road"},
    "tertiary_link":  {"level": 5, "color": "#00d4aa", "label": "Tertiary Link"},
    "residential":    {"level": 6, "color": "#3498db", "label": "Residential"},
    "service":        {"level": 7, "color": "#9b59b6", "label": "Service Road"},
    "unclassified":   {"level": 7, "color": "#7f8c8d", "label": "Unclassified"},
    "living_street":  {"level": 7, "color": "#1abc9c", "label": "Living Street"},
    "pedestrian":     {"level": 8, "color": "#95a5a6", "label": "Pedestrian"},
    "footway":        {"level": 8, "color": "#bdc3c7", "label": "Footway"},
    "cycleway":       {"level": 8, "color": "#16a085", "label": "Cycleway"},
    "path":           {"level": 9, "color": "#636e72", "label": "Path"},
    "track":          {"level": 9, "color": "#6c5ce7", "label": "Track"},
}


def fetch_osm_roads(bbox):
    """Fetch roads from Overpass API."""
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
    );
    out geom;
    """
    import urllib.parse
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=data,
        headers={"User-Agent": "ITS-RoadExtraction/2.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def road_length_km(geometry):
    """Calculate road length from a list of {lat, lon} nodes."""
    total = 0.0
    for i in range(1, len(geometry)):
        total += haversine(
            geometry[i - 1]["lat"], geometry[i - 1]["lon"],
            geometry[i]["lat"], geometry[i]["lon"],
        )
    return round(total, 4)


def build_analysis(osm_data, bbox, lat, lon, radius):
    """Process raw Overpass data into the structure the frontend expects."""
    elements = [e for e in osm_data.get("elements", []) if e["type"] == "way" and "geometry" in e]

    # ── Build GeoJSON features ────────────────────────────────────────────
    features = []
    type_totals = {}  # highway_type -> total_length_km
    all_nodes = {}    # node coordinate -> set of way ids (for intersection detection)

    for way in elements:
        highway = way.get("tags", {}).get("highway", "unclassified")
        info = ROAD_HIERARCHY.get(highway, {"level": 7, "color": "#7f8c8d", "label": highway.replace("_", " ").title()})
        geom = way.get("geometry", [])
        length = road_length_km(geom)

        coords = [[pt["lon"], pt["lat"]] for pt in geom]

        # Track nodes for intersection detection
        for pt in geom:
            key = (round(pt["lat"], 6), round(pt["lon"], 6))
            if key not in all_nodes:
                all_nodes[key] = set()
            all_nodes[key].add(way["id"])

        name = way.get("tags", {}).get("name", "")
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "id": way["id"],
                "highway": highway,
                "name": name,
                "label": info["label"],
                "color": info["color"],
                "level": info["level"],
                "length_km": length,
            },
        })

        type_totals[highway] = type_totals.get(highway, 0) + length

    geojson = {"type": "FeatureCollection", "features": features}

    # ── Intersections & dead ends ─────────────────────────────────────────
    intersections = []
    dead_ends_list = []
    for (nlat, nlon), way_ids in all_nodes.items():
        if len(way_ids) >= 3:
            intersections.append([nlat, nlon])
        # Dead ends: nodes that appear exactly once across all ways AND are endpoints
    # Simpler: count node occurrences; if a node appears only in one way and is an endpoint
    endpoint_counts = {}
    for way in elements:
        geom = way.get("geometry", [])
        if len(geom) < 2:
            continue
        for pt in [geom[0], geom[-1]]:
            key = (round(pt["lat"], 6), round(pt["lon"], 6))
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1

    for (nlat, nlon), way_ids in all_nodes.items():
        key = (nlat, nlon)
        if len(way_ids) == 1 and endpoint_counts.get(key, 0) == 1:
            dead_ends_list.append([nlat, nlon])

    # ── Type distribution ─────────────────────────────────────────────────
    total_length = sum(type_totals.values())
    type_distribution = []
    for highway, length in sorted(type_totals.items(), key=lambda x: -x[1]):
        info = ROAD_HIERARCHY.get(highway, {"level": 7, "color": "#7f8c8d", "label": highway.replace("_", " ").title()})
        pct = round((length / total_length * 100) if total_length > 0 else 0, 1)
        type_distribution.append({
            "type": highway,
            "label": info["label"],
            "color": info["color"],
            "percent": pct,
            "length_km": round(length, 2),
        })

    # ── Summary metrics ───────────────────────────────────────────────────
    area_km2 = math.pi * (radius / 1000) ** 2
    int_count = len(intersections)
    de_count = len(dead_ends_list)
    density = round(total_length / area_km2, 1) if area_km2 > 0 else 0
    connectivity = round(int_count / (int_count + de_count) * 100, 1) if (int_count + de_count) > 0 else 0

    # ITS readiness
    if density > 15 and connectivity > 65:
        its = {"label": "High", "desc": f"Dense, well-connected road network ({density} km/km²). Suitable for ITS deployment."}
    elif density > 8 or connectivity > 40:
        its = {"label": "Medium", "desc": f"Moderate road network ({density} km/km²). ITS deployment feasible with improvements."}
    else:
        its = {"label": "Low", "desc": f"Sparse road network ({density} km/km²). Significant infrastructure development needed for ITS."}

    summary = {
        "total_roads": len(elements),
        "total_length_km": round(total_length, 2),
        "road_density_km_km2": density,
        "intersection_count": int_count,
        "dead_end_count": de_count,
        "connectivity_index": connectivity,
        "its_readiness": its,
    }

    # ── Zone grid (for heatmap layer) ─────────────────────────────────────
    grid_n = 6
    lat_step = (bbox["lat_max"] - bbox["lat_min"]) / grid_n
    lon_step = (bbox["lon_max"] - bbox["lon_min"]) / grid_n
    zone_grid = []
    max_cell_length = 0.001  # avoid div-by-zero

    cells = []
    for row in range(grid_n):
        for col in range(grid_n):
            cell_lat_min = bbox["lat_min"] + row * lat_step
            cell_lat_max = cell_lat_min + lat_step
            cell_lon_min = bbox["lon_min"] + col * lon_step
            cell_lon_max = cell_lon_min + lon_step

            cell_length = 0
            for feat in features:
                for coord in feat["geometry"]["coordinates"]:
                    clon, clat = coord
                    if cell_lat_min <= clat <= cell_lat_max and cell_lon_min <= clon <= cell_lon_max:
                        cell_length += feat["properties"]["length_km"] / max(len(feat["geometry"]["coordinates"]), 1)
                        break

            cells.append({
                "bounds": {
                    "lat_min": round(cell_lat_min, 6),
                    "lat_max": round(cell_lat_max, 6),
                    "lon_min": round(cell_lon_min, 6),
                    "lon_max": round(cell_lon_max, 6),
                },
                "length_km": round(cell_length, 3),
            })
            max_cell_length = max(max_cell_length, cell_length)

    for cell in cells:
        cell["normalized"] = round(cell["length_km"] / max_cell_length, 3) if max_cell_length > 0 else 0
    zone_grid = cells

    return {
        "metrics": {
            "summary": summary,
            "type_distribution": type_distribution,
            "geojson": geojson,
            "intersections": intersections[:500],
            "dead_ends": dead_ends_list[:300],
        },
        "zone_grid": zone_grid,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "message": "Road Network Extraction API v2 is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── CITY ANALYSIS (OSM) ─────────────────────────────────────────────────────

@app.get("/analyze")
async def analyze(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius: float = Query(1500, description="Radius in meters"),
):
    """Fetch road network from OpenStreetMap and compute ITS metrics."""
    try:
        bbox = bbox_from_center(lat, lon, radius)
        osm_data = fetch_osm_roads(bbox)
        result = build_analysis(osm_data, bbox, lat, lon, radius)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"OSM analysis failed: {str(e)}")


# ── CV SATELLITE IMAGE DETECTION ────────────────────────────────────────────

@app.post("/cv/detect")
async def cv_detect(
    file: UploadFile = File(...),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    """Upload a satellite image → CV road extraction + optional OSM comparison."""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only JPG/PNG images are supported")

    img_path = save_upload(file)

    try:
        img = load_image(img_path)
        pre = preprocess(img)
        res = detect_roads(pre)
        metrics = compute_metrics(img, res)

        def to_b64(arr: np.ndarray) -> str:
            _, buf = cv2.imencode(".png", arr)
            return base64.b64encode(buf).decode()

        # Original image as base64
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Overlay: skeleton on original
        overlay = img.copy()
        overlay[res["skeleton"] > 0] = [255, 80, 80]
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv_result = {
            "metrics": {
                "road_area_percent": metrics["road_area_percent"],
                "num_segments": metrics["num_road_segments"],
                "road_length_pixels": metrics["road_length_pixels"],
                "road_density": metrics["road_density_per_px"],
            },
            "images": {
                "original": to_b64(img_bgr),
                "mask": to_b64(res["binary_mask"]),
                "skeleton": to_b64(res["skeleton"]),
                "overlay": to_b64(overlay_bgr),
            },
        }

        # Optional OSM comparison
        osm_result = None
        if lat is not None and lon is not None:
            try:
                bbox = bbox_from_center(lat, lon, 800)
                osm_data = fetch_osm_roads(bbox)
                analysis = build_analysis(osm_data, bbox, lat, lon, 800)
                osm_result = {"summary": analysis["metrics"]["summary"]}
            except Exception:
                osm_result = None

        return JSONResponse({"cv": cv_result, "osm": osm_result})

    except Exception as e:
        raise HTTPException(500, str(e))


# ── GEOJSON EXPORT ───────────────────────────────────────────────────────────

@app.get("/geojson")
async def export_geojson_endpoint(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: float = Query(1500),
):
    """Generate GeoJSON from OSM road data for download."""
    try:
        bbox = bbox_from_center(lat, lon, radius)
        osm_data = fetch_osm_roads(bbox)
        analysis = build_analysis(osm_data, bbox, lat, lon, radius)
        geojson = analysis["metrics"]["geojson"]

        output_path = str(TEMP_DIR / "roads.geojson")
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        return FileResponse(
            output_path,
            media_type="application/geo+json",
            filename="roads.geojson",
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── PDF REPORT ───────────────────────────────────────────────────────────────

@app.get("/report")
async def generate_report_endpoint(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: float = Query(1500),
):
    """Generate a PDF infrastructure report from OSM data."""
    try:
        bbox = bbox_from_center(lat, lon, radius)
        osm_data = fetch_osm_roads(bbox)
        analysis = build_analysis(osm_data, bbox, lat, lon, radius)
        s = analysis["metrics"]["summary"]

        output_path = str(TEMP_DIR / "its_report.pdf")

        # Build a simple PDF report using reportlab
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from datetime import datetime

        BLUE = colors.HexColor("#1a73e8")
        DARK = colors.HexColor("#0d1117")
        GREY = colors.HexColor("#5f6368")
        BORDER = colors.HexColor("#dadce0")
        LIGHT_BLUE = colors.HexColor("#e8f0fe")

        base_styles = getSampleStyleSheet()
        styles = {
            "title": ParagraphStyle("title", parent=base_styles["Title"], fontSize=22, textColor=DARK, fontName="Helvetica-Bold", alignment=TA_LEFT),
            "subtitle": ParagraphStyle("subtitle", parent=base_styles["Normal"], fontSize=11, textColor=GREY, spaceAfter=20),
            "section": ParagraphStyle("section", parent=base_styles["Heading1"], fontSize=13, textColor=BLUE, spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold"),
            "body": ParagraphStyle("body", parent=base_styles["Normal"], fontSize=10, textColor=DARK, spaceAfter=6, leading=15),
            "footer": ParagraphStyle("footer", parent=base_styles["Normal"], fontSize=8, textColor=GREY, alignment=TA_CENTER),
        }

        doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2.5*cm)
        story = []
        now = datetime.now().strftime("%d %B %Y, %H:%M")

        story.append(Paragraph("Road Network Infrastructure Report", styles["title"]))
        story.append(Paragraph(f"ITS Analysis  ·  {now}  ·  ({lat}, {lon}) radius {radius}m", styles["subtitle"]))
        story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=16))

        story.append(Paragraph("1. Network Summary", styles["section"]))
        story.append(Paragraph(
            f"Analysis of the road network within {radius}m of ({lat:.4f}, {lon:.4f}) "
            f"using OpenStreetMap data.", styles["body"]
        ))

        td = analysis["metrics"]["type_distribution"]

        table_data = [
            ["Metric", "Value"],
            ["Total Roads", str(s["total_roads"])],
            ["Total Length", f"{s['total_length_km']} km"],
            ["Road Density", f"{s['road_density_km_km2']} km/km²"],
            ["Intersections", str(s["intersection_count"])],
            ["Dead Ends", str(s["dead_end_count"])],
            ["Connectivity Index", f"{s['connectivity_index']}%"],
            ["ITS Readiness", s["its_readiness"]["label"]],
        ]
        t = Table(table_data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
            ("BOX", (0, 0), (-1, -1), 1, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("2. Road Type Distribution", styles["section"]))
        type_table = [["Road Type", "Length (km)", "Percentage"]]
        for item in td[:10]:
            type_table.append([item["label"], str(item["length_km"]), f"{item['percent']}%"])
        t2 = Table(type_table, colWidths=[6*cm, 5*cm, 5*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
            ("BOX", (0, 0), (-1, -1), 1, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("3. ITS Readiness Assessment", styles["section"]))
        story.append(Paragraph(f"<b>{s['its_readiness']['label']}</b>: {s['its_readiness']['desc']}", styles["body"]))
        story.append(Spacer(1, 1*cm))

        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
        story.append(Paragraph(f"Generated by ITS Road Network Analysis Tool  ·  NIT Surat  ·  {now}", styles["footer"]))

        doc.build(story)

        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename="its_report.pdf",
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── CHANGE DETECTION (keep from original) ────────────────────────────────────

@app.post("/change")
async def change_detection(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
):
    """Upload before + after images → get change metrics + colour overlay."""
    before_path = save_upload(before)
    after_path = save_upload(after)

    try:
        img_b = load_image(before_path)
        img_a = load_image(after_path)
        img_a = align_images(img_b, img_a)

        res_b = detect_roads(preprocess(img_b))
        res_a = detect_roads(preprocess(img_a))

        changes = detect_changes(res_b["binary_mask"], res_a["binary_mask"])
        metrics = change_metrics(changes, img_b.shape[0] * img_b.shape[1])

        overlay = build_change_overlay(img_b, changes)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", overlay_bgr)
        overlay_b64 = base64.b64encode(buf).decode()

        return JSONResponse({"metrics": metrics, "overlay": overlay_b64})
    except Exception as e:
        raise HTTPException(500, str(e))


# ── LEGACY CV ENDPOINTS (kept for backwards compatibility) ───────────────────

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Legacy: Upload a satellite image → get road metrics + base64 result images."""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only JPG/PNG images are supported")

    img_path = save_upload(file)
    try:
        img = load_image(img_path)
        pre = preprocess(img)
        res = detect_roads(pre)
        metrics = compute_metrics(img, res)

        def to_b64(arr: np.ndarray) -> str:
            _, buf = cv2.imencode(".png", arr)
            return base64.b64encode(buf).decode()

        overlay = img.copy()
        overlay[res["skeleton"] > 0] = [255, 80, 80]
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        return JSONResponse({
            "metrics": metrics,
            "images": {
                "road_mask": to_b64(res["binary_mask"]),
                "skeleton": to_b64(res["skeleton"]),
                "overlay": to_b64(overlay_bgr),
            },
        })
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
