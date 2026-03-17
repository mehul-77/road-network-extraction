"""
FastAPI Backend for Road Network Extraction
Deploy on Render (free tier)
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__) + "/../src")
from road_extractor import load_image, preprocess, detect_roads, compute_metrics
from geojson_exporter import export_roads_to_geojson
from report_generator import generate_report

app = FastAPI(
    title="Road Network Extraction API",
    description="ITS Project — Automated road detection from satellite imagery",
    version="1.0.0"
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


def save_upload(file: UploadFile) -> str:
    path = TEMP_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(path)


@app.get("/")
def root():
    return {"status": "ok", "message": "Road Network Extraction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ─────────────────────────────────────────────
# DETECT endpoint — main pipeline
# ─────────────────────────────────────────────

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Upload a satellite image → get road metrics + base64 result images."""
    import base64

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only JPG/PNG images are supported")

    img_path = save_upload(file)

    try:
        img     = load_image(img_path)
        pre     = preprocess(img)
        res     = detect_roads(pre)
        metrics = compute_metrics(img, res)

        def to_b64(arr: np.ndarray) -> str:
            _, buf = cv2.imencode(".png", arr)
            return base64.b64encode(buf).decode()

        # Overlay: skeleton on original
        overlay = img.copy()
        overlay[res["skeleton"] > 0] = [255, 80, 80]
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        return JSONResponse({
            "metrics": metrics,
            "images": {
                "road_mask":  to_b64(res["binary_mask"]),
                "skeleton":   to_b64(res["skeleton"]),
                "overlay":    to_b64(overlay_bgr),
            }
        })

    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# GEOJSON export endpoint
# ─────────────────────────────────────────────

@app.post("/export/geojson")
async def export_geojson(
    file: UploadFile = File(...),
    lat_min: float = 21.16,
    lat_max: float = 21.18,
    lon_min: float = 72.82,
    lon_max: float = 72.84
):
    """Upload image → download roads.geojson"""
    img_path   = save_upload(file)
    output_path = str(TEMP_DIR / "roads.geojson")
    bbox = {"lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max}
    try:
        img = load_image(img_path)
        pre = preprocess(img)
        res = detect_roads(pre)
        export_roads_to_geojson(res["binary_mask"], res["skeleton"], bbox, output_path)
        return FileResponse(output_path, media_type="application/geo+json",
                            filename="roads.geojson")
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# PDF report endpoint
# ─────────────────────────────────────────────

@app.post("/export/report")
async def export_report(file: UploadFile = File(...)):
    """Upload image → download PDF report"""
    img_path    = save_upload(file)
    output_path = str(TEMP_DIR / "report.pdf")
    try:
        img     = load_image(img_path)
        pre     = preprocess(img)
        res     = detect_roads(pre)
        metrics = compute_metrics(img, res)
        generate_report(
            image_path=img_path,
            metrics=metrics,
            mask=res["binary_mask"],
            skeleton=res["skeleton"],
            original_img=img,
            output_path=output_path
        )
        return FileResponse(output_path, media_type="application/pdf",
                            filename="road_report.pdf")
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# CHANGE detection endpoint
# ─────────────────────────────────────────────

@app.post("/change")
async def change_detection(
    before: UploadFile = File(...),
    after:  UploadFile = File(...)
):
    """Upload before + after images → get change metrics + colour overlay"""
    import base64
    sys.path.insert(0, os.path.dirname(__file__) + "/../src")
    from change_detector import align_images, detect_changes, change_metrics, build_change_overlay

    before_path = save_upload(before)
    after_path  = save_upload(after)

    try:
        img_b = load_image(before_path)
        img_a = load_image(after_path)
        img_a = align_images(img_b, img_a)

        res_b = detect_roads(preprocess(img_b))
        res_a = detect_roads(preprocess(img_a))

        changes = detect_changes(res_b["binary_mask"], res_a["binary_mask"])
        metrics = change_metrics(changes, img_b.shape[0] * img_b.shape[1])

        overlay     = build_change_overlay(img_b, changes)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", overlay_bgr)
        overlay_b64 = base64.b64encode(buf).decode()

        return JSONResponse({"metrics": metrics, "overlay": overlay_b64})
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
