"""
PDF Report Generator
Auto-generates a professional infrastructure analysis report
from road extraction results.

Usage:
    python report_generator.py <image_path> [--bbox bbox.json] [--change-dir output/change/]
"""

import os
import sys
import json
import tempfile
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak
)

sys.path.insert(0, os.path.dirname(__file__))
from road_extractor import load_image, preprocess, detect_roads, compute_metrics


# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────

DARK        = colors.HexColor("#0d1117")
BLUE        = colors.HexColor("#1a73e8")
LIGHT_BLUE  = colors.HexColor("#e8f0fe")
GREEN       = colors.HexColor("#34a853")
RED         = colors.HexColor("#ea4335")
GREY        = colors.HexColor("#5f6368")
WHITE       = colors.white
LIGHT_GREY  = colors.HexColor("#f8f9fa")
BORDER      = colors.HexColor("#dadce0")


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=22, textColor=DARK,
            spaceAfter=4, fontName="Helvetica-Bold",
            alignment=TA_LEFT
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, textColor=GREY,
            spaceAfter=20, fontName="Helvetica"
        ),
        "section": ParagraphStyle(
            "section", parent=base["Heading1"],
            fontSize=13, textColor=BLUE,
            spaceBefore=18, spaceAfter=8,
            fontName="Helvetica-Bold",
            borderPad=2
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, textColor=DARK,
            spaceAfter=6, fontName="Helvetica",
            leading=15
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=8.5, textColor=GREY,
            spaceAfter=12, fontName="Helvetica-Oblique",
            alignment=TA_CENTER
        ),
        "metric_label": ParagraphStyle(
            "metric_label", parent=base["Normal"],
            fontSize=9, textColor=GREY,
            fontName="Helvetica"
        ),
        "metric_value": ParagraphStyle(
            "metric_value", parent=base["Normal"],
            fontSize=15, textColor=BLUE,
            fontName="Helvetica-Bold", alignment=TA_CENTER
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontSize=8, textColor=GREY,
            fontName="Helvetica", alignment=TA_CENTER
        ),
    }
    return styles


# ─────────────────────────────────────────────
# HELPER: save matplotlib figure to temp file
# ─────────────────────────────────────────────

def fig_to_temp(fig, dpi=120) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return tmp.name


# ─────────────────────────────────────────────
# FIGURE: side-by-side original + road mask
# ─────────────────────────────────────────────

def make_detection_figure(img, mask, skeleton) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#f8f9fa")
    labels = ["Original Satellite Image", "Road Binary Mask", "Road Centerlines (Skeleton)"]
    datas  = [img, mask, skeleton]
    cmaps  = [None, "gray", "hot"]
    for ax, data, label, cmap in zip(axes, datas, labels, cmaps):
        ax.imshow(data, cmap=cmap)
        ax.set_title(label, fontsize=10, pad=6, color="#0d1117")
        ax.axis("off")
    fig.tight_layout()
    return fig_to_temp(fig)


# ─────────────────────────────────────────────
# FIGURE: road density bar chart
# ─────────────────────────────────────────────

def make_density_chart(metrics: dict) -> str:
    labels = ["Road Area %", "Road Density\n(×10⁻⁴)"]
    values = [
        metrics["road_area_percent"],
        metrics["road_density_per_px"] * 10000
    ]
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    bars = ax.bar(labels, values, color=["#1a73e8", "#34a853"], width=0.4, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#0d1117")
    ax.set_ylabel("Value", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig_to_temp(fig)


# ─────────────────────────────────────────────
# METRIC CARD TABLE
# ─────────────────────────────────────────────

def make_metric_table(metrics: dict, styles: dict) -> Table:
    data = [
        [
            Paragraph("ROAD AREA", styles["metric_label"]),
            Paragraph("ROAD LENGTH", styles["metric_label"]),
            Paragraph("ROAD DENSITY", styles["metric_label"]),
            Paragraph("SEGMENTS", styles["metric_label"]),
        ],
        [
            Paragraph(f"{metrics['road_area_percent']}%", styles["metric_value"]),
            Paragraph(f"{metrics['road_length_pixels']:,} px", styles["metric_value"]),
            Paragraph(f"{metrics['road_density_per_px']:.5f}", styles["metric_value"]),
            Paragraph(str(metrics["num_road_segments"]), styles["metric_value"]),
        ],
    ]
    t = Table(data, colWidths=[4*cm]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), LIGHT_BLUE),
        ("BACKGROUND", (0,1), (-1,1), WHITE),
        ("BOX",        (0,0), (-1,-1), 1, BORDER),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, BORDER),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_BLUE, WHITE]),
    ]))
    return t


# ─────────────────────────────────────────────
# CHANGE SECTION (optional)
# ─────────────────────────────────────────────

def make_change_section(change_dir: str, styles: dict) -> list:
    story = []
    metrics_path = os.path.join(change_dir, "change_metrics.json")
    vis_path     = os.path.join(change_dir, "change_visualization.png")

    if not os.path.exists(metrics_path):
        return []

    with open(metrics_path) as f:
        cm = json.load(f)

    story.append(Paragraph("4. Change Detection Analysis", styles["section"]))
    story.append(Paragraph(
        "The following section compares road networks across two time periods "
        "to identify infrastructure development and degradation.",
        styles["body"]
    ))

    change_data = [
        ["Metric", "Value"],
        ["New Roads Added",    f"{cm['new_road_area_pct']}%"],
        ["Roads Removed",      f"{cm['removed_road_area_pct']}%"],
        ["Net Change",         f"{cm['net_change_pct']:+.3f}%"],
        ["Unchanged Roads",    f"{cm['unchanged_road_pixels']:,} px"],
    ]
    t = Table(change_data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ("BOX",         (0,0), (-1,-1), 1, BORDER),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, BORDER),
        ("ALIGN",       (0,1), (-1,-1), "CENTER"),
        ("TOPPADDING",  (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    if os.path.exists(vis_path):
        story.append(RLImage(vis_path, width=16*cm, height=9*cm))
        story.append(Paragraph("Figure 2: Road change detection overlay — Green: new roads, Red: removed roads, White: unchanged.", styles["caption"]))

    return story


# ─────────────────────────────────────────────
# MAIN REPORT BUILDER
# ─────────────────────────────────────────────

def generate_report(
    image_path: str,
    metrics: dict,
    mask: np.ndarray,
    skeleton: np.ndarray,
    original_img: np.ndarray,
    bbox: dict = None,
    change_dir: str = None,
    output_path: str = "output/report.pdf"
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2.5*cm,
        title="Road Network Extraction Report",
        author="ITS Road Extraction Tool"
    )

    story = []
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    img_name = Path(image_path).name

    # ── HEADER ────────────────────────────────
    story.append(Paragraph("Road Network Feature Extraction", styles["title"]))
    story.append(Paragraph(f"Infrastructure Analysis Report  ·  {now}", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=16))

    # ── SECTION 1: OVERVIEW ───────────────────
    story.append(Paragraph("1. Overview", styles["section"]))
    story.append(Paragraph(
        f"This report presents the results of automated road network extraction "
        f"applied to satellite imagery (<b>{img_name}</b>) using classical image-processing "
        f"techniques including edge detection, adaptive thresholding, and morphological "
        f"operations. The extracted features are intended to support Intelligent "
        f"Transportation System (ITS) applications and urban infrastructure analysis.",
        styles["body"]
    ))

    if bbox:
        story.append(Paragraph(
            f"<b>Area of Interest:</b> "
            f"Lat [{bbox['lat_min']:.5f}, {bbox['lat_max']:.5f}], "
            f"Lon [{bbox['lon_min']:.5f}, {bbox['lon_max']:.5f}]",
            styles["body"]
        ))

    # ── SECTION 2: METRICS ────────────────────
    story.append(Paragraph("2. Infrastructure Metrics", styles["section"]))
    story.append(make_metric_table(metrics, styles))
    story.append(Spacer(1, 0.5*cm))

    # Metrics table detail
    detail_data = [
        ["Metric", "Value", "Description"],
        ["Image Size",       metrics["image_size"],              "Width × Height in pixels"],
        ["Total Pixels",     f"{metrics['total_pixels']:,}",     "Total image area"],
        ["Road Pixels",      f"{metrics['road_pixels']:,}",      "Pixels classified as road"],
        ["Road Area %",      f"{metrics['road_area_percent']}%", "Road coverage ratio"],
        ["Road Length",      f"{metrics['road_length_pixels']:,} px", "Skeleton centerline length"],
        ["Road Density",     str(metrics["road_density_per_px"]), "Length per total pixel"],
        ["Road Segments",    str(metrics["num_road_segments"]),  "Connected road components"],
    ]
    t = Table(detail_data, colWidths=[5*cm, 4*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",      (0,0), (-1,0), WHITE),
        ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ("BOX",            (0,0), (-1,-1), 1, BORDER),
        ("INNERGRID",      (0,0), (-1,-1), 0.5, BORDER),
        ("ALIGN",          (1,1), (1,-1), "CENTER"),
        ("TOPPADDING",     (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    # ── SECTION 3: VISUALIZATIONS ─────────────
    story.append(Paragraph("3. Detection Results", styles["section"]))
    fig_path = make_detection_figure(original_img, mask, skeleton)
    story.append(RLImage(fig_path, width=16*cm, height=5.5*cm))
    story.append(Paragraph("Figure 1: (Left) Original satellite image. (Centre) Binary road mask. (Right) Extracted road centerlines.", styles["caption"]))

    chart_path = make_density_chart(metrics)
    story.append(RLImage(chart_path, width=8*cm, height=5*cm))
    story.append(Paragraph("Figure 2: Road area percentage and density metric.", styles["caption"]))

    # ── SECTION 4: CHANGE DETECTION (optional) ─
    if change_dir and os.path.isdir(change_dir):
        story.extend(make_change_section(change_dir, styles))

    # ── SECTION 5: METHODOLOGY ────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Methodology", styles["section"]))
    story.append(Paragraph(
        "The pipeline applies the following stages to each input satellite image:",
        styles["body"]
    ))
    steps = [
        ("Preprocessing", "Colour-to-grayscale conversion, bilateral filtering for edge-preserving "
         "noise removal, and CLAHE (Contrast-Limited Adaptive Histogram Equalization) for local contrast enhancement."),
        ("Edge Detection", "Canny edge detector applied with thresholds calibrated for satellite "
         "imagery to identify high-gradient road boundaries."),
        ("Adaptive Thresholding", "Gaussian adaptive thresholding isolates bright road surfaces "
         "relative to their local neighbourhood."),
        ("Morphological Processing", "Dilation to connect broken road segments, morphological "
         "closing to fill small gaps, and connected-component filtering to remove noise blobs."),
        ("Skeletonization", "Iterative thinning produces single-pixel-wide road centerlines "
         "representative of the road network topology."),
        ("GeoJSON Export", "Pixel coordinates are transformed to geographic coordinates (WGS84) "
         "using the tile bounding box, and exported as GeoJSON for GIS compatibility."),
    ]
    for title_s, desc in steps:
        story.append(Paragraph(f"<b>{title_s}:</b> {desc}", styles["body"]))

    # ── SECTION 6: LIMITATIONS ────────────────
    story.append(Paragraph("6. Limitations", styles["section"]))
    story.append(Paragraph(
        "This system uses classical computer vision and does not employ machine learning. "
        "Accuracy may be reduced in areas with: (1) dense tree canopy covering roads, "
        "(2) roads with similar spectral signatures to surrounding terrain, "
        "(3) very low image resolution, or (4) complex urban road intersections. "
        "Results should be validated against OpenStreetMap ground truth before operational use.",
        styles["body"]
    ))

    # ── FOOTER ────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by Road Network Extraction Tool  ·  ITS Project  ·  NIT Surat  ·  {now}",
        styles["footer"]
    ))

    doc.build(story)
    print(f"✅ Report saved → {output_path}")

    # Cleanup temp files
    for p in [fig_path, chart_path]:
        if p and os.path.exists(p):
            os.unlink(p)

    return output_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ITS road extraction PDF report")
    parser.add_argument("image_path", help="Path to satellite image")
    parser.add_argument("--bbox",       default=None, help="Path to bbox JSON file")
    parser.add_argument("--change-dir", default=None, help="Path to change detection output dir")
    parser.add_argument("--output",     default="output/report.pdf", help="Output PDF path")
    args = parser.parse_args()

    bbox = None
    if args.bbox and os.path.exists(args.bbox):
        with open(args.bbox) as f:
            bbox = json.load(f)

    print(f"📡 Loading: {args.image_path}")
    img = load_image(args.image_path)
    pre = preprocess(img)
    res = detect_roads(pre)
    metrics = compute_metrics(img, res)

    generate_report(
        image_path=args.image_path,
        metrics=metrics,
        mask=res["binary_mask"],
        skeleton=res["skeleton"],
        original_img=img,
        bbox=bbox,
        change_dir=args.change_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
