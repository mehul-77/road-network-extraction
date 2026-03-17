# 🛣️ Road Network Feature Extraction from Satellite Imagery

<p align="center">
  <img src="assets/banner.png" alt="Road Network Extraction" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/ITS-Project-orange" />
  <img src="https://img.shields.io/badge/No%20GPU-Required-brightgreen" />
</p>

> An automated image-processing pipeline to extract road networks from satellite imagery and compute infrastructure density metrics — built for **Intelligent Transportation System (ITS)** applications.

---

## 📌 Overview

Manual extraction of road infrastructure from satellite imagery is slow and error-prone. This project automates the process using **classical computer vision** — no deep learning, no GPU required.

Given a satellite image, the system:
- Detects road regions via edge detection + adaptive thresholding
- Generates a **binary road mask**
- Extracts **road centerlines** via skeletonization
- Computes quantitative **ITS infrastructure metrics**

---

## 🖼️ Pipeline

```
Satellite Image
      │
      ▼
 Grayscale + Bilateral Filter   ← denoise, preserve edges
      │
      ▼
 CLAHE Enhancement              ← adaptive contrast boost
      │
      ├──► Canny Edge Detection
      ├──► Adaptive Thresholding
      │
      ▼
 Combine → Morphological Close → Remove Noise Blobs
      │
      ▼
 Binary Road Mask
      │
      ▼
 Skeletonize → Centerlines
      │
      ▼
 Metrics + Visualization
```

---

## 📊 Output Metrics

| Metric | Description |
|---|---|
| `road_area_percent` | % of image area covered by roads |
| `road_length_pixels` | Total centerline length (skeleton pixels) |
| `road_density_per_px` | Road length ÷ total image area |
| `num_road_segments` | Count of connected road components |

---

## ✨ Features

| Feature | Script | What it does |
|---|---|---|
| Road detection | `road_extractor.py` | Binary mask + centerlines + metrics |
| Data download | `dataset_downloader.py` | Satellite tiles + OSM road vectors |
| Batch processing | `batch_process.py` | Process entire tile folders |
| **GeoJSON export** | `geojson_exporter.py` | Converts roads to real lat/lon coordinates usable in QGIS / Google Maps |
| **Change detection** | `change_detector.py` | Before/after diff — highlights new & removed roads |
| **PDF report** | `report_generator.py` | Auto-generates a professional infrastructure analysis report |

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/road-network-extraction.git
cd road-network-extraction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download satellite data
```bash
python src/dataset_downloader.py
```
Downloads satellite tiles from **ESRI World Imagery** (free, no API key) and road vectors from **OpenStreetMap**. Default location: Surat, India — edit `LOCATIONS` in the script to change.

### 4. Run on a single tile
```bash
python src/road_extractor.py data/tiles/surat_india/tile_17_96102_57098.jpg
```
Outputs to `output/`:
- `road_mask.png` — binary road mask
- `visualization.png` — 8-panel analysis
- `metrics.json` — JSON metrics

### 5. Batch process a folder
```bash
python src/batch_process.py data/tiles/surat_india/
```
Outputs aggregated `batch_summary.json`.

### 5. Export roads to GeoJSON (real coordinates)
```bash
python src/geojson_exporter.py data/tiles/surat_india/tile.jpg data/tiles/surat_india/tile_index.json
```
Output: `output/roads.geojson` — open in QGIS, Google My Maps, or Leaflet.

### 6. Before / After change detection
```bash
python src/change_detector.py data/tiles/area_2023.jpg data/tiles/area_2025.jpg
```
Output: `output/change/` — colour-coded diff showing new roads (green) and removed roads (red).

### 7. Generate PDF report
```bash
python src/report_generator.py data/tiles/tile.jpg --change-dir output/change/ --output output/report.pdf
```
Output: A professional multi-page PDF with metrics, figures, methodology, and change analysis.

---

```
road-network-extraction/
├── src/
│   ├── road_extractor.py       ← Core pipeline
│   ├── dataset_downloader.py   ← Tile + OSM data fetcher
│   ├── batch_process.py        ← Multi-tile processing
│   ├── geojson_exporter.py     ← Export roads to real geo coordinates
│   ├── change_detector.py      ← Before/after road change detection
│   └── report_generator.py     ← Auto-generate PDF report
├── data/
│   └── sample/                 ← Sample satellite images
├── output/                     ← Generated results (gitignored)
├── notebooks/                  ← Jupyter exploration notebooks
├── tests/                      ← Unit tests
├── assets/                     ← Figures for README
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🗺️ Data Sources

| Source | Type | Access |
|---|---|---|
| [ESRI World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9) | Satellite tiles | Free, no key |
| [OpenStreetMap Overpass API](https://overpass-api.de/) | Road vectors (ground truth) | Free |

---

## 🔬 Tech Stack

- **Python 3.9+**
- **OpenCV** — image processing
- **NumPy** — array operations
- **Matplotlib** — visualization
- **ReportLab** — PDF generation

---

## 🗓️ Development Timeline

| Day | Task |
|---|---|
| 1 | Project setup, data download, initial exploration |
| 2 | Preprocessing pipeline + threshold tuning |
| 3 | Road detection + skeletonization |
| 4 | Metrics computation + batch processing |
| 5 | Visualization + OSM comparison |
| 6 | Testing + documentation |
| 7 | Final polish + submission |

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## 👤 Author

**Mehul** — B.Tech Computer Science, NIT Surat  
Built as part of the Intelligent Transportation Systems (ITS) course project.
