import sys
sys.path.insert(0, 'src')

from road_extractor import load_image, preprocess, detect_roads, compute_metrics
from geojson_exporter import export_roads_to_geojson
from report_generator import generate_report
from change_detector import align_images, detect_changes, change_metrics, build_change_overlay
import fastapi, uvicorn, numpy, cv2
print("All backend src imports: OK")

sys.path.insert(0, 'backend')
import importlib.util
spec = importlib.util.spec_from_file_location('main', 'backend/main.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("backend/main.py loaded: OK")

routes = [r.path for r in mod.app.routes]
print("Routes registered:", routes)
