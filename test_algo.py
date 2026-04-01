import cv2
import numpy as np
import sys
import json
import os

def detect_roads_advanced(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        sys.exit(1)
        
    h, w = img.shape[:2]
    
    # 1. Color Masking (HSV) - roads are generally gray/dark with low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Asphalt/concrete limits (adjust value range for brightness tolerance)
    lower_gray = np.array([0, 0, 0])
    upper_gray = np.array([179, 60, 200]) # Saturation < 60, Value < 200
    color_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # 2. Extract edges with Canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite("test_1_enhanced.jpg", enhanced)
    
    edges = cv2.Canny(enhanced, 50, 150)
    cv2.imwrite("test_2_edges.jpg", edges)
    
    # 3. Combine edges with color mask to remove tree/colorful roof edges
    road_edges = cv2.bitwise_and(edges, edges, mask=color_mask)
    cv2.imwrite("test_3_color_masked.jpg", road_edges)
    
    # Also combine with adaptive thresholding inside the mask
    adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    road_blobs = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=color_mask)
    
    # 4. Dilate to connect broken road segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(road_blobs, kernel, iterations=2)
    
    # 5. Directional Morphological Opening (Path Opening)
    # This keeps long linear structures and removes building/square blobs
    length = 15 # length of the line filter
    directional_max = np.zeros_like(dilated)
    
    # Create line kernels at different angles (0 to 180 degrees)
    for angle in range(0, 180, 22):
        k = np.zeros((length, length), dtype=np.uint8)
        # Calculate center
        cx, cy = length // 2, length // 2
        # Calculate end points
        x1 = int(cx + (length/2) * np.cos(np.radians(angle)))
        y1 = int(cy - (length/2) * np.sin(np.radians(angle)))
        x2 = int(cx - (length/2) * np.cos(np.radians(angle)))
        y2 = int(cy + (length/2) * np.sin(np.radians(angle)))
        
        cv2.line(k, (x1, y1), (x2, y2), 1, 1)
        
        # Apply opening with this line kernel
        opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, k)
        directional_max = cv2.bitwise_max(directional_max, opened)
        
    cv2.imwrite("test_4_directional.jpg", directional_max)

    # 6. Final cleanup (remove small components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(directional_max, connectivity=8)
    clean = np.zeros_like(directional_max)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 400: # increased threshold to remove tiny noise
            clean[labels == i] = 255
            
    cv2.imwrite("test_5_clean_mask.jpg", clean)
    
    # 7. Skeletonize
    skel = np.zeros_like(clean)
    img_skel = clean.copy()
    skel_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img_skel, skel_kernel)
        temp = cv2.dilate(eroded, skel_kernel)
        temp = cv2.subtract(img_skel, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_skel = eroded.copy()
        if cv2.countNonZero(img_skel) == 0:
            break
            
    cv2.imwrite("test_6_skeleton.jpg", skel)
    
    # 8. Visual overlay
    overlay = img.copy()
    overlay[skel > 0] = [0, 0, 255] # Red BGR
    cv2.imwrite("test_7_overlay.jpg", overlay)
    
    print("Test images generated.")

if __name__ == "__main__":
    detect_roads_advanced("data/tiles/mumbai_test.jpg")
