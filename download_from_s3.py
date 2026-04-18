import boto3
import os
from pathlib import Path
from botocore.client import Config
import pickle
import cv2
import numpy as np
import json
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import xml.etree.ElementTree as ET
import re
import argparse
from detect_crop_rows import detect_crop_row_angle
from correction_functions import (
    get_line_box_counts, 
    find_h_sections_with_indices,
    apply_h_section_correction
)

# S3 Object Storage Configuration (Hetzner Object Storage)
S3_BUCKET_NAME = "groglanceprod"
S3_REGION = "eu-central"
S3_ACCESS_KEY = "BLSXRZVK1A3DPNL7KFDV"
S3_SECRET_KEY = "3MpPIcuuUdgO9I2OmNIueCJghCVkODy561sysBAY"
S3_ENDPOINT = "https://nbg1.your-objectstorage.com"

# Path to the folder containing subfolders
S3_BASE_PATH = "user@demo.com/fmtest02/images/"

# Local download folders
IMAGES_DIR = "downloaded_images"
PICKLES_DIR = "downloaded_pickles"
ANNOTATED_DIR = "annotated_images"

# Corn field orientation (degrees from north)
CORN_FIELD_ORIENTATION = -60.5

# Skip download and only annotate existing files (True/False)
SKIP_DOWNLOAD = True

def clear_directory(dir_path):
    """Clear all files in a directory."""
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared: {dir_path}")

def create_directories(clear_existing=False):
    """Create local directories for storing downloaded files."""
    if clear_existing:
        clear_directory(IMAGES_DIR)
        clear_directory(PICKLES_DIR)
        clear_directory(ANNOTATED_DIR)
    
    Path(IMAGES_DIR).mkdir(exist_ok=True)
    Path(PICKLES_DIR).mkdir(exist_ok=True)
    Path(ANNOTATED_DIR).mkdir(exist_ok=True)
    print(f"Created directories: {IMAGES_DIR}, {PICKLES_DIR}, and {ANNOTATED_DIR}")

def connect_to_s3():
    """Establish connection to Hetzner Object Storage."""
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        config=Config(signature_version='s3v4')
    )
    return s3_client

def list_all_files(s3_client, bucket, prefix):
    """List all files in a given prefix (handles pagination)."""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])
    
    return files

def download_file(s3_client, bucket, s3_key, local_path):
    """Download a single file from S3."""
    try:
        # Create parent directories if they don't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"✓ Downloaded: {s3_key} → {local_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {s3_key}: {str(e)}")
        return False

def load_pickle_detections(pickle_path):
    """Load bounding box detections from pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"✗ Failed to load pickle {pickle_path}: {str(e)}")
        return None

def extract_boxes_from_sahi_result(sahi_result, debug=False, confidence_threshold=0.1):
    """Extract bounding boxes from SAHI detection result object.
    
    Args:
        sahi_result: SAHI detection result
        debug: If True, print debug information
        confidence_threshold: Minimum confidence level to include (default: 0.2)
    """
    boxes = []
    filtered_count = 0
    
    def get_confidence(pred):
        """Extract confidence value from prediction object."""
        try:
            # Try category.confidence first
            if hasattr(pred, 'category') and hasattr(pred.category, 'confidence'):
                conf = pred.category.confidence
                # Handle PredictionScore objects
                if hasattr(conf, 'value'):
                    return float(conf.value)
                return float(conf)
            # Try direct score attribute
            elif hasattr(pred, 'score'):
                score = pred.score
                if hasattr(score, 'value'):
                    return float(score.value)
                return float(score)
            # Try direct confidence attribute
            elif hasattr(pred, 'confidence'):
                conf = pred.confidence
                if hasattr(conf, 'value'):
                    return float(conf.value)
                return float(conf)
            # Try dict access
            elif isinstance(pred, dict):
                if 'category' in pred and isinstance(pred['category'], dict):
                    if 'confidence' in pred['category']:
                        conf = pred['category']['confidence']
                        if hasattr(conf, 'value'):
                            return float(conf.value)
                        return float(conf)
                if 'score' in pred:
                    score = pred['score']
                    if hasattr(score, 'value'):
                        return float(score.value)
                    return float(score)
            # Default to 1.0 if not found
            return 1.0
        except:
            return 1.0
    
    try:
        if debug:
            print(f"  Debug extract: type={type(sahi_result)}, min_confidence={confidence_threshold}")
        
        # Handle direct list of ObjectPrediction objects
        if isinstance(sahi_result, list):
            if debug and len(sahi_result) > 0:
                print(f"  Debug: list with {len(sahi_result)} items, first type: {type(sahi_result[0])}")
                print(f"  Debug: first item has bbox: {hasattr(sahi_result[0], 'bbox')}")
            
            for pred_idx, pred in enumerate(sahi_result):
                # Check confidence level
                confidence = get_confidence(pred)
                
                # Skip if below threshold
                if confidence < confidence_threshold:
                    filtered_count += 1
                    continue
                
                if hasattr(pred, 'bbox'):
                    bbox = pred.bbox
                    # BoundingBox object with minx, miny, maxx, maxy attributes
                    if hasattr(bbox, 'minx') and hasattr(bbox, 'maxx') and hasattr(bbox, 'miny') and hasattr(bbox, 'maxy'):
                        boxes.append([float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)])
                    elif isinstance(bbox, tuple) and len(bbox) == 4:
                        boxes.append(list(bbox))
                    elif hasattr(bbox, 'box'):
                        # Try the .box attribute
                        box_tuple = bbox.box
                        if isinstance(box_tuple, tuple) and len(box_tuple) == 4:
                            boxes.append(list(box_tuple))
        
        # Handle SAHI Results object
        elif hasattr(sahi_result, 'object_prediction_list'):
            if debug:
                print(f"  Debug: Results object with object_prediction_list")
            for pred in sahi_result.object_prediction_list:
                # Check confidence level
                confidence = get_confidence(pred)
                
                # Skip if below threshold
                if confidence < confidence_threshold:
                    filtered_count += 1
                    continue
                
                if hasattr(pred, 'bbox'):
                    bbox = pred.bbox
                    if hasattr(bbox, 'minx') and hasattr(bbox, 'maxx'):
                        boxes.append([float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)])
                    elif isinstance(bbox, tuple) and len(bbox) == 4:
                        boxes.append(list(bbox))
        
        # Handle dict with results
        elif isinstance(sahi_result, dict) and 'object_prediction_list' in sahi_result:
            if debug:
                print(f"  Debug: Dict with object_prediction_list")
            for pred in sahi_result['object_prediction_list']:
                # Check confidence level
                confidence = get_confidence(pred)
                
                # Skip if below threshold
                if confidence < confidence_threshold:
                    filtered_count += 1
                    continue
                
                if isinstance(pred, dict) and 'bbox' in pred:
                    boxes.append(pred['bbox'])
                elif hasattr(pred, 'bbox'):
                    bbox = pred.bbox
                    if hasattr(bbox, 'minx'):
                        boxes.append([float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)])
        
        if debug and filtered_count > 0:
            print(f"  Debug: Filtered out {filtered_count} low-confidence detections")
        if debug:
            print(f"  Debug: Extracted {len(boxes)} boxes (kept)")
    
    except Exception as e:
        print(f"  Error extracting SAHI boxes: {str(e)}")
    
    return boxes

def line_intersects_box(x1, y1, x2, y2, box_x1, box_y1, box_x2, box_y2):
    """Check if a line segment intersects with a bounding box.
    
    Args:
        x1, y1, x2, y2: Line segment coordinates
        box_x1, box_y1, box_x2, box_y2: Bounding box coordinates
    
    Returns:
        True if line intersects box
    """
    def ccw(ax, ay, bx, by, cx, cy):
        """Check if three points are in counter-clockwise order."""
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    
    def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        """Check if two line segments intersect."""
        return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
               ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)
    
    # Check if either endpoint is inside the box
    if (box_x1 <= x1 <= box_x2) and (box_y1 <= y1 <= box_y2):
        return True
    if (box_x1 <= x2 <= box_x2) and (box_y1 <= y2 <= box_y2):
        return True
    
    # Check if line intersects any of the four box edges
    box_edges = [
        (box_x1, box_y1, box_x2, box_y1),  # top
        (box_x2, box_y1, box_x2, box_y2),  # right
        (box_x2, box_y2, box_x1, box_y2),  # bottom
        (box_x1, box_y2, box_x1, box_y1),  # left
    ]
    
    for edge in box_edges:
        if segments_intersect(x1, y1, x2, y2, edge[0], edge[1], edge[2], edge[3]):
            return True
    
    return False

def calculate_angle_score(boxes, angle_degrees, width, height, num_lines=50):
    """Calculate how well an angle aligns with bounding boxes.
    
    Args:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        angle_degrees: Angle to test in degrees
        width, height: Image dimensions
        num_lines: Number of lines to test
    
    Returns:
        Number of boxes intersected by the lines
    """
    if not boxes:
        return 0
    
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    
    # Normalize direction
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    intersections = 0
    max_extent = max(width, height) * 2
    
    # Test 30 lines evenly spaced
    y_positions = np.linspace(0, height, num_lines + 2)[1:-1]
    
    for y_pos in y_positions:
        y_pos = int(y_pos)
        
        # Calculate line endpoints
        start_x = int(width / 2 - max_extent * dx)
        start_y = int(y_pos - max_extent * dy)
        end_x = int(width / 2 + max_extent * dx)
        end_y = int(y_pos + max_extent * dy)
        
        # Check intersection with each box
        for box in boxes:
            x1, y1, x2, y2 = box
            if line_intersects_box(start_x, start_y, end_x, end_y, int(x1), int(y1), int(x2), int(y2)):
                intersections += 1
    
    return intersections

def kmedians_clustering(data, k=3):
    """K-means clustering for finding optimal cluster centers.
    
    Args:
        data: List or array of values to cluster
        k: Number of clusters (optimized for k=3)
    
    Returns:
        List of cluster centers (means) sorted in ascending order
    """
    if len(data) == 0:
        return [0, 0, 0] if k == 3 else [0] * k
    
    data = np.array(data, dtype=float)
    
    # Initialize centers using percentiles (k-means++ style initialization)
    percentiles = [i * 100 / (k + 1) for i in range(1, k + 1)]
    centers = np.array([np.percentile(data, p) for p in percentiles])
    
    # K-means iteration
    max_iterations = 100
    for iteration in range(max_iterations):
        # Assign each point to nearest center
        distances = np.abs(data[:, np.newaxis] - centers[np.newaxis, :])
        assignments = np.argmin(distances, axis=1)
        
        # Calculate new centers as means of assigned points
        new_centers = np.zeros(k)
        for i in range(k):
            cluster_points = data[assignments == i]
            if len(cluster_points) > 0:
                new_centers[i] = np.mean(cluster_points)
            else:
                new_centers[i] = centers[i]
        
        # Check for convergence
        if np.allclose(centers, np.sort(new_centers)):
            centers = np.sort(new_centers)
            break
        
        centers = np.sort(new_centers)
    
    return sorted(centers)

def normalize_boxes_to_average_size(boxes):
    """Normalize smaller bounding boxes to the size of the top 20% largest boxes.
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
    
    Returns:
        List of normalized bounding boxes
    """
    if not boxes or len(boxes) == 0:
        return boxes
    
    # Calculate size for each box (area and dimensions)
    box_sizes = []
    for box in boxes:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height
            box_sizes.append({
                'box': box,
                'width': width,
                'height': height,
                'area': area
            })
    
    if not box_sizes:
        return boxes
    
    # Sort by area (descending) and get top 20%
    sorted_by_area = sorted(box_sizes, key=lambda x: x['area'], reverse=True)
    top_20_percent_count = max(1, len(sorted_by_area) // 5)  # At least 1 box
    top_20_percent = sorted_by_area[:top_20_percent_count]
    
    # Calculate average dimensions of top 20% largest boxes
    avg_width = np.mean([b['width'] for b in top_20_percent])
    avg_height = np.mean([b['height'] for b in top_20_percent])
    avg_area = avg_width * avg_height
    
    print(f"Top 20% largest boxes (n={top_20_percent_count}): avg size {avg_width:.1f}x{avg_height:.1f} (area: {avg_area:.0f})")
    
    # Normalize boxes smaller than the top 20% average to that size
    normalized_boxes = []
    normalized_count = 0
    
    for box_info in box_sizes:
        box = box_info['box']
        area = box_info['area']
        
        # If box is smaller than top 20% average, enlarge it
        if area < avg_area:
            x1, y1, x2, y2 = box
            
            # Calculate center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate new half dimensions
            new_half_width = avg_width / 2
            new_half_height = avg_height / 2
            
            # Create new box centered at original center
            new_x1 = center_x - new_half_width
            new_y1 = center_y - new_half_height
            new_x2 = center_x + new_half_width
            new_y2 = center_y + new_half_height
            
            normalized_boxes.append([new_x1, new_y1, new_x2, new_y2])
            normalized_count += 1
        else:
            normalized_boxes.append(box)
    
    print(f"Normalized {normalized_count} boxes to top 20% size")
    return normalized_boxes

def find_best_angle(boxes, base_angle, width, height, max_deviation=30):
    """Find the best angle by testing ±max_deviation from base angle.
    
    Args:
        boxes: List of bounding boxes
        base_angle: The calculated angle to start from
        width, height: Image dimensions
        max_deviation: Test range (±degrees)
    
    Returns:
        Tuple of (best_angle, best_score)
    """
    best_angle = base_angle
    best_score = calculate_angle_score(boxes, base_angle, width, height)
    
    # Test angles from -max_deviation to +max_deviation in 0.5 degree steps
    for offset in np.arange(-max_deviation, max_deviation + 0.5, 0.5):
        test_angle = base_angle + offset
        score = calculate_angle_score(boxes, test_angle, width, height)
        
        if score > best_score:
            best_score = score
            best_angle = test_angle
    
    return best_angle, best_score

def draw_orientation_lines_with_counts(image, boxes, angle_degrees, line_color=(255, 0, 0), num_lines=100, thickness=3, cluster_centers=None):
    """Draw orientation lines with box intersection counts displayed and smoothed colors.
    
    Returns:
        Tuple of (intersection_counts, line_colors, line_positions)
    """
    height, width = image.shape[:2]
    
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    
    # Normalize the direction vector
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    # Calculate perpendicular direction for spacing
    perp_x = -dy
    perp_y = dx
    
    # Calculate the distance across the image
    diag_length = np.sqrt(width**2 + height**2)
    max_extent = diag_length
    
    intersection_counts = []
    line_colors = []
    line_positions = []
    
    # First pass: collect all data and assign initial colors
    for i in range(num_lines):
        # Distribute spacing from -max_extent/2 to +max_extent/2
        offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
        
        # Start from center and go perpendicular
        center_x = width / 2
        center_y = height / 2
        
        # Position along perpendicular
        perp_offset_x = perp_x * offset
        perp_offset_y = perp_y * offset
        
        # Line center point
        line_center_x = center_x + perp_offset_x
        line_center_y = center_y + perp_offset_y
        
        # Extend in the line direction
        start_x = int(line_center_x - max_extent * dx)
        start_y = int(line_center_y - max_extent * dy)
        end_x = int(line_center_x + max_extent * dx)
        end_y = int(line_center_y + max_extent * dy)
        
        line_positions.append((start_x, start_y, end_x, end_y, int(line_center_x), int(line_center_y)))
        
        # Count boxes that overlap/touch this line (considering box size, not just center)
        # A box is counted if any part of it is within threshold distance from the line
        intersections = 0
        line_thickness_threshold = 40  # Boxes within ~40 pixels of line
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Get the four corners of the box
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            
            # Calculate perpendicular distance for each corner
            distances = []
            for cx, cy in corners:
                d = (cx - line_center_x) * perp_x + (cy - line_center_y) * perp_y
                distances.append(d)
            
            min_d = min(distances)
            max_d = max(distances)
            
            # Check if the line passes through the box OR any corner is within threshold
            if (min_d <= 0 and max_d >= 0) or min(abs(d) for d in distances) < line_thickness_threshold:
                intersections += 1
        
        intersection_counts.append(intersections)
    
    # Smooth intersection counts to eliminate isolated low-count lines
    smoothed_intersection_counts = intersection_counts.copy()
    for i in range(1, num_lines - 1):
        left_count = intersection_counts[i - 1]
        center_count = intersection_counts[i]
        right_count = intersection_counts[i + 1]
        
        # If center is isolated (different from both neighbors), use median of the three
        if center_count != left_count and center_count != right_count:
            median_val = np.median([left_count, center_count, right_count])
            smoothed_intersection_counts[i] = int(median_val)
    
    intersection_counts = smoothed_intersection_counts
    
    # First pass: assign initial colors based on smoothed counts
    close_to_c2_indices = []  # Track lines close to C2 (for testing visualization)
    
    # Extract cluster centers for use in expansion logic
    if cluster_centers is not None and len(cluster_centers) == 3:
        c1, c2, c3 = cluster_centers
    else:
        c1, c2, c3 = 0, 10, 30  # Default fallback values
    
    for i, intersections in enumerate(intersection_counts):
        # Determine initial color based on clusters
        if cluster_centers is not None and len(cluster_centers) == 3:
            dist_to_c1 = abs(intersections - c1)
            dist_to_c2 = abs(intersections - c2)
            dist_to_c3 = abs(intersections - c3)
            
            # Classify as 'l' if closer to c1 or c2, else 'h' (c3)
            min_dist_low = min(dist_to_c1, dist_to_c2)
            if min_dist_low <= dist_to_c3:
                # Line is classified as low (red)
                # Check if it's closer to C2 than C1
                if dist_to_c2 < dist_to_c1:
                    close_to_c2_indices.append(i)  # Track this for later
                draw_color = (0, 0, 255)  # Red for low cluster
            else:
                draw_color = (255, 0, 0)    # Blue for high cluster
        else:
            draw_color = line_color
        
        line_colors.append(draw_color)
    
    # Expand blue (high) lines: for each blue line, make 3 before and after blue too
    # Unconditional expansion - always add 3 rows before and after
    # Do this in a single pass (not iterative) to avoid cascading
    blue_indices = set()
    for i in range(len(line_colors)):
        if line_colors[i] == (255, 0, 0):  # If blue (high)
            blue_indices.add(i)
            # Mark 3 lines before - unconditionally
            for j in range(max(0, i - 3), i):
                blue_indices.add(j)
            # Mark 3 lines after - unconditionally
            for j in range(i + 1, min(len(line_colors), i + 4)):
                blue_indices.add(j)
    
    # Apply the expansion in one pass
    for i in blue_indices:
        line_colors[i] = (255, 0, 0)
    
    # Change lines close to C2 to yellow for testing visualization
    # BUT only if they're still red (not expanded to blue)
    for i in close_to_c2_indices:
        if line_colors[i] == (0, 0, 255):  # Only if still red
            line_colors[i] = (0, 255, 255)  # Yellow for lines close to C2
    
    # Draw all lines with their colors (no island checking)
    for i in range(num_lines):
        start_x, start_y, end_x, end_y, label_x, label_y = line_positions[i]
        
        # Draw the line with its color
        cv2.line(image, (start_x, start_y), (end_x, end_y), line_colors[i], thickness)
        
        # Draw intersection count at the top of the line
        label_x_pos = int(label_x - max_extent * dx * 0.95)
        label_y_pos = int(label_y - max_extent * dy * 0.95)
        
        # Ensure label is within image bounds with good margins
        label_x_pos = max(30, min(width - 50, label_x_pos))
        label_y_pos = max(40, min(height - 20, label_y_pos))
        
        # Create text string
        text = str(intersection_counts[i])
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        text_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        
        # Draw background rectangle for text
        padding = 5
        cv2.rectangle(image, 
                     (label_x_pos - padding, label_y_pos - text_size[1] - padding),
                     (label_x_pos + text_size[0] + padding, label_y_pos + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Draw text in white
        cv2.putText(image, text, (label_x_pos, label_y_pos), 
                   font, font_scale, (255, 255, 255), text_thickness)
    
    return intersection_counts, line_colors, line_positions

def get_line_colors_for_boxes_with_line_colors(image, boxes, angle_degrees, line_colors=None, cluster_centers=None, num_lines=100):
    """
    Assign box colors based on closest line.
    If line_colors are provided (from line drawing), use those exact colors.
    Otherwise calculate them independently.
    """
    height, width = image.shape[:2]
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    perp_x = -dy
    perp_y = dx
    diag_length = np.sqrt(width**2 + height**2)
    
    # Calculate line positions
    line_positions = []
    for i in range(num_lines):
        offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
        center_x = width / 2
        center_y = height / 2
        perp_offset_x = perp_x * offset
        perp_offset_y = perp_y * offset
        line_center_x = center_x + perp_offset_x
        line_center_y = center_y + perp_offset_y
        
        line_positions.append((line_center_x, line_center_y))
    
    # Use provided line colors or fall back to None
    actual_line_colors = line_colors if line_colors is not None else [None] * num_lines
    
    # Assign colors to boxes based on which lines cross them
    box_colors = []
    high_count = 0
    low_count = 0
    
    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Get the four corners of the box
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        # Count blue and red lines that intersect this box
        blue_line_count = 0
        red_line_count = 0
        
        for line_idx, (line_x, line_y) in enumerate(line_positions):
            # Calculate perpendicular distance for each corner
            distances = []
            for cx, cy in corners:
                d = (cx - line_x) * perp_x + (cy - line_y) * perp_y
                distances.append(d)
            
            min_d = min(distances)
            max_d = max(distances)
            
            # Check if the line passes through the box (or intersects it within threshold)
            line_thickness_threshold = 40
            if (min_d <= 0 and max_d >= 0) or min(abs(d) for d in distances) < line_thickness_threshold:
                # Line crosses this box - count its color
                if actual_line_colors[line_idx] is not None:
                    if actual_line_colors[line_idx] == (255, 0, 0):  # Blue
                        blue_line_count += 1
                    else:  # Red or yellow
                        red_line_count += 1
        
        # Assign box color based on which color has more lines crossing it
        if blue_line_count >= red_line_count:
            box_color = (255, 0, 0)  # Blue = high
        else:
            box_color = (0, 0, 255)  # Red = low
        
        box_colors.append((box_idx, box_color))
        
        if box_color == (255, 0, 0):  # Blue = high
            high_count += 1
        else:  # Red = low
            low_count += 1
    
    return box_colors, high_count, low_count


def get_line_colors_for_boxes(image, boxes, angle_degrees, cluster_centers=None, num_lines=100):
    """
    Assign box colors based on closest line's color.
    Uses perpendicular distance-based counting (same as line drawing).
    """
    height, width = image.shape[:2]
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    perp_x = -dy
    perp_y = dx
    diag_length = np.sqrt(width**2 + height**2)
    
    # Calculate line positions
    line_positions = []
    
    for i in range(num_lines):
        offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
        center_x = width / 2
        center_y = height / 2
        perp_offset_x = perp_x * offset
        perp_offset_y = perp_y * offset
        line_center_x = center_x + perp_offset_x
        line_center_y = center_y + perp_offset_y
        
        line_positions.append((line_center_x, line_center_y))
    
    # Count boxes that overlap/touch each line (considering box size, not just center)
    line_box_counts = [0] * num_lines
    line_thickness_threshold = 40  # Same as line drawing
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Get the four corners of the box
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        # For each line, check if the box overlaps/touches it
        for line_idx, (line_x, line_y) in enumerate(line_positions):
            # Calculate perpendicular distance for each corner
            distances = []
            for cx, cy in corners:
                d = (cx - line_x) * perp_x + (cy - line_y) * perp_y
                distances.append(d)
            
            min_d = min(distances)
            max_d = max(distances)
            
            # Check if the line passes through the box OR any corner is within threshold
            if (min_d <= 0 and max_d >= 0) or min(abs(d) for d in distances) < line_thickness_threshold:
                line_box_counts[line_idx] += 1
    
    # Smooth box counts to eliminate isolated low-count lines
    smoothed_counts = line_box_counts.copy()
    for i in range(1, num_lines - 1):
        left_count = line_box_counts[i - 1]
        center_count = line_box_counts[i]
        right_count = line_box_counts[i + 1]
        
        # If center is isolated (different from both neighbors), use median of the three
        if center_count != left_count and center_count != right_count:
            median_val = np.median([left_count, center_count, right_count])
            smoothed_counts[i] = int(median_val)
    
    line_box_counts = smoothed_counts
    
    # Color each line based on smoothed box count
    line_colors_map = []
    # Extract cluster centers for use in expansion logic
    if cluster_centers is not None and len(cluster_centers) == 3:
        c1, c2, c3 = cluster_centers
    else:
        c1, c2, c3 = 0, 10, 30  # Default fallback values
    
    for count in line_box_counts:
        # Determine line color based on cluster
        if cluster_centers is not None and len(cluster_centers) == 3:
            dist_to_c1 = abs(count - c1)
            dist_to_c2 = abs(count - c2)
            dist_to_c3 = abs(count - c3)
            
            # Classify as 'l' if closer to c1 or c2, else 'h'
            min_dist_low = min(dist_to_c1, dist_to_c2)
            if min_dist_low <= dist_to_c3:
                line_color = (0, 0, 255)    # Red for low
            else:
                line_color = (255, 0, 0)   # Blue for high
        else:
            line_color = (0, 0, 255)       # Default red
        
        line_colors_map.append(line_color)
    
    # Expand blue (high) lines: for each blue line, make 3 before and after blue too
    # Unconditional expansion - always add 3 rows before and after
    # Do this in a single pass (not iterative) to avoid cascading
    blue_indices = set()
    for i in range(len(line_colors_map)):
        if line_colors_map[i] == (255, 0, 0):  # If blue (high)
            blue_indices.add(i)
            # Mark 3 lines before - unconditionally
            for j in range(max(0, i - 3), i):
                blue_indices.add(j)
            # Mark 3 lines after - unconditionally
            for j in range(i + 1, min(len(line_colors_map), i + 4)):
                blue_indices.add(j)
    
    # Apply the expansion in one pass
    for i in blue_indices:
        line_colors_map[i] = (255, 0, 0)
    
    # Assign colors to boxes based on which lines cross them
    box_colors = []
    high_count = 0
    low_count = 0
    
    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Get the four corners of the box
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        # Count blue and red lines that intersect this box
        blue_line_count = 0
        red_line_count = 0
        
        for line_idx, (line_x, line_y) in enumerate(line_positions):
            # Calculate perpendicular distance for each corner
            distances = []
            for cx, cy in corners:
                d = (cx - line_x) * perp_x + (cy - line_y) * perp_y
                distances.append(d)
            
            min_d = min(distances)
            max_d = max(distances)
            
            # Check if the line passes through the box (or intersects it within threshold)
            line_thickness_threshold = 40
            if (min_d <= 0 and max_d >= 0) or min(abs(d) for d in distances) < line_thickness_threshold:
                # Line crosses this box - count its color
                if line_colors_map[line_idx] == (255, 0, 0):  # Blue
                    blue_line_count += 1
                else:  # Red
                    red_line_count += 1
        
        # Assign box color based on which color has more lines crossing it
        if blue_line_count >= red_line_count:
            box_color = (255, 0, 0)  # Blue = high
        else:
            box_color = (0, 0, 255)  # Red = low
        
        box_colors.append((box_idx, box_color))
        
        if box_color == (255, 0, 0):  # Blue = high
            high_count += 1
        else:  # Red = low
            low_count += 1
    
    return box_colors, high_count, low_count

def assign_box_colors_from_rows(image, boxes, angle_degrees, cluster_centers=None, num_lines=100):
    """
    Assign colors to boxes based on DENSITY of boxes near each row.
    Uses closest-line assignment to count box density per row (not intersection-based).
    Returns: (box_colors, line_colors, line_positions, high_count, low_count)
    - box_colors: list of (box_idx, color) tuples
    - line_colors: list of colors for each line (for drawing)
    - line_positions: list of (start_x, start_y, end_x, end_y) for each line (for drawing)
    - high_count, low_count: box counts
    """
    height, width = image.shape[:2]
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    perp_x = -dy
    perp_y = dx
    diag_length = np.sqrt(width**2 + height**2)
    max_extent = diag_length
    
    # Calculate all line positions (centers only for density calculation)
    line_positions = []
    line_segments = []  # For drawing: (start_x, start_y, end_x, end_y)
    for i in range(num_lines):
        offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
        center_x = width / 2
        center_y = height / 2
        perp_offset_x = perp_x * offset
        perp_offset_y = perp_y * offset
        line_center_x = center_x + perp_offset_x
        line_center_y = center_y + perp_offset_y
        
        line_positions.append((line_center_x, line_center_y))
        
        # Calculate line segment endpoints for drawing
        start_x = int(line_center_x - max_extent * dx)
        start_y = int(line_center_y - max_extent * dy)
        end_x = int(line_center_x + max_extent * dx)
        end_y = int(line_center_y + max_extent * dy)
        line_segments.append((start_x, start_y, end_x, end_y))
    
    # Step 1: Assign each box to its closest line (DENSITY-BASED, not intersection)
    box_assignments = {}  # {box_idx: closest_line_idx}
    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Find closest line to this box center
        min_dist = float('inf')
        closest_line_idx = 0
        
        for line_idx, (line_x, line_y) in enumerate(line_positions):
            # Distance from box center to line center in perpendicular direction
            dist = abs((box_center_x - line_x) * perp_x + (box_center_y - line_y) * perp_y)
            if dist < min_dist:
                min_dist = dist
                closest_line_idx = line_idx
        
        box_assignments[box_idx] = closest_line_idx
    
    # Step 2: Count boxes assigned to each line (DENSITY of boxes, not intersections)
    line_box_counts = [0] * num_lines
    for closest_line_idx in box_assignments.values():
        line_box_counts[closest_line_idx] += 1
    
    # Step 2.5: Smooth box counts to eliminate isolated low-count lines
    smoothed_counts = line_box_counts.copy()
    for i in range(1, num_lines - 1):
        left_count = line_box_counts[i - 1]
        center_count = line_box_counts[i]
        right_count = line_box_counts[i + 1]
        
        # If center is isolated (different from both neighbors), use median of the three
        if center_count != left_count and center_count != right_count:
            median_val = np.median([left_count, center_count, right_count])
            smoothed_counts[i] = int(median_val)
    
    line_box_counts = smoothed_counts
    
    # Step 3: Color each line based on box count (DENSITY-BASED)
    line_colors_map = []
    # Extract cluster centers for use in expansion logic
    if cluster_centers is not None and len(cluster_centers) == 3:
        c1, c2, c3 = cluster_centers
    else:
        c1, c2, c3 = 0, 10, 30  # Default fallback values
    
    for count in line_box_counts:
        # Determine line color based on cluster
        if cluster_centers is not None and len(cluster_centers) == 3:
            dist_to_c1 = abs(count - c1)
            dist_to_c2 = abs(count - c2)
            dist_to_c3 = abs(count - c3)
            
            # Classify as 'l' if closer to c1 or c2, else 'h'
            min_dist_low = min(dist_to_c1, dist_to_c2)
            if min_dist_low <= dist_to_c3:
                line_color = (0, 0, 255)    # Red for low
            else:
                line_color = (255, 0, 0)   # Blue for high
        else:
            line_color = (0, 0, 255)       # Default red
        
        line_colors_map.append(line_color)
    
    # Expand blue (high) lines: for each blue line, make 3 before and after blue too
    # Unconditional expansion - always add 3 rows before and after
    # Do this in a single pass (not iterative) to avoid cascading
    blue_indices = set()
    for i in range(len(line_colors_map)):
        if line_colors_map[i] == (255, 0, 0):  # If blue (high)
            blue_indices.add(i)
            # Mark 3 lines before - unconditionally
            for j in range(max(0, i - 3), i):
                blue_indices.add(j)
            # Mark 3 lines after - unconditionally
            for j in range(i + 1, min(len(line_colors_map), i + 4)):
                blue_indices.add(j)
    
    # Apply the expansion in one pass
    for i in blue_indices:
        line_colors_map[i] = (255, 0, 0)
    
    # Step 4: Smooth line colors to prevent color islands
    smoothed_colors = line_colors_map.copy()
    for i in range(1, num_lines - 1):
        left_color = smoothed_colors[i - 1]
        right_color = smoothed_colors[i + 1]
        current_color = smoothed_colors[i]
        
        if left_color == right_color and current_color != left_color:
            smoothed_colors[i] = left_color
    
    # Step 5: Assign colors to boxes based on which lines cross them
    box_colors = []
    high_count = 0
    low_count = 0
    
    # Recalculate line positions in full (with extended segments for intersection checking)
    diag_length = np.sqrt(width**2 + height**2)
    max_extent = diag_length
    
    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Get the four corners of the box
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        # Count blue and red lines that intersect this box
        blue_line_count = 0
        red_line_count = 0
        
        for line_idx, (line_x, line_y) in enumerate(line_positions):
            # Calculate perpendicular distance for each corner
            distances = []
            for cx, cy in corners:
                d = (cx - line_x) * perp_x + (cy - line_y) * perp_y
                distances.append(d)
            
            min_d = min(distances)
            max_d = max(distances)
            
            # Check if the line passes through the box (or intersects it within threshold)
            line_thickness_threshold = 40
            if (min_d <= 0 and max_d >= 0) or min(abs(d) for d in distances) < line_thickness_threshold:
                # Line crosses this box - count its color
                if smoothed_colors[line_idx] == (255, 0, 0):  # Blue
                    blue_line_count += 1
                else:  # Red
                    red_line_count += 1
        
        # Assign box color based on which color has more lines crossing it
        if blue_line_count >= red_line_count:
            color = (255, 0, 0)  # Blue = high
        else:
            color = (0, 0, 255)  # Red = low
        
        box_colors.append((box_idx, color))
        
        if color == (255, 0, 0):  # Blue = high
            high_count += 1
        else:  # Red = low
            low_count += 1
    
    return box_colors, smoothed_colors, line_segments, high_count, low_count

def find_repeating_pattern(sequence, max_period=10):
    """
    Find the smallest repeating unit in a sequence.
    
    Args:
        sequence: String of 'h' and 'l' characters
        max_period: Maximum period length to check (default 10)
    
    Returns:
        Tuple of (pattern, period, confidence)
        - pattern: The repeating unit (e.g., "lhh")
        - period: Length of the pattern
        - confidence: Percentage of sequence that matches the pattern
    """
    best_pattern = sequence  # Fallback to full sequence
    best_period = len(sequence)
    best_score = 0
    
    # Try different period lengths
    for period in range(1, min(max_period + 1, len(sequence) // 2 + 1)):
        # Build the most common pattern for this period
        # by finding the most frequent character at each position mod period
        pattern_chars = []
        for pos in range(period):
            # Collect all characters at positions that are pos mod period
            chars_at_pos = [sequence[i] for i in range(len(sequence)) if i % period == pos]
            # Use the most common character
            if chars_at_pos:
                most_common_char = max(set(chars_at_pos), key=chars_at_pos.count)
                pattern_chars.append(most_common_char)
        
        pattern_unit = ''.join(pattern_chars)
        
        # Check how many lines match this repeating pattern
        matches = 0
        for i in range(len(sequence)):
            if sequence[i] == pattern_unit[i % period]:
                matches += 1
        
        score = matches / len(sequence) if len(sequence) > 0 else 0
        
        if score > best_score:
            best_score = score
            best_pattern = pattern_unit
            best_period = period
    
    return best_pattern, best_period, best_score

def analyze_h_sections(pattern):
    """
    Find all continuous sections of 'h' in a pattern string.
    
    Args:
        pattern: String of 'h' and 'l' characters (e.g., "llhhhhllhhhll")
    
    Returns:
        List of lengths of continuous 'h' sections
    """
    h_sections = []
    current_h_length = 0
    
    for char in pattern:
        if char == 'h':
            current_h_length += 1
        else:
            if current_h_length > 0:
                h_sections.append(current_h_length)
                current_h_length = 0
    
    # Don't forget the last section if pattern ends with 'h'
    if current_h_length > 0:
        h_sections.append(current_h_length)
    
    return h_sections

def analyze_red_sections_between_blue(line_colors):
    """
    Find all continuous red sections that are sandwiched between blue sections.
    Yellow lines count as red.
    
    Args:
        line_colors: List of color tuples (BGR format)
    
    Returns:
        List of lengths of red sections that are between blue sections
    """
    # Convert colors to h/l pattern: blue (255,0,0)='h', red (0,0,255)='l', yellow (0,255,255)='l'
    pattern = ""
    for color in line_colors:
        if color == (255, 0, 0):  # Blue = high
            pattern += "h"
        else:  # Red or yellow = low
            pattern += "l"
    
    # Find all red sections with their positions
    red_sections_with_pos = []
    current_l_length = 0
    start_pos = 0
    
    for i, char in enumerate(pattern):
        if char == 'l':
            if current_l_length == 0:
                start_pos = i
            current_l_length += 1
        else:
            if current_l_length > 0:
                red_sections_with_pos.append((start_pos, current_l_length))
                current_l_length = 0
    
    # Don't forget the last section if pattern ends with 'l'
    if current_l_length > 0:
        red_sections_with_pos.append((start_pos, current_l_length))
    
    # Filter to only red sections that have blue sections on both sides
    red_sections_between_blue = []
    for pos, length in red_sections_with_pos:
        # Check if there's blue before this red section
        has_blue_before = pos > 0 and pattern[pos - 1] == 'h'
        # Check if there's blue after this red section
        has_blue_after = (pos + length) < len(pattern) and pattern[pos + length] == 'h'
        
        if has_blue_before and has_blue_after:
            red_sections_between_blue.append(length)
    
    return red_sections_between_blue

def get_row_pattern(image, boxes, angle_degrees, cluster_centers=None, num_lines=100):
    """
    Extract the full h/l pattern of crop rows from an image.
    Returns tuple of (original_sequence, expanded_sequence)
    - original_sequence: Pattern before expansion
    - expanded_sequence: Pattern after expansion (for drawing/coloring)
    """
    height, width = image.shape[:2]
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    perp_x = -dy
    perp_y = dx
    diag_length = np.sqrt(width**2 + height**2)
    max_extent = diag_length
    
    # Calculate line positions and count intersections
    intersection_counts = []
    for i in range(num_lines):
        offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
        center_x = width / 2
        center_y = height / 2
        perp_offset_x = perp_x * offset
        perp_offset_y = perp_y * offset
        line_center_x = center_x + perp_offset_x
        line_center_y = center_y + perp_offset_y
        
        start_x = int(line_center_x - max_extent * dx)
        start_y = int(line_center_y - max_extent * dy)
        end_x = int(line_center_x + max_extent * dx)
        end_y = int(line_center_y + max_extent * dy)
        
        intersections = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            if line_intersects_box(start_x, start_y, end_x, end_y, int(x1), int(y1), int(x2), int(y2)):
                intersections += 1
        
        intersection_counts.append(intersections)
    
    # Classify each line as high or low
    if cluster_centers is not None and len(cluster_centers) == 3:
        c1, c2, c3 = cluster_centers
        sequence = ""
        for count in intersection_counts:
            dist_to_c1 = abs(count - c1)
            dist_to_c2 = abs(count - c2)
            dist_to_c3 = abs(count - c3)
            
            # Classify as 'l' if closer to c1 or c2, else 'h'
            min_dist_low = min(dist_to_c1, dist_to_c2)
            if min_dist_low <= dist_to_c3:
                sequence += "l"  # Low (clusters 1-2)
            else:
                sequence += "h"  # High (cluster 3)
    else:
        sequence = "h" * num_lines  # Default to all high
    
    # Store original pattern before expansion
    original_sequence = sequence
    
    # Expand 'h' rows: for each 'h', mark the 3 rows before and after as 'h' too
    sequence_list = list(sequence)
    for i in range(len(sequence_list)):
        if sequence_list[i] == 'h':
            # Mark 3 rows before
            for j in range(max(0, i - 3), i):
                sequence_list[j] = 'h'
            # Mark 3 rows after
            for j in range(i + 1, min(len(sequence_list), i + 4)):
                sequence_list[j] = 'h'
    
    expanded_sequence = ''.join(sequence_list)
    return original_sequence, expanded_sequence


def draw_pattern_visualization(pattern, angle_degrees, num_lines=100, output_path="pattern_visualization.jpg"):
    """
    Draw the repeating row pattern on a blank image.
    The pattern repeats across the entire image.
    
    Args:
        pattern: String of 'h' and 'l' characters (repeating unit, e.g., "lhh")
        angle_degrees: Angle of rows
        num_lines: Number of total lines to draw across the image
        output_path: Where to save the visualization
    """
    # Create blank image (landscape orientation)
    width, height = 1280, 720
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    angle_rad = np.radians(angle_degrees)
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx /= norm
        dy /= norm
    
    perp_x = -dy
    perp_y = dx
    diag_length = np.sqrt(width**2 + height**2)
    max_extent = diag_length
    
    pattern_len = len(pattern)
    
    # Draw lines, repeating the pattern
    for i in range(num_lines):
        offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
        center_x = width / 2
        center_y = height / 2
        perp_offset_x = perp_x * offset
        perp_offset_y = perp_y * offset
        line_center_x = center_x + perp_offset_x
        line_center_y = center_y + perp_offset_y
        
        start_x = int(line_center_x - max_extent * dx)
        start_y = int(line_center_y - max_extent * dy)
        end_x = int(line_center_x + max_extent * dx)
        end_y = int(line_center_y + max_extent * dy)
        
        # Get color from repeating pattern
        pattern_idx = i % pattern_len
        if pattern[pattern_idx] == 'h':
            color = (255, 0, 0)  # Blue for high
        else:
            color = (0, 0, 255)  # Red for low
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness=3)
    
    # Save
    cv2.imwrite(output_path, image)
    print(f"✓ Pattern visualization saved: {output_path}")


def draw_bounding_boxes(image_path, detections, output_path, color=(0, 0, 255), thickness=3, debug_first=False, cluster_centers=None, collect_only=False, draw_lines=False, override_angle=None, num_lines=100, is_h_section_outlier=False, avg_h_length=None, yellow_to_blue_indices=None):
    """Draw bounding boxes on image and save it.
    
    Args:
        image_path: Path to input image
        detections: Detection objects or list
        output_path: Path to save annotated image
        color: BGR color tuple (default red: 0,0,255)
        thickness: Line thickness (default 3)
        debug_first: Enable debug for first image
        cluster_centers: List of 2 cluster centers for line coloring
        collect_only: If True, only collect counts without saving (for k-means calculation)
        draw_lines: If True, draw the orientation lines on top of boxes
        override_angle: If provided, use this angle instead of detecting it from the image
        num_lines: Number of crop rows to detect per image (default: 100)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Failed to load image: {image_path}")
            return False if not collect_only else []
        
        height, width = image.shape[:2]
        
        # Extract boxes from various formats
        # We need two versions: filtered (for line classification) and all (for drawing)
        filtered_boxes = None
        all_boxes = None
        
        # Check if it's a SAHI result
        if isinstance(detections, list) or hasattr(detections, 'object_prediction_list') or (isinstance(detections, dict) and 'object_prediction_list' in detections):
            # Extract filtered boxes (confidence >= 0.2) for line classification
            filtered_boxes = extract_boxes_from_sahi_result(detections, debug=debug_first, confidence_threshold=0.1)
            # Extract all boxes (no filtering) for drawing on image
            all_boxes = extract_boxes_from_sahi_result(detections, debug=debug_first, confidence_threshold=0.0)
        elif isinstance(detections, dict):
            # Try common keys
            for key in ['boxes', 'detections', 'bounding_boxes', 'bbox']:
                if key in detections:
                    filtered_boxes = detections[key]
                    all_boxes = detections[key]
                    break
        
        if not filtered_boxes:
            if debug_first:
                print(f"⊘ No detections found in {image_path}")
            return False if not collect_only else []
        
        # Normalize boxes to average size (make small boxes larger)
        filtered_boxes = normalize_boxes_to_average_size(filtered_boxes)
        all_boxes = normalize_boxes_to_average_size(all_boxes) if all_boxes else filtered_boxes
        
        # If collecting only, just get counts without drawing/saving
        if collect_only:
            try:
                base_angle = detect_crop_row_angle(image_path, verbose=False)
                # Create a temporary copy for counting (don't save)
                temp_image = image.copy()
                result = draw_orientation_lines_with_counts(temp_image, filtered_boxes, base_angle, num_lines=num_lines, thickness=3, cluster_centers=None)
                # Extract just the intersection counts from the tuple
                if isinstance(result, tuple) and len(result) == 3:
                    intersection_counts = result[0]
                    return intersection_counts
                else:
                    return result
            except Exception as e:
                if debug_first:
                    print(f"⊘ Could not detect crop row angle: {str(e)}")
                return []
        
        # Get crop row angle for line drawing
        try:
            # Use override_angle if provided, otherwise detect from image
            if override_angle is not None:
                base_angle = override_angle
            else:
                base_angle = detect_crop_row_angle(image_path, verbose=False)
        except Exception as e:
            if debug_first:
                print(f"⊘ Could not detect crop row angle: {str(e)}")
            return False if not collect_only else []
        
        # Draw the orientation lines first (they provide the correct row coloring reference)
        # Use FILTERED boxes for line classification (confidence >= 0.2)
        line_colors = None
        line_positions = None
        if draw_lines:
            _, line_colors, line_positions = draw_orientation_lines_with_counts(image, filtered_boxes, base_angle, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
        
        # Get box colors using the same line colors that were just drawn
        # Calculate for ALL boxes (same set being drawn), not just filtered
        box_colors, high_count, low_count = get_line_colors_for_boxes_with_line_colors(image, all_boxes, base_angle, line_colors=line_colors, cluster_centers=cluster_centers, num_lines=num_lines)
        
        drawn_count = 0
        
        # Calculate line positions for mapping boxes to lines
        angle_rad = np.radians(base_angle)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx /= norm
            dy /= norm
        
        perp_x = -dy
        perp_y = dx
        diag_length = np.sqrt(width**2 + height**2)
        
        # Calculate line positions for closest line lookup
        line_positions_for_mapping = []
        for i in range(num_lines):
            offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
            center_x = width / 2
            center_y = height / 2
            perp_offset_x = perp_x * offset
            perp_offset_y = perp_y * offset
            line_center_x = center_x + perp_offset_x
            line_center_y = center_y + perp_offset_y
            line_positions_for_mapping.append((line_center_x, line_center_y))
        
        # Determine which line each box in all_boxes is closest to
        # Get line color map from pre-calculated line colors if available
        # Make sure we have line_positions if we need to apply conversions
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        needs_line_positions = yellow_to_blue_indices is not None and img_basename in yellow_to_blue_indices
        
        if (line_colors is None or line_positions is None) and (draw_lines or needs_line_positions):
            # Recalculate using filtered_boxes 
            _, recalc_line_colors, recalc_line_positions = draw_orientation_lines_with_counts(image.copy(), filtered_boxes, base_angle, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
            if line_colors is None:
                line_colors = recalc_line_colors
            if line_positions is None:
                line_positions = recalc_line_positions
        
        # Apply h-section correction for outlier images
        if is_h_section_outlier and avg_h_length is not None:
            try:
                # Get box counts per line
                box_counts = get_line_box_counts(
                    all_boxes, height, width, base_angle, num_lines
                )
                
                # Find h-sections
                h_sections = find_h_sections_with_indices(line_colors)
                
                # Apply correction
                target_h_length = int(np.round(avg_h_length))
                corrected_line_colors, corrections_made = apply_h_section_correction(
                    line_colors, h_sections, box_counts, target_h_length
                )
                
                # Find which lines changed from blue to red
                corrected_indices = []
                for i in range(len(line_colors)):
                    if line_colors[i] == (255, 0, 0) and corrected_line_colors[i] == (0, 0, 255):
                        corrected_indices.append(i)
                
                # Redraw the corrected lines in red on the image
                if corrected_indices and draw_lines:
                    angle_rad_redraw = np.radians(base_angle)
                    dx_redraw = np.sin(angle_rad_redraw)
                    dy_redraw = -np.cos(angle_rad_redraw)
                    norm_redraw = np.sqrt(dx_redraw**2 + dy_redraw**2)
                    if norm_redraw > 0:
                        dx_redraw /= norm_redraw
                        dy_redraw /= norm_redraw
                    
                    perp_x_redraw = -dy_redraw
                    perp_y_redraw = dx_redraw
                    diag_length_redraw = np.sqrt(width**2 + height**2)
                    max_extent_redraw = diag_length_redraw
                    
                    # Redraw corrected lines in red
                    for line_idx in corrected_indices:
                        offset_redraw = (line_idx - num_lines / 2.0) * (diag_length_redraw / (num_lines - 1 if num_lines > 1 else 1))
                        center_x_redraw = width / 2
                        center_y_redraw = height / 2
                        perp_offset_x_redraw = perp_x_redraw * offset_redraw
                        perp_offset_y_redraw = perp_y_redraw * offset_redraw
                        line_center_x_redraw = center_x_redraw + perp_offset_x_redraw
                        line_center_y_redraw = center_y_redraw + perp_offset_y_redraw
                        
                        start_x_redraw = int(line_center_x_redraw - max_extent_redraw * dx_redraw)
                        start_y_redraw = int(line_center_y_redraw - max_extent_redraw * dy_redraw)
                        end_x_redraw = int(line_center_x_redraw + max_extent_redraw * dx_redraw)
                        end_y_redraw = int(line_center_y_redraw + max_extent_redraw * dy_redraw)
                        
                        # Draw in red (0, 0, 255)
                        cv2.line(image, (start_x_redraw, start_y_redraw), (end_x_redraw, end_y_redraw), (0, 0, 255), thickness=2)
                
                # Use the corrected colors going forward
                line_colors = corrected_line_colors
                
            except Exception as e:
                print(f"  Warning: Could not apply h-section correction: {str(e)}")
        
        # Apply yellow-to-blue conversions if this image has them
        if yellow_to_blue_indices is not None and img_basename in yellow_to_blue_indices:
            yellow_indices_to_convert = yellow_to_blue_indices[img_basename]
            
            # Apply conversions without additional filtering - they were already validated during analysis
            if yellow_indices_to_convert and line_positions is not None and len(line_positions) > 0:
                print(f"  [CONVERTING] {img_basename}: Converting {len(yellow_indices_to_convert)} yellow lines to blue")
                
                # Calculate average blue section size from existing blue sections
                # Find all continuous blue sections in current line_colors
                blue_section_lengths = []
                in_blue_section = False
                section_length = 0
                
                for color in line_colors:
                    if color == (255, 0, 0):  # Blue
                        if not in_blue_section:
                            in_blue_section = True
                            section_length = 1
                        else:
                            section_length += 1
                    else:
                        if in_blue_section:
                            blue_section_lengths.append(section_length)
                            in_blue_section = False
                            section_length = 0
                
                # Don't forget last section if it ends with blue
                if in_blue_section and section_length > 0:
                    blue_section_lengths.append(section_length)
                
                # Calculate average blue section size (half on each side)
                if blue_section_lengths:
                    avg_blue_section_size = int(np.mean(blue_section_lengths))
                    expansion_radius = (avg_blue_section_size - 1) // 2  # How many lines on each side
                else:
                    expansion_radius = 3  # Default fallback
                
                expansion_radius = max(2, expansion_radius)  # At least 2 on each side
                print(f"  [EXPANSION] Creating blue sections with radius {expansion_radius} (avg section size: {avg_blue_section_size if blue_section_lengths else 'N/A'})")
                
                # Convert each yellow index to a full blue section
                lines_converted = 0
                for center_idx in yellow_indices_to_convert:
                    if center_idx < len(line_colors) and center_idx < len(line_positions):
                        # Expand around this index to create a proper section
                        for expand_idx in range(max(0, center_idx - expansion_radius), min(len(line_colors), center_idx + expansion_radius + 1)):
                            if line_colors[expand_idx] != (255, 0, 0):  # Only if not already blue
                                line_colors[expand_idx] = (255, 0, 0)  # Convert to blue
                                # Redraw the line on the image with the new color
                                start_x, start_y, end_x, end_y, _, _ = line_positions[expand_idx]
                                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), thickness=2)
                                lines_converted += 1
                
                print(f"  [RESULT] Created {len(yellow_indices_to_convert)} new blue section(s), total {lines_converted} lines converted")
                
                # Post-processing: Extend blue sections by absorbing adjacent yellow lines
                # Keep expanding until all adjacent yellows are converted to blue
                print(f"  [EXTENDING] Checking for adjacent yellow lines to absorb into blue sections...")
                extended_count = 0
                max_extension_passes = len(line_colors)  # Prevent infinite loops
                pass_count = 0
                
                while pass_count < max_extension_passes:
                    pass_count += 1
                    extended_in_pass = 0
                    
                    # Find all current blue sections (contiguous runs of blue)
                    blue_sections = []
                    in_blue = False
                    start_idx = 0
                    
                    for i in range(len(line_colors)):
                        if line_colors[i] == (255, 0, 0):  # Blue
                            if not in_blue:
                                start_idx = i
                                in_blue = True
                        else:
                            if in_blue:
                                blue_sections.append((start_idx, i - 1))  # End of blue section
                                in_blue = False
                    
                    if in_blue:  # Don't forget last section if it ends with blue
                        blue_sections.append((start_idx, len(line_colors) - 1))
                    
                    # For each blue section, try to extend it if adjacent lines are yellow
                    for section_start, section_end in blue_sections:
                        # Check line before the section
                        if section_start > 0 and line_colors[section_start - 1] == (0, 255, 255):  # Yellow
                            line_colors[section_start - 1] = (255, 0, 0)  # Convert to blue
                            start_x, start_y, end_x, end_y, _, _ = line_positions[section_start - 1]
                            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), thickness=2)
                            extended_in_pass += 1
                            extended_count += 1
                        
                        # Check line after the section
                        if section_end < len(line_colors) - 1 and line_colors[section_end + 1] == (0, 255, 255):  # Yellow
                            line_colors[section_end + 1] = (255, 0, 0)  # Convert to blue
                            start_x, start_y, end_x, end_y, _, _ = line_positions[section_end + 1]
                            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), thickness=2)
                            extended_in_pass += 1
                            extended_count += 1
                    
                    # If nothing was extended in this pass, we're done
                    if extended_in_pass == 0:
                        break
                
                if extended_count > 0:
                    print(f"  [EXTENDED] Absorbed {extended_count} additional yellow lines into blue sections ({pass_count} pass{'es' if pass_count != 1 else ''})")
                
                # Recalculate box colors based on the updated line_colors
                print(f"  [RECALCULATING] Updating box colors based on new blue sections...")
                new_box_colors, high_count, low_count = get_line_colors_for_boxes_with_line_colors(
                    image, all_boxes, base_angle, line_colors=line_colors, 
                    cluster_centers=cluster_centers, num_lines=num_lines
                )
                print(f"  [BOX RECOLORING] Updated: High: {high_count}, Low: {low_count}")
                
                # Redraw boxes that changed color from red to blue
                print(f"  [REDRAWING BOXES] Redrawing boxes with updated colors...")
                boxes_redrawn = 0
                for box_idx, (idx, new_color) in enumerate(new_box_colors):
                    old_color = box_colors[box_idx][1]
                    # If box changed from red to blue, redraw it
                    if old_color == (0, 0, 255) and new_color == (255, 0, 0):  # Red to blue
                        box = all_boxes[idx]
                        x1, y1, x2, y2 = box
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), new_color, thickness=2)
                        boxes_redrawn += 1
                
                # Update box_colors with the new colors
                box_colors = new_box_colors
                print(f"  [BOX REDRAWN] Redrawn {boxes_redrawn} boxes from red to blue")
                
                # CRITICAL: Redraw ALL lines with their final colors to ensure conversions are visible
                print(f"  [FINAL LINE REDRAW] Redrawing ALL {num_lines} lines with final colors...")
                lines_redrawn_final = 0
                if line_positions is not None and len(line_positions) == len(line_colors):
                    for line_idx in range(len(line_positions)):
                        start_x, start_y, end_x, end_y, _, _ = line_positions[line_idx]
                        final_color = line_colors[line_idx]
                        # Redraw this line with its final color
                        cv2.line(image, (start_x, start_y), (end_x, end_y), final_color, thickness=2)
                        lines_redrawn_final += 1
                print(f"  [FINAL LINE REDRAW] Redrawn {lines_redrawn_final} lines with final colors")
        
        
        # Use ALL boxes for drawing (no confidence filtering)
        for box_idx, box in enumerate(all_boxes):
            try:
                if isinstance(box, (list, tuple, np.ndarray)):
                    box = np.array(box, dtype=np.float32)
                    
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        
                        # Get the pre-calculated box color based on intersection voting
                        box_color = (0, 0, 255)  # Default red
                        for calc_idx, calc_color in box_colors:
                            if calc_idx == box_idx:
                                box_color = calc_color
                                break
                        
                        # Convert to int for drawing
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        # Only draw if box has positive area
                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
                            drawn_count += 1
                    elif len(box) == 2:
                        # Assuming [[x1, y1], [x2, y2]] format
                        
                        # Get the pre-calculated box color based on intersection voting
                        box_color = (0, 0, 255)  # Default red
                        for calc_idx, calc_color in box_colors:
                            if calc_idx == box_idx:
                                box_color = calc_color
                                break
                        
                        pt1 = tuple(box[0].astype(int))
                        pt2 = tuple(box[1].astype(int))
                        cv2.rectangle(image, pt1, pt2, box_color, thickness)
                        drawn_count += 1
            except Exception as e:
                print(f"  Warning: Failed to draw box {box_idx}: {str(e)}")
                continue
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save annotated image
        cv2.imwrite(output_path, image)
        print(f"✓ Annotated: {os.path.basename(image_path)} ({drawn_count} boxes) | High: {high_count}, Low: {low_count}")
        return (high_count, low_count) if not collect_only else (high_count, low_count)
    
    except Exception as e:
        print(f"✗ Failed to process image {image_path}: {str(e)}")
        return False if not collect_only else []

def match_and_annotate_images(draw_lines=False, num_lines=100):
    """Match pickle detections with images and draw bounding boxes.
    
    Args:
        draw_lines: If True, also draw the orientation lines on the images
        num_lines: Number of crop rows to detect per image (default: 100)
    """
    print("\n" + "=" * 60)
    print("Drawing Bounding Box Annotations")
    print("=" * 60)
    
    pkl_files = [f for f in os.listdir(PICKLES_DIR) if f.lower().endswith('.pkl')]
    
    if not pkl_files:
        print("No pickle files found.")
        return [], 0
    
    print(f"\nFound {len(pkl_files)} pickle files\n")
    
    # First pass: collect all intersection counts for k-means clustering
    print("Collecting intersection counts for k-means analysis...")
    all_intersection_counts = []
    patterns_collected = []  # Track row patterns from each image
    
    for pkl_filename in pkl_files:
        pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
        
        # Try to match pickle file with image
        base_name = os.path.splitext(pkl_filename)[0]
        matched_image = None
        
        for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
            if os.path.exists(img_path):
                matched_image = img_path
                break
        
        if not matched_image:
            continue
        
        # Load detections and collect counts
        detections = load_pickle_detections(pkl_path)
        if detections is not None:
            counts = draw_bounding_boxes(matched_image, detections, "", collect_only=True, num_lines=num_lines)
            if isinstance(counts, list) and len(counts) > 0:
                all_intersection_counts.extend(counts)
    
    # Calculate k-medians clustering on all intersection counts
    print(f"Processed {len(all_intersection_counts)} total line intersections across all images")
    cluster_centers = kmedians_clustering(all_intersection_counts, k=3)
    
    # Check if the 2 highest means have at least 30% difference
    if len(cluster_centers) == 3:
        c1, c2, c3 = cluster_centers
        percent_diff = (c3 - c2) / c2 if c2 > 0 else float('inf')
        
        print(f"K-medians cluster centers: C1={c1:.1f}, C2={c2:.1f}, C3={c3:.1f}")
        print(f"Difference between C3 and C2: {percent_diff * 100:.1f}%")
        
        # If difference < 30%, drop the highest mean and re-cluster with k=2
        if percent_diff < 0.30:
            print(f"Warning: C3-C2 difference ({percent_diff * 100:.1f}%) is less than 30%")
            print(f"Dropping C3 and re-clustering with k=2...")
            cluster_centers = kmedians_clustering(all_intersection_counts, k=2)
            print(f"K-medians cluster centers (k=2): C1={cluster_centers[0]:.1f}, C2={cluster_centers[1]:.1f}")
            print(f"Rows with counts <= C1 will be classified as 'l', only counts > C1 will be 'h'\n")
        else:
            print(f"Rows with counts <= C2 will be classified as 'l', only counts > C2 will be 'h'\n")
    else:
        print(f"Error: Expected 3 cluster centers, got {len(cluster_centers)}\n")
    
    # Pre-analysis pass: collect gimbal yaw and detected angles to calculate median ratio
    print("Pre-analyzing detected angles and gimbal yaw values...")
    angle_data = {}  # {matched_image: {'gimbal_yaw': X, 'detected_angle': Y}}
    pkl_to_image = {}  # {pkl_filename: matched_image}
    
    for pkl_filename in pkl_files:
        pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
        
        # Try to match pickle file with image
        base_name = os.path.splitext(pkl_filename)[0]
        matched_image = None
        
        for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
            if os.path.exists(img_path):
                matched_image = img_path
                break
        
        if not matched_image:
            continue
        
        # Store mapping for later use
        pkl_to_image[pkl_filename] = matched_image
        
        # Extract gimbal yaw and detected angle
        gimbal_yaw = extract_gimbal_yaw_from_image(matched_image)
        detected_angle = detect_crop_row_angle(matched_image, verbose=False)
        gimbal_yaw_val = gimbal_yaw if gimbal_yaw is not None else 0
        
        # Normalize angles to 0-180 range (crop rows repeat every 180°)
        gimbal_yaw_normalized = gimbal_yaw_val % 180
        detected_angle_normalized = detected_angle % 180
        
        angle_data[matched_image] = {
            'gimbal_yaw': gimbal_yaw_val,
            'gimbal_yaw_normalized': gimbal_yaw_normalized,
            'detected_angle': detected_angle,
            'detected_angle_normalized': detected_angle_normalized
        }
    
    # Calculate median offset and identify outliers
    # offset = detected_angle_normalized - gimbal_yaw_normalized (both in 0-180 range)
    # Normalize offset to -90 to +90 range to handle 180° cycle
    angle_offsets = []
    offset_data = {}  # Store offsets for each image
    
    for matched_image, data in angle_data.items():
        # Calculate raw offset
        offset = data['detected_angle_normalized'] - data['gimbal_yaw_normalized']
        
        # Normalize offset to -90 to +90 range (since angles repeat every 180°)
        if offset > 90:
            offset = offset - 180
        elif offset < -90:
            offset = offset + 180
        
        angle_offsets.append(offset)
        offset_data[matched_image] = offset
    
    median_offset = np.median(angle_offsets) if angle_offsets else 0
    
    # Identify outlier images that need angle correction
    corrected_angles = {}  # {matched_image: corrected_angle}
    threshold = 30  # 30° threshold
    
    for matched_image, data in angle_data.items():
        if matched_image in offset_data:
            offset = offset_data[matched_image]
            deviation = abs(offset - median_offset)
            if deviation > threshold:
                # Calculate corrected angle: gimbal_normalized + median_offset, normalized to 0-180
                corrected_angle_raw = data['gimbal_yaw_normalized'] + median_offset
                corrected_angle = corrected_angle_raw % 180
                corrected_angles[matched_image] = corrected_angle
                print(f"  → Outlier detected: {os.path.basename(matched_image)}")
                print(f"    Gimbal Yaw: {data['gimbal_yaw']:.1f}° (normalized: {data['gimbal_yaw_normalized']:.1f}°)")
                print(f"    Detected: {data['detected_angle']:.1f}° (normalized: {data['detected_angle_normalized']:.1f}°)")
                print(f"    Offset: {offset:+.1f}° (deviation: {deviation:+.1f}° from median {median_offset:+.1f}°)")
                print(f"    Using corrected angle: {corrected_angle:.1f}° (gimbal_normalized: {data['gimbal_yaw_normalized']:.1f}° + median offset: {median_offset:+.1f}°)")
    
    # Analysis pass: collect ACTUAL line colors from drawing and analyze h-sections BEFORE drawing
    print("\n" + "=" * 60)
    print("H-Section Analysis (Using Actual Line Colors)")
    print("=" * 60)
    print("Analyzing continuous 'h' section lengths from actual line drawing...\n")
    
    all_h_section_lengths = []  # Track all h-section lengths across all images
    all_red_section_lengths = []  # Track red sections between blue sections
    
    for pkl_filename in pkl_files:
        try:
            pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
            
            # Try to match pickle file with image
            base_name = os.path.splitext(pkl_filename)[0]
            matched_image = None
            
            for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
                if os.path.exists(img_path):
                    matched_image = img_path
                    break
            
            if not matched_image:
                continue
            
            # Load detections and get ACTUAL line colors from drawing function
            detections = load_pickle_detections(pkl_path)
            if detections is not None:
                # Use corrected angle if available, otherwise use detected angle
                if matched_image in corrected_angles:
                    angle_to_use = corrected_angles[matched_image]
                elif matched_image in angle_data:
                    angle_to_use = angle_data[matched_image]['detected_angle']
                else:
                    continue
                
                # Get the ACTUAL line colors by calling draw_orientation_lines_with_counts
                image = cv2.imread(matched_image)
                if image is not None:
                    filtered_boxes = extract_boxes_from_sahi_result(detections, confidence_threshold=0.2)
                    if filtered_boxes:
                        # This uses the ACTUAL counting method with smoothing
                        _, actual_line_colors, _ = draw_orientation_lines_with_counts(image.copy(), filtered_boxes, angle_to_use, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
                        
                        if actual_line_colors:
                            # Convert line colors to h/l pattern: blue (255,0,0)='h', red (0,0,255)='l'
                            pattern = ""
                            for color in actual_line_colors:
                                if color == (255, 0, 0):  # Blue = high
                                    pattern += "h"
                                else:  # Red = low
                                    pattern += "l"
                            
                            # Analyze h-sections in the ACTUAL line color pattern
                            h_sections = analyze_h_sections(pattern)
                            # Only include images with at least 2 blue sections
                            if h_sections and len(h_sections) >= 2:
                                all_h_section_lengths.extend(h_sections)
                                
                                # Analyze red sections between blue sections (yellow counts as red)
                                red_sections = analyze_red_sections_between_blue(actual_line_colors)
                                if red_sections:
                                    all_red_section_lengths.extend(red_sections)
        except Exception as e:
            continue
    
    # Calculate per-image average h-section lengths
    if all_h_section_lengths:
        # For each image, calculate average h-section length
        per_image_averages = []  # List of (image_name, average_h_length, num_h_sections)
        all_h_lengths_flat = []  # Flatten all h-section lengths for global stats
        
        for pkl_filename in pkl_files:
            try:
                pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
                
                # Try to match pickle file with image
                base_name = os.path.splitext(pkl_filename)[0]
                matched_image = None
                
                for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
                    if os.path.exists(img_path):
                        matched_image = img_path
                        break
                
                if not matched_image:
                    continue
                
                # Load detections and get ACTUAL line colors from drawing function
                detections = load_pickle_detections(pkl_path)
                if detections is not None:
                    # Use corrected angle if available, otherwise use detected angle
                    if matched_image in corrected_angles:
                        angle_to_use = corrected_angles[matched_image]
                    elif matched_image in angle_data:
                        angle_to_use = angle_data[matched_image]['detected_angle']
                    else:
                        continue
                    
                    # Get the ACTUAL line colors by calling draw_orientation_lines_with_counts
                    image = cv2.imread(matched_image)
                    if image is not None:
                        filtered_boxes = extract_boxes_from_sahi_result(detections, confidence_threshold=0.2)
                        if filtered_boxes:
                            # This uses the ACTUAL counting method with smoothing
                            _, actual_line_colors, _ = draw_orientation_lines_with_counts(image.copy(), filtered_boxes, angle_to_use, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
                            
                            if actual_line_colors:
                                # Convert line colors to h/l pattern: blue (255,0,0)='h', red (0,0,255)='l'
                                pattern = ""
                                for color in actual_line_colors:
                                    if color == (255, 0, 0):  # Blue = high
                                        pattern += "h"
                                    else:  # Red = low
                                        pattern += "l"
                                
                                # Analyze h-sections in the ACTUAL line color pattern
                                h_sections = analyze_h_sections(pattern)
                                # Only include images with at least 2 blue sections
                                if h_sections and len(h_sections) >= 2:
                                    image_avg = np.median(h_sections)
                                    per_image_averages.append((base_name, image_avg, len(h_sections)))
                                    all_h_lengths_flat.extend(h_sections)
            except Exception as e:
                continue
        
        # Calculate overall statistics
        overall_avg_h_length = np.median([avg for _, avg, _ in per_image_averages])
        
        print(f"H-Section Statistics (Per-Image Median):")
        print(f"  Number of images analyzed: {len(per_image_averages)}")
        print(f"  MEDIAN of per-image medians: {overall_avg_h_length:.2f} rows")
        
        if all_h_lengths_flat:
            min_h_length = np.min(all_h_lengths_flat)
            max_h_length = np.max(all_h_lengths_flat)
            std_h_length = np.std(all_h_lengths_flat)
            total_h_sections = len(all_h_lengths_flat)
            
            print(f"  Total h-sections found: {total_h_sections}")
            print(f"  Min h-section length: {min_h_length} rows")
            print(f"  Max h-section length: {max_h_length} rows")
            print(f"  Std deviation (of all sections): {std_h_length:.2f} rows")
            print(f"\n  Per-image medians:")
            for image_name, avg, num_sections in sorted(per_image_averages, key=lambda x: x[1], reverse=True):
                print(f"    {image_name}: {avg:.2f} rows ({num_sections} h-sections)")
            
            # Find images with h-sections 22% or more larger than average
            threshold = overall_avg_h_length * 1.22
            large_h_section_images = [
                (img_name, avg, num_sections)
                for img_name, avg, num_sections in per_image_averages
                if avg >= threshold
            ]
            
            print(f"\n  Images with h-sections ≥22% larger than average ({threshold:.2f} rows):")
            if large_h_section_images:
                for image_name, med, num_sections in sorted(large_h_section_images, key=lambda x: x[1], reverse=True):
                    percentage_larger = ((med - overall_avg_h_length) / overall_avg_h_length) * 100
                    print(f"    → {image_name}: {med:.2f} rows ({percentage_larger:.1f}% larger, {num_sections} h-sections)")
            else:
                print(f"    (No images found with h-sections ≥{threshold:.2f} rows)")
            
            print(f"\n  H-section length distribution (all sections):")
            
            # Create histogram
            from collections import Counter
            length_counts = Counter(int(round(x)) for x in all_h_lengths_flat)
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                percentage = (count / total_h_sections) * 100
                bar = "█" * int(percentage / 2)
                print(f"    {length:3} rows: {count:3} sections ({percentage:5.1f}%) {bar}")
        
        avg_h_length = overall_avg_h_length
    else:
        avg_h_length = 0
        print("  ⊘ No h-sections found in patterns")
    
    # RED SECTION ANALYSIS (between blue sections)
    print(f"\n" + "=" * 60)
    print("Red Section Analysis (Red sections between Blue sections)")
    print("=" * 60)
    print("Yellow lines count as red in this analysis.\n")
    
    overall_avg_red_length = 0  # Initialize to 0
    avg_red_sections_per_image = 0  # Initialize to 0
    
    if all_red_section_lengths:
        # Calculate per-image red section statistics
        per_image_red_averages = []
        per_image_red_section_counts = []  # Track number of red sections per image
        all_red_lengths_flat = all_red_section_lengths.copy()
        
        for pkl_filename in pkl_files:
            try:
                pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
                
                # Try to match pickle file with image
                base_name = os.path.splitext(pkl_filename)[0]
                matched_image = None
                
                for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
                    if os.path.exists(img_path):
                        matched_image = img_path
                        break
                
                if not matched_image:
                    continue
                
                # Load detections and get ACTUAL line colors
                detections = load_pickle_detections(pkl_path)
                if detections is not None:
                    # Use corrected angle if available, otherwise use detected angle
                    if matched_image in corrected_angles:
                        angle_to_use = corrected_angles[matched_image]
                    elif matched_image in angle_data:
                        angle_to_use = angle_data[matched_image]['detected_angle']
                    else:
                        continue
                    
                    # Get the ACTUAL line colors by calling draw_orientation_lines_with_counts
                    image = cv2.imread(matched_image)
                    if image is not None:
                        filtered_boxes = extract_boxes_from_sahi_result(detections, confidence_threshold=0.2)
                        if filtered_boxes:
                            _, actual_line_colors, _ = draw_orientation_lines_with_counts(image.copy(), filtered_boxes, angle_to_use, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
                            
                            if actual_line_colors:
                                # First check if this image has at least 2 blue sections
                                pattern = ""
                                for color in actual_line_colors:
                                    if color == (255, 0, 0):  # Blue = high
                                        pattern += "h"
                                    else:  # Red = low
                                        pattern += "l"
                                h_sections = analyze_h_sections(pattern)
                                
                                # Only include images with at least 2 blue sections
                                if h_sections and len(h_sections) >= 2:
                                    # Analyze red sections between blue sections
                                    image_red_sections = analyze_red_sections_between_blue(actual_line_colors)
                                    if image_red_sections:
                                        image_red_avg = np.median(image_red_sections)
                                        num_red_sections = len(image_red_sections)
                                        per_image_red_averages.append((base_name, image_red_avg, num_red_sections))
                                        per_image_red_section_counts.append(num_red_sections)
                                    all_red_section_lengths.extend(image_red_sections)
            except Exception as e:
                continue
        
        # Calculate overall red statistics
        overall_avg_red_length = np.median([avg for _, avg, _ in per_image_red_averages]) if per_image_red_averages else 0
        avg_red_sections_per_image = np.mean(per_image_red_section_counts) if per_image_red_section_counts else 0
        
        avg_red_length = np.mean(all_red_section_lengths)
        median_red_length = np.median(all_red_section_lengths)
        min_red_length = np.min(all_red_section_lengths)
        max_red_length = np.max(all_red_section_lengths)
        std_red_length = np.std(all_red_section_lengths)
        total_red_sections = len(all_red_section_lengths)
        
        print(f"Red Section Statistics (between Blue sections):")
        print(f"  Number of images analyzed: {len(per_image_red_averages)}")
        print(f"  Total red sections (between blue): {total_red_sections}")
        print(f"  MEDIAN of per-image medians: {overall_avg_red_length:.2f} rows")
        print(f"  AVERAGE red section length (all): {avg_red_length:.2f} rows")
        print(f"  MEDIAN red section length (all): {median_red_length:.2f} rows")
        print(f"  Min red section length: {min_red_length} rows")
        print(f"  Max red section length: {max_red_length} rows")
        print(f"  Std deviation: {std_red_length:.2f} rows")
        
        print(f"\n  Per-image medians:")
        for image_name, avg, num_sections in sorted(per_image_red_averages, key=lambda x: x[1], reverse=True):
            print(f"    {image_name}: {avg:.2f} rows ({num_sections} red-sections)")
        
        print(f"\n  Red section length distribution:")
        
        # Create histogram
        from collections import Counter
        red_length_counts = Counter(int(round(x)) for x in all_red_section_lengths)
        for length in sorted(red_length_counts.keys()):
            count = red_length_counts[length]
            percentage = (count / total_red_sections) * 100
            bar = "█" * int(percentage / 2)
            print(f"    {length:3} rows: {count:3} sections ({percentage:5.1f}%) {bar}")
    else:
        print(f"  ⊘ No red sections found between blue sections")
    
    # SUMMARY OF SECTION STATISTICS
    print(f"\n" + "=" * 60)
    print("SECTION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Blue Section (H-section) Statistics:")
    if all_h_section_lengths and 'per_image_averages' in locals() and per_image_averages:
        avg_blue_sections_per_image = np.mean([num for _, _, num in per_image_averages])
        print(f"  Average rows per blue section: {overall_avg_h_length:.2f} rows")
        print(f"  Average blue sections per image: {avg_blue_sections_per_image:.1f} sections")
        print(f"  Total blue sections across all images: {len(all_h_section_lengths)}")
    else:
        print(f"  ⊘ No blue sections found")
    
    print(f"\nRed Section (between Blue) Statistics:")
    if all_red_section_lengths and 'per_image_red_averages' in locals() and per_image_red_averages:
        print(f"  Average rows per red section: {overall_avg_red_length:.2f} rows")
        print(f"  Average red sections per image: {avg_red_sections_per_image:.1f} sections")
        print(f"  Total red sections across all images: {len(all_red_section_lengths)}")
    else:
        print(f"  ⊘ No red sections found")
    
    # UNIFIED EXPERT ANALYSIS: Yellow-to-Blue Conversion Opportunities
    print(f"\n" + "=" * 60)
    print("YELLOW-TO-BLUE CONVERSION ANALYSIS")
    print("=" * 60)
    print("Analyzing yellow line conversion opportunities across all criteria...\n")
    
    avg_blue_sections_threshold = avg_blue_sections_per_image if 'avg_blue_sections_per_image' in locals() else 0
    red_section_tolerance = 0.30  # Accept red sections within ±30% of average
    large_red_threshold = overall_avg_red_length * 0.20 if overall_avg_red_length > 0 else 0  # 20% of average
    
    if overall_avg_red_length > 0 and avg_blue_sections_threshold > 0:
        min_acceptable_red = overall_avg_red_length * (1 - red_section_tolerance)
        max_acceptable_red = overall_avg_red_length * (1 + red_section_tolerance)
        
        print(f"Target red section length: {overall_avg_red_length:.2f} rows")
        print(f"Acceptable range: {min_acceptable_red:.2f} - {max_acceptable_red:.2f} rows (±{red_section_tolerance*100:.0f}%)")
        print(f"Large red threshold: ≥{large_red_threshold:.2f} rows\n")
        
        conversion_candidates = []
        
        # Unified analysis: single pass through all images
        for pkl_filename in pkl_files:
            try:
                pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
                base_name = os.path.splitext(pkl_filename)[0]
                matched_image = None
                
                for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
                    if os.path.exists(img_path):
                        matched_image = img_path
                        break
                
                if not matched_image:
                    continue
                
                detections = load_pickle_detections(pkl_path)
                if detections is None:
                    continue
                
                # Get angle
                if matched_image in corrected_angles:
                    angle_to_use = corrected_angles[matched_image]
                elif matched_image in angle_data:
                    angle_to_use = angle_data[matched_image]['detected_angle']
                else:
                    continue
                
                # Get actual line colors
                image = cv2.imread(matched_image)
                if image is None:
                    continue
                
                filtered_boxes = extract_boxes_from_sahi_result(detections, confidence_threshold=0.2)
                if not filtered_boxes:
                    continue
                
                _, actual_line_colors, _ = draw_orientation_lines_with_counts(image.copy(), filtered_boxes, angle_to_use, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
                
                # Count current blue sections
                pattern = ""
                for color in actual_line_colors:
                    if color == (255, 0, 0):
                        pattern += "h"
                    else:
                        pattern += "l"
                
                h_sections = analyze_h_sections(pattern)
                current_blue_count = len(h_sections)
                
                # Analyze red sections
                red_sections = analyze_red_sections_between_blue(actual_line_colors)
                large_red_sections = [rs for rs in red_sections if rs >= large_red_threshold]
                
                # Find yellow lines
                yellow_indices = [i for i, color in enumerate(actual_line_colors) if color == (0, 255, 255)]
                
                if not yellow_indices:
                    continue
                
                # Determine problem criteria
                has_fewer_blue_sections = current_blue_count < avg_blue_sections_threshold
                has_large_red_sections = len(large_red_sections) > 0
                
                # Skip if no problems identified
                if not (has_fewer_blue_sections or has_large_red_sections):
                    continue
                
                # Try converting yellow lines iteratively until all large red sections are broken up
                best_conversion = None
                best_score = -1
                best_large_reduction = -1
                
                # For images with large red sections, keep converting more yellows until fixed
                if has_large_red_sections:
                    # Iteratively find the best conversion strategy
                    test_colors = actual_line_colors.copy()
                    remaining_large_red = len(large_red_sections)
                    converted_yellows = []
                    iteration = 0
                    max_iterations = len(yellow_indices)  # Safety limit
                    
                    while remaining_large_red > 0 and iteration < max_iterations and len(converted_yellows) < len(yellow_indices):
                        iteration += 1
                        best_iteration_reduction = 0
                        best_next_yellow_idx = -1
                        best_iteration_colors = None
                        
                        # Try converting each remaining yellow line individually
                        for yellow_idx in yellow_indices:
                            if yellow_idx in converted_yellows:
                                continue  # Already converted
                            
                            # Test converting this yellow
                            test_colors_candidate = test_colors.copy()
                            test_colors_candidate[yellow_idx] = (255, 0, 0)
                            
                            # Calculate resulting red sections
                            test_red_sections = analyze_red_sections_between_blue(test_colors_candidate)
                            test_large_red = [rs for rs in test_red_sections if rs >= large_red_threshold]
                            reduction = (len(test_large_red) - remaining_large_red) / remaining_large_red if remaining_large_red > 0 else 0
                            
                            # Choose the yellow that gives best reduction (most negative = best improvement)
                            if reduction < best_iteration_reduction:  # More negative is better (bigger reduction)
                                best_iteration_reduction = reduction
                                best_next_yellow_idx = yellow_idx
                                best_iteration_colors = test_colors_candidate
                        
                        # If we found an improvement, apply it
                        if best_next_yellow_idx >= 0 and best_iteration_colors is not None:
                            test_colors = best_iteration_colors
                            converted_yellows.append(best_next_yellow_idx)
                            remaining_large_red += int(best_iteration_reduction)  # Update remaining count
                        else:
                            # No more improvements possible
                            break
                    
                    # Evaluate the final conversion
                    if converted_yellows:
                        test_pattern = ""
                        for color in test_colors:
                            if color == (255, 0, 0):
                                test_pattern += "h"
                            else:
                                test_pattern += "l"
                        
                        test_h_sections = analyze_h_sections(test_pattern)
                        test_red_sections = analyze_red_sections_between_blue(test_colors)
                        
                        if test_h_sections and test_red_sections:
                            # Calculate final reduction
                            final_large_red = [rs for rs in test_red_sections if rs >= large_red_threshold]
                            final_large_reduction = (len(large_red_sections) - len(final_large_red)) / len(large_red_sections)
                            
                            best_conversion = {
                                'num_yellows': len(converted_yellows),
                                'yellow_indices': converted_yellows,
                                'new_blue_count': len(test_h_sections),
                                'new_red_sections': test_red_sections,
                                'score': final_large_reduction,
                                'large_reduction': final_large_reduction
                            }
                else:
                    # No large red sections - use standard scoring
                    best_large_reduction = -1
                    
                    # Test converting different numbers of yellow lines
                    for num_yellows_to_convert in range(1, len(yellow_indices) + 1):
                        test_colors = actual_line_colors.copy()
                        for i in range(min(num_yellows_to_convert, len(yellow_indices))):
                            test_colors[yellow_indices[i]] = (255, 0, 0)  # Convert to blue
                        
                        # Calculate resulting blue and red sections
                        test_pattern = ""
                        for color in test_colors:
                            if color == (255, 0, 0):
                                test_pattern += "h"
                            else:
                                test_pattern += "l"
                        
                        test_h_sections = analyze_h_sections(test_pattern)
                        test_red_sections = analyze_red_sections_between_blue(test_colors)
                        
                        if not (test_h_sections and len(test_h_sections) >= 2 and test_red_sections):
                            continue
                        
                        # Evaluate conversion based on identified problems
                        score_components = []
                        
                        # Component 1: Improve low blue section count
                        if has_fewer_blue_sections:
                            new_blue_count = len(test_h_sections)
                            blue_improvement = (new_blue_count - current_blue_count) / avg_blue_sections_threshold
                            score_components.append(blue_improvement)
                        
                        # Component 2: Create good red sections
                        good_sections = sum(1 for rs in test_red_sections if min_acceptable_red <= rs <= max_acceptable_red)
                        red_quality = good_sections / len(test_red_sections) if test_red_sections else 0
                        avg_distance = np.mean([abs(rs - overall_avg_red_length) for rs in test_red_sections]) if test_red_sections else 0
                        red_fit = red_quality - (avg_distance / overall_avg_red_length) * 0.1 if overall_avg_red_length > 0 else 0
                        score_components.append(red_fit)
                        
                        # Combined score: average of all relevant components
                        combined_score = np.mean(score_components) if score_components else 0
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_conversion = {
                                'num_yellows': num_yellows_to_convert,
                                'yellow_indices': yellow_indices[:num_yellows_to_convert],
                                'new_blue_count': len(test_h_sections),
                                'new_red_sections': test_red_sections,
                                'score': combined_score,
                                'large_reduction': 0
                            }
                
                # Record if conversion is beneficial
                should_record = False
                if best_conversion is not None:
                    if has_large_red_sections:
                        # Accept if we reduced large red sections
                        should_record = best_conversion.get('large_reduction', 0) > 0
                    else:
                        # Accept if score > 0.1
                        should_record = best_conversion['score'] > 0.1
                
                if should_record and best_conversion is not None:
                    new_large_red_count = sum(1 for rs in best_conversion['new_red_sections'] if rs >= large_red_threshold)
                    conversion_candidates.append({
                        'image': base_name,
                        'problems': {
                            'fewer_blue_sections': has_fewer_blue_sections,
                            'large_red_sections': has_large_red_sections
                        },
                        'current_blue_sections': current_blue_count,
                        'current_large_red_count': len(large_red_sections),
                        'current_red_sections': red_sections,
                        'new_blue_sections': best_conversion['new_blue_count'],
                        'new_large_red_count': new_large_red_count,
                        'new_red_sections': best_conversion['new_red_sections'],
                        'yellow_count': len(yellow_indices),
                        'yellows_to_convert': best_conversion['num_yellows'],
                        'yellow_indices_to_convert': best_conversion['yellow_indices'],
                        'score': best_conversion['score']
                    })
            except Exception as e:
                continue
        
        # Print results grouped by problem type
        if conversion_candidates:
            print(f"Images with conversion opportunities ({len(conversion_candidates)} found):\n")
            
            # Separate by problem type
            fewer_blue_candidates = [c for c in conversion_candidates if c['problems']['fewer_blue_sections']]
            large_red_candidates = [c for c in conversion_candidates if c['problems']['large_red_sections']]
            both_candidates = [c for c in conversion_candidates if c['problems']['fewer_blue_sections'] and c['problems']['large_red_sections']]
            
            # Print "both" candidates first (most interesting)
            if both_candidates:
                print(f"  [{len(both_candidates)}] Images with BOTH fewer blue sections AND large red sections:\n")
                for candidate in sorted(both_candidates, key=lambda x: x['score'], reverse=True):
                    print(f"    {candidate['image']} (score: {candidate['score']:.3f})")
                    print(f"      Blue sections: {candidate['current_blue_sections']} → {candidate['new_blue_sections']}")
                    print(f"      Large red sections: {candidate['current_large_red_count']} → {candidate['new_large_red_count']}")
                    print(f"      Yellow lines: {candidate['yellow_count']} total → Convert {candidate['yellows_to_convert']} to blue")
                    print(f"      New red sections: {candidate['new_red_sections']}\n")
            
            # Print "fewer blue" candidates
            if fewer_blue_candidates and fewer_blue_candidates not in both_candidates:
                print(f"  [{len(fewer_blue_candidates)}] Images with fewer blue sections than average:\n")
                for candidate in sorted(fewer_blue_candidates, key=lambda x: x['score'], reverse=True):
                    if candidate not in both_candidates:
                        print(f"    {candidate['image']} (score: {candidate['score']:.3f})")
                        print(f"      Blue sections: {candidate['current_blue_sections']} → {candidate['new_blue_sections']}")
                        print(f"      Yellow lines: {candidate['yellow_count']} total → Convert {candidate['yellows_to_convert']} to blue")
                        print(f"      New red sections: {candidate['new_red_sections']}\n")
            
            # Print "large red" candidates
            if large_red_candidates and large_red_candidates not in both_candidates:
                print(f"  [{len(large_red_candidates)}] Images with abnormally large red sections:\n")
                for candidate in sorted(large_red_candidates, key=lambda x: x['score'], reverse=True):
                    if candidate not in both_candidates:
                        print(f"    {candidate['image']} (score: {candidate['score']:.3f})")
                        print(f"      Large red sections: {candidate['current_large_red_count']} → {candidate['new_large_red_count']} (threshold: {large_red_threshold:.2f} rows)")
                        print(f"      Yellow lines: {candidate['yellow_count']} total → Convert {candidate['yellows_to_convert']} to blue")
                        print(f"      New red sections: {candidate['new_red_sections']}\n")
        else:
            print(f"⊘ No images found where yellow-to-blue conversion would provide meaningful improvements.")
    else:
        print(f"⊘ Cannot perform analysis (insufficient statistics)")
    
    # Create mapping of image names to yellow-to-blue conversions to apply
    conversions_to_apply = {}
    if conversion_candidates:
        print(f"\n" + "=" * 60)
        print("APPLYING YELLOW-TO-BLUE CONVERSIONS")
        print("=" * 60)
        print(f"Converting yellow lines to blue for {len(conversion_candidates)} images...\n")
        
        for candidate in conversion_candidates:
            img_name = candidate['image']
            yellow_indices = candidate['yellow_indices_to_convert']
            conversions_to_apply[img_name] = yellow_indices
            print(f"  {img_name}: Converting {len(yellow_indices)} yellow lines ({yellow_indices}) to blue")
    
    # Define outlier_image_base_names early so it's available in the drawing loop
    outlier_image_base_names = [img_name for img_name, _, _ in large_h_section_images] if 'large_h_section_images' in locals() else []
    
    annotated_count = 0
    annotation_data = []
    line_stats = []  # Track (image_name, high_count, low_count) for each image
    
    print(f"\nStarting second pass: processing {len(pkl_files)} pickle files...")
    
    for pkl_filename in pkl_files:
        try:
            pkl_path = os.path.join(PICKLES_DIR, pkl_filename)
            
            # Try to match pickle file with image
            base_name = os.path.splitext(pkl_filename)[0]
            matched_image = None
            
            for img_ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                img_path = os.path.join(IMAGES_DIR, base_name + img_ext)
                if os.path.exists(img_path):
                    matched_image = img_path
                    break
            
            if not matched_image:
                continue
            
            # Load detections and draw with colors
            detections = load_pickle_detections(pkl_path)
            if detections is not None:
                # Use corrected angle if available (for outlier images), otherwise use detected angle
                if matched_image in corrected_angles:
                    angle_to_use = corrected_angles[matched_image]
                    is_corrected = True
                elif matched_image in angle_data:
                    angle_to_use = angle_data[matched_image]['detected_angle']
                    is_corrected = False
                else:
                    print(f"⊘ Skipping {base_name}: angle data not found")
                    continue
            
            # Get gimbal yaw for tracking (should exist if angle_data has the key)
            gimbal_yaw = angle_data[matched_image].get('gimbal_yaw', 0)
            
            output_path = os.path.join(ANNOTATED_DIR, base_name + ".jpg")
            if is_corrected:
                print(f"  [CORRECTING] {base_name}: Using corrected angle {angle_to_use:.1f}°")
            
            # Check if this is an outlier image that needs h-section correction
            is_outlier = base_name in outlier_image_base_names
            
            result = draw_bounding_boxes(matched_image, detections, output_path, cluster_centers=tuple(cluster_centers), draw_lines=draw_lines, override_angle=angle_to_use, num_lines=num_lines, is_h_section_outlier=is_outlier, avg_h_length=avg_h_length, yellow_to_blue_indices=conversions_to_apply)
            
            if result and isinstance(result, tuple) and len(result) == 2:
                high_count, low_count = result
                annotated_count += 1
                
                # Extract row pattern from this image
                image = cv2.imread(matched_image)
                if image is not None:
                    filtered_boxes = extract_boxes_from_sahi_result(detections, confidence_threshold=0.1)
                    if filtered_boxes:
                        original_pattern, expanded_pattern = get_row_pattern(image, filtered_boxes, angle_to_use, cluster_centers=tuple(cluster_centers), num_lines=num_lines)
                        # Use expanded pattern for visualization/reporting (shows the expansion effect)
                        patterns_collected.append((base_name, expanded_pattern))
                
                # Track line statistics with gimbal yaw and angle offset
                detected_angle = angle_data[matched_image]['detected_angle']
                detected_angle_normalized = angle_data[matched_image]['detected_angle_normalized']
                gimbal_yaw_val = gimbal_yaw if gimbal_yaw is not None else 0
                gimbal_yaw_normalized = angle_data[matched_image]['gimbal_yaw_normalized']
                
                # Calculate offset (normalized to -90 to +90 range)
                offset = detected_angle_normalized - gimbal_yaw_normalized
                if offset > 90:
                    offset = offset - 180
                elif offset < -90:
                    offset = offset + 180
                
                # Calculate offset for used angle (corrected if applicable)
                used_angle_normalized = angle_to_use % 180
                used_offset = used_angle_normalized - gimbal_yaw_normalized
                if used_offset > 90:
                    used_offset = used_offset - 180
                elif used_offset < -90:
                    used_offset = used_offset + 180
                
                # For corrected images, calculate what counts would have been without correction
                if is_corrected:
                    # Re-calculate counts using the detected (wrong) angle to show the difference
                    from PIL import Image as PILImage
                    temp_image = cv2.imread(matched_image)
                    if temp_image is not None:
                        boxes_check = extract_boxes_from_sahi_result(detections)
                        if boxes_check:
                            wrong_colors, wrong_high, wrong_low = get_line_colors_for_boxes(temp_image, boxes_check, detected_angle, cluster_centers=tuple(cluster_centers), num_lines=num_lines)
                        else:
                            wrong_high, wrong_low = 0, 0
                    else:
                        wrong_high, wrong_low = 0, 0
                else:
                    wrong_high, wrong_low = high_count, low_count
                
                line_stats.append({
                    'image_name': os.path.basename(matched_image),
                    'high_count': high_count,
                    'low_count': low_count,
                    'wrong_high_count': wrong_high,
                    'wrong_low_count': wrong_low,
                    'gimbal_yaw': gimbal_yaw_val,
                    'gimbal_yaw_normalized': gimbal_yaw_normalized,
                    'detected_angle': detected_angle,
                    'detected_angle_normalized': detected_angle_normalized,
                    'angle_used': angle_to_use,
                    'offset': offset,
                    'used_offset': used_offset,
                    'was_corrected': is_corrected
                })
                
                # Extract data for GeoJSON
                detection_count = len(extract_boxes_from_sahi_result(detections))
                gps_coords = extract_gps_from_image(matched_image)
                
                annotation_data.append({
                    'image_name': os.path.basename(matched_image),
                    'tassel_count': detection_count,
                    'gps': gps_coords
                })
        except Exception as e:
            print(f"✗ Error processing {pkl_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate and report line distribution statistics
    if line_stats:
        print("\n" + "=" * 60)
        print("Line Distribution Analysis")
        print("=" * 60)
        
        high_counts = [stat['high_count'] for stat in line_stats]
        low_counts = [stat['low_count'] for stat in line_stats]
        
        avg_high = np.mean(high_counts)
        avg_low = np.mean(low_counts)
        std_high = np.std(high_counts)
        std_low = np.std(low_counts)
        
        print(f"\nAverage line distribution across {len(line_stats)} images:")
        print(f"  High lines: {avg_high:.1f} ± {std_high:.1f}")
        print(f"  Low lines: {avg_low:.1f} ± {std_low:.1f}")
        
        # Identify anomalous images - significant deviation from average
        print(f"\nImages with significant deviation from average:")
        threshold_std = 1.0  # Flag images ±1 std dev from average
        
        anomalies = []
        for stat in line_stats:
            high_dev = abs(stat['high_count'] - avg_high) / std_high if std_high > 0 else 0
            low_dev = abs(stat['low_count'] - avg_low) / std_low if std_low > 0 else 0
            
            if high_dev > threshold_std or low_dev > threshold_std:
                lacking_type = "HIGH" if stat['high_count'] < avg_high else "LOW"
                anomalies.append({
                    'image': stat['image_name'],
                    'high': stat['high_count'],
                    'low': stat['low_count'],
                    'lacking': lacking_type,
                    'high_dev': high_dev,
                    'low_dev': low_dev
                })
        
        if anomalies:
            for anom in sorted(anomalies, key=lambda x: max(x['high_dev'], x['low_dev']), reverse=True):
                print(f"  • {anom['image']}")
                print(f"    High: {anom['high']} (avg: {avg_high:.1f})")
                print(f"    Low: {anom['low']} (avg: {avg_low:.1f})")
                print(f"    ➔ LACKING {anom['lacking']} LINES")
        else:
            print("  (All images are within 1 standard deviation of average)")
        
        # Analyze detected angle to gimbal yaw offset
        print("\n" + "=" * 60)
        print("Crop Row Angle vs Gimbal Yaw Analysis (Offset in 0-180° intervals)")
        print("=" * 60)
        
        angle_offsets = [stat['offset'] for stat in line_stats]
        
        if angle_offsets:
            median_offset = np.median(angle_offsets)
            print(f"\n📊 Median angle offset (detected - gimbal, normalized): {median_offset:+.1f}°")
            
            # Show corrected images
            corrected_images = [stat for stat in line_stats if stat.get('was_corrected', False)]
            if corrected_images:
                print(f"\n✓ Applied angle corrections to {len(corrected_images)} outlier images:")
                for stat in sorted(corrected_images, key=lambda x: abs(x['offset'] - median_offset), reverse=True):
                    print(f"  • {stat['image_name']}")
                    print(f"    Gimbal: {stat['gimbal_yaw']:.1f}° (normalized: {stat['gimbal_yaw_normalized']:.1f}°)")
                    print(f"    Detected: {stat['detected_angle']:.1f}° (normalized: {stat['detected_angle_normalized']:.1f}°)")
                    print(f"    Offset: {stat['offset']:+.1f}° (deviation: {abs(stat['offset'] - median_offset):+.1f}° from median)")
                    print(f"    Corrected angle: {stat['angle_used']:.1f}° (offset: {stat['used_offset']:+.1f}°)")
                    print(f"    Without correction: High={stat['wrong_high_count']}, Low={stat['wrong_low_count']}")
                    print(f"    With correction:    High={stat['high_count']}, Low={stat['low_count']}")
            
            # Find any remaining images more than 30° off from median offset
            threshold = 30  # 30° threshold
            offset_outliers = []
            
            for stat in line_stats:
                deviation = abs(stat['offset'] - median_offset)
                if deviation > threshold:
                    offset_outliers.append({
                        'image': stat['image_name'],
                        'gimbal_yaw': stat['gimbal_yaw'],
                        'gimbal_yaw_normalized': stat['gimbal_yaw_normalized'],
                        'detected_angle': stat['detected_angle'],
                        'detected_angle_normalized': stat['detected_angle_normalized'],
                        'offset': stat['offset'],
                        'deviation': deviation
                    })
            
            if offset_outliers:
                print(f"\n⚠️  Images with angle offset deviation > 30° from median (uncorrected):")
                for outlier in sorted(offset_outliers, key=lambda x: x['deviation'], reverse=True):
                    print(f"  • {outlier['image']}")
                    print(f"    Gimbal: {outlier['gimbal_yaw']:.1f}° (normalized: {outlier['gimbal_yaw_normalized']:.1f}°)")
                    print(f"    Detected: {outlier['detected_angle']:.1f}° (normalized: {outlier['detected_angle_normalized']:.1f}°)")
                    print(f"    Offset: {outlier['offset']:+.1f}° (deviation: {outlier['deviation']:+.1f}° from median {median_offset:+.1f}°)")
            else:
                print(f"\n✓ All remaining images within 30° of median offset {median_offset:+.1f}°")
        else:
            print("\n⊘ Could not calculate angle offsets")
    
    # Analyze row patterns
    if patterns_collected:
        print("\n" + "=" * 60)
        print("Row Pattern Analysis")
        print("=" * 60)
        
        # Save all patterns to file
        patterns_file = os.path.join(ANNOTATED_DIR, "row_patterns.txt")
        with open(patterns_file, 'w') as f:
            for image_name, pattern in patterns_collected:
                f.write(f"{image_name}: {pattern}\n")
        
        print(f"\n✓ All row patterns saved to: {os.path.basename(patterns_file)}")
        print(f"\nImage patterns ({len(patterns_collected)} images):")
        for image_name, pattern in patterns_collected:
            print(f"  {image_name}: {pattern}")
        
        # Analyze repeating patterns across all sequences
        repeating_patterns = []
        for image_name, full_sequence in patterns_collected:
            pattern, period, confidence = find_repeating_pattern(full_sequence)
            repeating_patterns.append((pattern, period, confidence, image_name))
        
        # Find most common repeating pattern
        from collections import Counter
        pattern_counter = Counter(pattern for pattern, _, _, _ in repeating_patterns)
        most_common_repeating = pattern_counter.most_common(1)[0][0]
        most_common_count = pattern_counter.most_common(1)[0][1]
        
        print(f"\n\nRepeating pattern analysis:")
        print(f"Found {len(pattern_counter)} unique repeating patterns")
        print(f"Most frequent repeating unit: {most_common_repeating} (appears in {most_common_count} image(s))")
        print(f"\nAll repeating patterns:")
        for pattern, count in pattern_counter.most_common():
            print(f"  • {pattern:15} - {count:2} image(s)")
        
        # Estimate detected angle for pattern visualization
        # Use the average detected angle from all images
        if line_stats:
            avg_detected_angle = np.mean([stat['detected_angle'] for stat in line_stats])
        else:
            avg_detected_angle = 0
        
        # Draw pattern visualization with most common repeating pattern
        pattern_viz_path = os.path.join(ANNOTATED_DIR, "pattern_visualization.jpg")
        draw_pattern_visualization(most_common_repeating, avg_detected_angle, num_lines=num_lines, output_path=pattern_viz_path)
        
        # Show pattern details
        print(f"\n✓ Pattern visualization saved")
        print(f"  Pattern: {most_common_repeating}")
        print(f"  Angle: {avg_detected_angle:.1f}°")
        print(f"  Lines: {num_lines}")
    
    return annotation_data, annotated_count

def extract_gimbal_yaw_from_image(image_path):
    """Extract gimbal yaw angle from image XMP metadata.
    
    Returns:
        Gimbal yaw angle (float) or None if not found
    """
    try:
        image = Image.open(image_path)
        
        # Extract XMP metadata
        if 'xmp' not in image.info:
            return None
        
        xmp_data = image.info['xmp']
        try:
            xmp_str = xmp_data.decode('utf-8', errors='ignore')
        except:
            return None
        
        # Look for GimbalYawDegree using regex (faster than XML parsing for single value)
        match = re.search(r'drone-dji:GimbalYawDegree="([^"]+)"', xmp_str)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        
        # Fallback: try XML parsing
        try:
            root = ET.fromstring(xmp_str)
            # Find GimbalYawDegree elements
            for elem in root.iter():
                if 'GimbalYawDegree' in elem.tag:
                    try:
                        return float(elem.text)
                    except:
                        pass
        except:
            pass
    
    except Exception as e:
        pass
    
    return None

def extract_gps_from_image(image_path):
    """Extract GPS coordinates from image EXIF data.
    
    Returns:
        [longitude, latitude] or None if not found
    """
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()
        
        if not exif_data:
            return None
        
        # Get GPS IFD (0x8825 is the tag for GPSInfo)
        try:
            gps_ifd = exif_data.get_ifd(0x8825)
        except:
            return None
        
        if not gps_ifd:
            return None
        
        # Extract latitude and longitude using tag numbers
        # 1: LatitudeRef, 2: Latitude, 3: LongitudeRef, 4: Longitude
        def dms_to_decimal(dms_data):
            """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees."""
            d, m, s = dms_data
            decimal = float(d) + float(m) / 60 + float(s) / 3600
            return decimal
        
        if 2 in gps_ifd and 4 in gps_ifd:
            lat = dms_to_decimal(gps_ifd[2])
            lon = dms_to_decimal(gps_ifd[4])
            
            # Check latitude reference (1: N/S)
            if 1 in gps_ifd and gps_ifd[1] == 'S':
                lat = -lat
            
            # Check longitude reference (3: E/W)
            if 3 in gps_ifd and gps_ifd[3] == 'W':
                lon = -lon
            
            return [lon, lat]  # GeoJSON uses [lon, lat]
    
    except Exception as e:
        print(f"  Warning: Could not extract GPS from {image_path}: {str(e)}")
    
    return None

def create_geojson(annotation_data):
    """Create GeoJSON FeatureCollection from annotation data.
    
    Args:
        annotation_data: List of dicts with 'image_name', 'tassel_count', 'gps'
    
    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []
    
    for data in annotation_data:
        if data['gps'] is None:
            print(f"  ⊘ Skipping {data['image_name']} - no GPS data")
            continue
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": data['gps']
            },
            "properties": {
                "image_name": data['image_name'],
                "tassel_count": data['tassel_count']
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def save_geojson(geojson_data, output_path):
    """Save GeoJSON to file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        print(f"✓ GeoJSON saved: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to save GeoJSON: {str(e)}")
        return False

def download_and_organize_files(draw_lines=False, num_lines=100):
    """Main function to download and organize files.
    
    Args:
        draw_lines: If True, also draw the orientation lines on the images
        num_lines: Number of crop rows to detect per image (default: 100)
    """
    print("=" * 60)
    print("Hetzner Object Storage Downloader & Annotator")
    print("=" * 60)
    
    # Create local directories
    create_directories(clear_existing=not SKIP_DOWNLOAD)
    
    # Always clear annotated folder (regenerate annotations each run)
    clear_directory(ANNOTATED_DIR)
    
    if not SKIP_DOWNLOAD:
        # Connect to S3
        print("\nConnecting to Hetzner Object Storage...")
        s3_client = connect_to_s3()
        
        try:
            # List all files in the base path
            print(f"\nFetching files from: {S3_BASE_PATH}")
            all_files = list_all_files(s3_client, S3_BUCKET_NAME, S3_BASE_PATH)
            
            if not all_files:
                print("No files found in the specified path.")
                return
            
            print(f"Found {len(all_files)} total files")
            
            # Separate and download files by format
            jpg_count = 0
            pkl_count = 0
            failed_count = 0
            
            print("\nDownloading and organizing files...\n")
            
            for s3_key in all_files:
                # Skip if it's a folder (ends with /)
                if s3_key.endswith('/'):
                    continue
                
                file_name = os.path.basename(s3_key)
                
                # Determine file type and destination
                if s3_key.lower().endswith('.jpg') or s3_key.lower().endswith('.jpeg'):
                    local_path = os.path.join(IMAGES_DIR, file_name)
                    if download_file(s3_client, S3_BUCKET_NAME, s3_key, local_path):
                        jpg_count += 1
                    else:
                        failed_count += 1
                        
                elif s3_key.lower().endswith('.pkl'):
                    local_path = os.path.join(PICKLES_DIR, file_name)
                    if download_file(s3_client, S3_BUCKET_NAME, s3_key, local_path):
                        pkl_count += 1
                    else:
                        failed_count += 1
                else:
                    print(f"⊘ Skipped (unknown format): {file_name}")
            
            # Print download summary
            print("\n" + "=" * 60)
            print("Download Summary")
            print("=" * 60)
            print(f"JPG files downloaded: {jpg_count}")
            print(f"PKL files downloaded: {pkl_count}")
            if failed_count > 0:
                print(f"Failed downloads: {failed_count}")
            print(f"Total processed: {jpg_count + pkl_count}")
            print("=" * 60)
        
        except Exception as e:
            print(f"Error during download process: {str(e)}")
    else:
        print("\n⊘ Skipped download - working with existing files only\n")
    
    # Annotate images with bounding boxes
    annotation_data, annotated_count = match_and_annotate_images(draw_lines=draw_lines, num_lines=num_lines)
    
    print("\n" + "=" * 60)
    print("Annotation Summary")
    print("=" * 60)
    print(f"Images annotated: {annotated_count}")
    print(f"Output folder: {ANNOTATED_DIR}")
    print("=" * 60)
    
    # Create and save GeoJSON
    print("\n" + "=" * 60)
    print("Creating GeoJSON")
    print("=" * 60)
    geojson_data = create_geojson(annotation_data)
    geojson_path = os.path.join(os.getcwd(), "detections.geojson")
    if save_geojson(geojson_data, geojson_path):
        print(f"Features in GeoJSON: {len(geojson_data['features'])}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download drone images and annotate with detections')
    parser.add_argument('--draw-lines', action='store_true', help='Draw the orientation lines on the annotated images')
    parser.add_argument('--num-lines', type=int, default=100, help='Number of crop rows to detect per image (default: 100)')
    args = parser.parse_args()
    
    download_and_organize_files(draw_lines=args.draw_lines, num_lines=args.num_lines)
