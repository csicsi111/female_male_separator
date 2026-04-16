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

# S3 Object Storage Configuration (Hetzner Object Storage)
S3_BUCKET_NAME = "groglanceprod"
S3_REGION = "eu-central"
S3_ACCESS_KEY = "BLSXRZVK1A3DPNL7KFDV"
S3_SECRET_KEY = "3MpPIcuuUdgO9I2OmNIueCJghCVkODy561sysBAY"
S3_ENDPOINT = "https://nbg1.your-objectstorage.com"

# Path to the folder containing subfolders
S3_BASE_PATH = "user@demo.com/20m_overlap_test/images/"

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

def extract_boxes_from_sahi_result(sahi_result, debug=False, confidence_threshold=0.5):
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

def kmedians_clustering(data, k=2):
    """K-medians clustering using percentiles (more robust than random initialization).
    
    Args:
        data: List or array of values to cluster
        k: Number of clusters (currently optimized for k=2)
    
    Returns:
        List of cluster centers (medians)
    """
    if len(data) == 0:
        return [0, 0]
    
    data = np.array(data, dtype=float)
    
    # For k=2, use tertiles: find median of lower third and median of upper third
    # This is more robust than trying to initialize randomly
    if k == 2:
        sorted_data = np.sort(data)
        # Low cluster median from lower 40% of data
        low_third = sorted_data[:len(sorted_data)//2]
        # High cluster median from upper 40% of data
        high_third = sorted_data[len(sorted_data)//2:]
        
        low_center = np.median(low_third)
        high_center = np.median(high_third)
        
        return sorted([low_center, high_center])
    
    # Fallback for other k values
    sorted_data = np.sort(data)
    percentiles = [i * 100 / k for i in range(1, k)]
    centers = [np.median(data)] + [np.percentile(data, p) for p in percentiles]
    return sorted(centers)
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
    
    Args:
        image: OpenCV image
        boxes: List of bounding boxes for intersection counting
        angle_degrees: Angle in degrees
        line_color: BGR color tuple (default if no clusters)
        num_lines: Number of lines to draw
        thickness: Line thickness
        cluster_centers: List of 2 cluster centers [low, high] for color assignment
    
    Returns:
        List of intersection counts for this image
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
        
        # Count intersections with boxes
        intersections = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            if line_intersects_box(start_x, start_y, end_x, end_y, int(x1), int(y1), int(x2), int(y2)):
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
    for i, intersections in enumerate(intersection_counts):
        # Determine initial color based on clusters
        if cluster_centers is not None:
            low_center, high_center = cluster_centers
            dist_to_low = abs(intersections - low_center)
            dist_to_high = abs(intersections - high_center)
            
            if dist_to_low <= dist_to_high:
                draw_color = (255, 0, 255)  # Purple/Magenta for low cluster
            else:
                draw_color = (255, 0, 0)    # Blue for high cluster
        else:
            draw_color = line_color
        
        line_colors.append(draw_color)
    
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
    
    return intersection_counts

def get_line_colors_for_boxes(image, boxes, angle_degrees, cluster_centers=None, num_lines=100):
    """
    Determine the cluster color (high/low) for each box based on its closest row line.
    Each box is assigned to the closest line, then lines are colored based on box count.
    Returns a list of (box_idx, color) tuples.
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
    
    # Calculate all line positions
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
    
    # Step 1: Assign each box to its closest line
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
    
    # Step 2: Count boxes assigned to each line
    line_box_counts = [0] * num_lines
    for closest_line_idx in box_assignments.values():
        line_box_counts[closest_line_idx] += 1
    
    # Step 2.5: Smooth box counts to eliminate isolated low-count lines
    # Use median-based smoothing to remove outlier counts
    smoothed_counts = line_box_counts.copy()
    for i in range(1, num_lines - 1):
        # Use median of 3 consecutive lines
        left_count = line_box_counts[i - 1]
        center_count = line_box_counts[i]
        right_count = line_box_counts[i + 1]
        
        # If center is isolated (different from both neighbors), use median of the three
        if center_count != left_count and center_count != right_count:
            median_val = np.median([left_count, center_count, right_count])
            smoothed_counts[i] = int(median_val)
    
    line_box_counts = smoothed_counts
    
    # Step 3: Color each line based on box count
    line_colors_map = []
    for count in line_box_counts:
        # Determine line color based on cluster
        if cluster_centers is not None:
            low_center, high_center = cluster_centers
            dist_to_low = abs(count - low_center)
            dist_to_high = abs(count - high_center)
            
            if dist_to_low <= dist_to_high:
                line_color = (0, 0, 255)    # Red for low
            else:
                line_color = (255, 0, 0)   # Blue for high
        else:
            line_color = (0, 0, 255)       # Default red
        
        line_colors_map.append(line_color)
    
    # Step 4: Smooth line colors
    smoothed_colors = line_colors_map.copy()
    for i in range(1, num_lines - 1):
        left_color = smoothed_colors[i - 1]
        right_color = smoothed_colors[i + 1]
        current_color = smoothed_colors[i]
        
        if left_color == right_color and current_color != left_color:
            smoothed_colors[i] = left_color
    
    # Step 5: Assign colors to boxes based on their assigned line's color
    box_colors = []
    high_count = 0
    low_count = 0
    
    for box_idx in range(len(boxes)):
        closest_line_idx = box_assignments[box_idx]
        color = smoothed_colors[closest_line_idx]
        box_colors.append((box_idx, color))
        
        if color == (255, 0, 0):  # Blue = high
            high_count += 1
        else:  # Red = low
            low_count += 1
    
    return box_colors, high_count, low_count

def assign_box_colors_from_rows(image, boxes, angle_degrees, cluster_centers=None, num_lines=100):
    """
    Assign colors to boxes based on their closest row's color.
    Uses intersection-based coloring (same as lines).
    Returns list of (box_idx, color) tuples and counts.
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
        
        # Line endpoints for intersection checking
        start_x = int(line_center_x - max_extent * dx)
        start_y = int(line_center_y - max_extent * dy)
        end_x = int(line_center_x + max_extent * dx)
        end_y = int(line_center_y + max_extent * dy)
        
        line_positions.append((start_x, start_y, end_x, end_y, line_center_x, line_center_y))
    
    # Calculate intersection counts for each line
    intersection_counts = []
    for i in range(num_lines):
        start_x, start_y, end_x, end_y, line_center_x, line_center_y = line_positions[i]
        
        intersections = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            if line_intersects_box(start_x, start_y, end_x, end_y, int(x1), int(y1), int(x2), int(y2)):
                intersections += 1
        
        intersection_counts.append(intersections)
    
    # Smooth intersection counts to eliminate isolated outliers
    smoothed_counts = intersection_counts.copy()
    for i in range(1, num_lines - 1):
        left_count = intersection_counts[i - 1]
        center_count = intersection_counts[i]
        right_count = intersection_counts[i + 1]
        
        if center_count != left_count and center_count != right_count:
            median_val = np.median([left_count, center_count, right_count])
            smoothed_counts[i] = int(median_val)
    
    # Assign colors to lines based on smoothed counts
    line_colors = []
    for count in smoothed_counts:
        if cluster_centers is not None:
            low_center, high_center = cluster_centers
            dist_to_low = abs(count - low_center)
            dist_to_high = abs(count - high_center)
            
            if dist_to_low <= dist_to_high:
                line_color = (0, 0, 255)    # Red for low
            else:
                line_color = (255, 0, 0)   # Blue for high
        else:
            line_color = (255, 0, 0)       # Default blue
        
        line_colors.append(line_color)
    
    # Use line colors directly (no island checking)
    smoothed_colors = line_colors
    
    # Assign each box to its closest row and get that row's color
    box_colors = []
    high_count = 0
    low_count = 0
    
    line_center_positions = [(line_pos[4], line_pos[5]) for line_pos in line_positions]
    
    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Find closest line to this box
        min_dist = float('inf')
        closest_line_idx = 0
        
        for line_idx, (line_x, line_y) in enumerate(line_center_positions):
            # Perpendicular distance to line
            dist = abs((box_center_x - line_x) * perp_x + (box_center_y - line_y) * perp_y)
            if dist < min_dist:
                min_dist = dist
                closest_line_idx = line_idx
        
        # Get the color of the closest line
        color = smoothed_colors[closest_line_idx]
        box_colors.append((box_idx, color))
        
        if color == (255, 0, 0):  # Blue = high
            high_count += 1
        else:  # Red = low
            low_count += 1
    
    return box_colors, high_count, low_count

def draw_bounding_boxes(image_path, detections, output_path, color=(0, 0, 255), thickness=3, debug_first=False, cluster_centers=None, collect_only=False, draw_lines=False, override_angle=None, num_lines=100):
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
        boxes = None
        
        # Check if it's a SAHI result
        if isinstance(detections, list) or hasattr(detections, 'object_prediction_list') or (isinstance(detections, dict) and 'object_prediction_list' in detections):
            boxes = extract_boxes_from_sahi_result(detections, debug=debug_first)
        elif isinstance(detections, dict):
            # Try common keys
            for key in ['boxes', 'detections', 'bounding_boxes', 'bbox']:
                if key in detections:
                    boxes = detections[key]
                    break
        
        if not boxes:
            if debug_first:
                print(f"⊘ No detections found in {image_path}")
            return False if not collect_only else []
        
        # If collecting only, just get counts without drawing/saving
        if collect_only:
            try:
                base_angle = detect_crop_row_angle(image_path, verbose=False)
                # Create a temporary copy for counting (don't save)
                temp_image = image.copy()
                counts = draw_orientation_lines_with_counts(temp_image, boxes, base_angle, num_lines=num_lines, thickness=3, cluster_centers=None)
                return counts  # Return counts without saving
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
        if draw_lines:
            draw_orientation_lines_with_counts(image, boxes, base_angle, num_lines=num_lines, thickness=2, cluster_centers=cluster_centers)
        
        # Get box colors based on closest row
        box_colors, high_count, low_count = assign_box_colors_from_rows(image, boxes, base_angle, cluster_centers=cluster_centers, num_lines=num_lines)
        
        drawn_count = 0
        
        # Draw each box with its assigned color based on closest row
        for box_idx, box in enumerate(boxes):
            try:
                if isinstance(box, (list, tuple, np.ndarray)):
                    box = np.array(box, dtype=np.float32)
                    
                    # Handle different box formats
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        
                        # Convert to int for drawing
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        # Only draw if box has positive area
                        if x2 > x1 and y2 > y1:
                            # Get color from box_colors list
                            box_color = (0, 0, 255)  # Default red
                            for b_idx, b_color in box_colors:
                                if b_idx == box_idx:
                                    box_color = b_color
                                    break
                            
                            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
                            drawn_count += 1
                    elif len(box) == 2:
                        # Assuming [[x1, y1], [x2, y2]] format
                        pt1 = tuple(box[0].astype(int))
                        pt2 = tuple(box[1].astype(int))
                        
                        # Get color from box_colors list
                        box_color = (0, 0, 255)  # Default red
                        for b_idx, b_color in box_colors:
                            if b_idx == box_idx:
                                box_color = b_color
                                break
                        
                        cv2.rectangle(image, pt1, pt2, box_color, thickness)
                        drawn_count += 1
                        high_count += 1
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
    cluster_centers = kmedians_clustering(all_intersection_counts, k=2)
    print(f"K-medians cluster centers: Low={cluster_centers[0]:.1f}, High={cluster_centers[1]:.1f}\n")
    
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
    
    # Second pass: draw all images with cluster-based coloring and corrected angles where needed
    annotated_count = 0
    annotation_data = []
    line_stats = []  # Track (image_name, high_count, low_count) for each image
    
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
        
        # Load detections and draw with colors
        detections = load_pickle_detections(pkl_path)
        if detections is not None:
            # Use corrected angle if available (for outlier images), otherwise use detected angle
            if matched_image in corrected_angles:
                angle_to_use = corrected_angles[matched_image]
                is_corrected = True
            else:
                angle_to_use = angle_data[matched_image]['detected_angle']
                is_corrected = False
            
            # Get gimbal yaw for tracking
            gimbal_yaw = angle_data[matched_image]['gimbal_yaw']
            
            output_path = os.path.join(ANNOTATED_DIR, base_name + ".jpg")
            if is_corrected:
                print(f"  [CORRECTING] {base_name}: Using corrected angle {angle_to_use:.1f}°")
            result = draw_bounding_boxes(matched_image, detections, output_path, cluster_centers=tuple(cluster_centers), draw_lines=draw_lines, override_angle=angle_to_use, num_lines=num_lines)
            
            if result and isinstance(result, tuple) and len(result) == 2:
                high_count, low_count = result
                annotated_count += 1
                
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
