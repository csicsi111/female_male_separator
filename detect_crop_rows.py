#!/usr/bin/env python3
"""
Crop Row Angle Detection Script
Detects crop row orientation angle from an image.

Usage:
    python detect_crop_rows.py <input_image>
    
Example:
    python detect_crop_rows.py field.jpg

Output: Prints the detected crop row angle in degrees (0-180)

Resolution: Automatically resizes to 512 height maintaining aspect ratio
Parameters (matching optimized defaults):
    Hough Threshold: 60
    Angle Error: 5°
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys
from PIL import Image

# Optimized parameters (matching user-tuned demo settings)
HOUGH_THRESHOLD = 60
ANGLE_ERROR = 5
CLUSTERING_TOLERANCE = 28
TARGET_HEIGHT = 512  # Resize to 512 height, maintaining aspect ratio


def get_vegetation_mask_rgb(image_path, target_size=None):
    """
    Detect vegetation in RGB image using Excess Green Index.
    
    Parameters:
    - image_path: Path to image file
    - target_size: Resize to this height, maintaining aspect ratio (default: 512)
    """
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use default target size if not provided
    if target_size is None:
        target_size = TARGET_HEIGHT
    
    # Resize maintaining aspect ratio
    h, w = image_rgb.shape[:2]
    if target_size > 0 and target_size != h:
        aspect_ratio = w / h
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Get dimensions
    h, w = image_rgb.shape[:2]
    
    # Extract channels
    b = image_rgb[:, :, 2].astype(float)  # B
    g = image_rgb[:, :, 1].astype(float)  # G
    r = image_rgb[:, :, 0].astype(float)  # R
    
    # Excess Green Index: 2*G - R - B
    excess_green = 2 * g - r - b
    
    # Normalize to 0-255
    excess_green_norm = ((excess_green - excess_green.min()) / 
                         (excess_green.max() - excess_green.min() + 1e-6) * 255).astype(np.uint8)
    
    # Threshold
    _, vegetation_mask = cv2.threshold(excess_green_norm, 80, 255, cv2.THRESH_BINARY)
    
    # Light morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Additional smoothing to reduce texture noise
    kernel_larger = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel_larger, iterations=1)
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel_larger, iterations=1)
    
    # Gaussian blur to smooth edges
    vegetation_mask = cv2.GaussianBlur(vegetation_mask, (5, 5), 0)
    
    return vegetation_mask, image_rgb


def detect_row_orientation(vegetation_mask, hough_threshold=90, angle_tolerance=20, verbose=False):
    """
    Detect the dominant crop row orientation using Hough line detection.
    Uses the EXACT same logic as the Streamlit demo, including angle filtering.
    Returns the dominant angle, aligned lines list, and count.
    
    Parameters:
    - vegetation_mask: Binary mask of vegetation
    - hough_threshold: Threshold for Hough line detection
    - angle_tolerance: Tolerance for angle filtering (degrees)
    - verbose: If True, print debug information
    """
    # Edge detection
    edges = cv2.Canny(vegetation_mask, 50, 150)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=50,
        maxLineGap=15
    )
    
    if lines is None or len(lines) == 0:
        print("Warning: No lines detected")
        return 90, [], 0  # Default to vertical, empty list, 0 aligned lines
    
    # Calculate angles and line info - SAME AS STREAMLIT
    angles = []
    line_infos = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle and normalize to 0-180
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angles.append(angle)
        line_infos.append({'angle': angle, 'length': length, 'line': line[0]})
    
    # Find dominant angle using binning - SAME AS STREAMLIT
    angle_bins = {}
    for angle in angles:
        bin_center = round(angle / 15) * 15  # Bin to nearest 15 degrees
        angle_bins[bin_center] = angle_bins.get(bin_center, 0) + 1
    
    dominant_angle = max(angle_bins.items(), key=lambda x: x[1])[0]
    
    # Filter lines to keep only those aligned with dominant angle - SAME AS STREAMLIT!
    parallel_lines = []
    for line_info in line_infos:
        angle_diff = abs(line_info['angle'] - dominant_angle)
        # Handle wrap-around at 180°
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        if angle_diff < angle_tolerance and line_info['length'] > 40:
            parallel_lines.append(line_info)
    
    if verbose:
        print(f"📐 Detected row orientation: {dominant_angle}° (from {len(parallel_lines)} aligned lines out of {len(lines)} total)")
    
    return dominant_angle, parallel_lines, len(parallel_lines)


def detect_crop_row_angle(input_path, verbose=True):
    """
    Detect crop row orientation angle from image.
    
    Parameters:
    - input_path: Path to input image
    - verbose: If True, print progress messages; if False, run silently
    
    Returns:
    - angle: Detected crop row angle in degrees (0-180)
    """
    if verbose:
        print(f"\n🌾 Processing: {input_path}")
        print(f"Parameters: Hough={HOUGH_THRESHOLD}, Angle Error={ANGLE_ERROR}")
    
    # Load image and detect vegetation
    if verbose:
        print("🔍 Detecting vegetation...")
    vegetation_mask, image = get_vegetation_mask_rgb(input_path, target_size=TARGET_HEIGHT)
    if verbose:
        print(f"📐 Image resized to: {image.shape[1]}×{image.shape[0]} (maintaining aspect ratio from input)")
    
    # Detect orientation
    if verbose:
        print("📐 Analyzing row orientation...")
    orientation, parallel_lines, aligned_count = detect_row_orientation(
        vegetation_mask, 
        hough_threshold=HOUGH_THRESHOLD,
        angle_tolerance=ANGLE_ERROR,
        verbose=verbose
    )
    
    # Calculate average angle of detected parallel lines
    if verbose:
        print(f"📊 Calculating average angle from {len(parallel_lines)} detected lines...")
    if parallel_lines:
        detected_angles = [line_info['angle'] for line_info in parallel_lines]
        avg_detected_angle = np.mean(detected_angles)
        if verbose:
            print(f"   Average: {avg_detected_angle:.1f}°")
    else:
        avg_detected_angle = orientation
        if verbose:
            print(f"   No parallel lines detected, using dominant angle: {avg_detected_angle:.1f}°")
    
    # Add 90 degrees to the result
    final_angle = (avg_detected_angle + 90) % 180
    
    if verbose:
        print(f"✓ Using crop row angle: {final_angle:.1f}° (original: {avg_detected_angle:.1f}° + 90°)")
    
    return final_angle


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_crop_rows.py <input_image>")
        print("\nExample: python detect_crop_rows.py field.jpg")
        print("\nOutput: Detected crop row angle in degrees (0-180)")
        sys.exit(1)
    
    input_image = sys.argv[1]
    
    # Check if input exists
    if not Path(input_image).exists():
        print(f"Error: Input image not found: {input_image}")
        sys.exit(1)
    
    try:
        angle = detect_crop_row_angle(input_image)
        print(f"\n✅ Detected crop row angle: {angle:.1f}°")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
