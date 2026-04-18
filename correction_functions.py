"""
Functions for correcting oversized h-sections in crop row detection.
These functions determine whether to shrink h-sections from left or right
based on maximizing bounding boxes in the remaining h-section.
"""

import numpy as np
import cv2
from collections import defaultdict

def get_line_box_counts(boxes, height, width, angle_degrees, num_lines):
    """
    Count how many bounding boxes are closest to each line.
    
    Returns:
        List of box counts per line (length num_lines)
    """
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
    line_thickness_threshold = 40
    
    box_counts = [0] * num_lines
    box_to_closest_line = {}
    
    for box_idx, box in enumerate(boxes):
        if isinstance(box, (list, tuple, np.ndarray)):
            box = np.array(box, dtype=np.float32)
            if len(box) == 4:
                x1, y1, x2, y2 = box
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                
                min_dist = float('inf')
                closest_line = 0
                
                for i in range(num_lines):
                    offset = (i - num_lines / 2.0) * (diag_length / (num_lines - 1 if num_lines > 1 else 1))
                    center_x = width / 2
                    center_y = height / 2
                    perp_offset_x = perp_x * offset
                    perp_offset_y = perp_y * offset
                    line_center_x = center_x + perp_offset_x
                    line_center_y = center_y + perp_offset_y
                    
                    dist_to_line = abs((box_center_x - line_center_x) * perp_x + (box_center_y - line_center_y) * perp_y)
                    
                    if dist_to_line < min_dist:
                        min_dist = dist_to_line
                        closest_line = i
                
                if min_dist < line_thickness_threshold:
                    box_counts[closest_line] += 1
                    box_to_closest_line[box_idx] = closest_line
    
    return box_counts


def analyze_h_section_box_counts(h_sections_info, line_colors, box_counts):
    """
    Get the box count for each h-section.
    
    Args:
        h_sections_info: List of (start_idx, end_idx, length) for each h-section
        line_colors: List of colors for each line
        box_counts: List of box counts per line
    
    Returns:
        Dict mapping h-section index to total box count
    """
    h_section_boxes = {}
    
    for h_idx, (start_idx, end_idx, length) in enumerate(h_sections_info):
        total_boxes = sum(box_counts[start_idx:end_idx+1])
        h_section_boxes[h_idx] = total_boxes
    
    return h_section_boxes


def find_h_sections_with_indices(line_colors):
    """
    Find all h-sections and their start/end indices.
    
    Returns:
        List of (start_idx, end_idx, length) tuples
    """
    h_sections = []
    start_idx = None
    
    for i, color in enumerate(line_colors):
        is_h = (color == (255, 0, 0))  # Blue = high
        
        if is_h and start_idx is None:
            start_idx = i
        elif not is_h and start_idx is not None:
            h_sections.append((start_idx, i - 1, i - start_idx))
            start_idx = None
    
    if start_idx is not None:
        h_sections.append((start_idx, len(line_colors) - 1, len(line_colors) - start_idx))
    
    return h_sections


def find_best_shrink_direction(h_start_idx, h_length, target_length, box_counts):
    """
    Determine whether to shrink h-section from left or right.
    
    Args:
        h_start_idx: Starting index of the h-section
        h_length: Current length of h-section
        target_length: Desired length
        box_counts: List of box counts per line
    
    Returns:
        'left' if shrinking from left is better, 'right' otherwise
    """
    reduction_needed = h_length - target_length
    
    # Option 1: Shrink from left (remove from start)
    left_remaining_start = h_start_idx + reduction_needed
    left_remaining_end = h_start_idx + h_length - 1
    left_boxes = sum(box_counts[left_remaining_start:left_remaining_end+1])
    
    # Option 2: Shrink from right (remove from end)
    right_remaining_start = h_start_idx
    right_remaining_end = h_start_idx + target_length - 1
    right_boxes = sum(box_counts[right_remaining_start:right_remaining_end+1])
    
    if left_boxes >= right_boxes:
        return 'left', left_boxes
    else:
        return 'right', right_boxes


def apply_h_section_correction(line_colors, h_sections_info, box_counts, target_length):
    """
    Apply corrections to oversized h-sections.
    
    Returns:
        Corrected line_colors list
    """
    corrected_colors = line_colors.copy()
    corrections_made = []
    
    for h_idx, (start_idx, end_idx, h_length) in enumerate(h_sections_info):
        if h_length > target_length:
            direction, boxes_result = find_best_shrink_direction(
                start_idx, h_length, target_length, box_counts
            )
            
            reduction = h_length - target_length
            
            if direction == 'left':
                # Change first N lines to red
                for i in range(start_idx, start_idx + reduction):
                    corrected_colors[i] = (0, 0, 255)  # Red = low
                corrections_made.append(f"H-section {h_idx} ({start_idx}-{end_idx}): {h_length}→{target_length} rows (LEFT: changed lines {start_idx}-{start_idx+reduction-1}, boxes={boxes_result})")
            else:  # right
                # Change last N lines to red
                for i in range(end_idx - reduction + 1, end_idx + 1):
                    corrected_colors[i] = (0, 0, 255)  # Red = low
                corrections_made.append(f"H-section {h_idx} ({start_idx}-{end_idx}): {h_length}→{target_length} rows (RIGHT: changed lines {end_idx-reduction+1}-{end_idx}, boxes={boxes_result})")
    
    return corrected_colors, corrections_made
