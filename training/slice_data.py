#!/usr/bin/env python3
"""
Data Slicing Script
===================
Slices large images into smaller overlapping patches for training.
Crucial for single-image datasets to create enough training samples.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def load_yolo_labels(label_path):
    """Load YOLO labels from file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # class_id, cx, cy, w, h
                boxes.append(list(map(float, parts)))
    return boxes

def save_yolo_labels(boxes, save_path):
    """Save YOLO labels to file."""
    with open(save_path, 'w') as f:
        for box in boxes:
            # class_id, cx, cy, w, h
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def slice_image(image_path, label_path, output_dir, patch_size=416, stride=208, min_area_ratio=0.3):
    """
    Slice image and labels into patches.
    
    Parameters:
    -----------
    min_area_ratio : float
        Minimum overlap of a bounding box to be included in the patch (0-1).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading {image_path}")
        return

    img_h, img_w = img.shape[:2]
    all_boxes = load_yolo_labels(label_path)
    
    # Convert normalized YOLO to pixel coords [x1, y1, x2, y2]
    pixel_boxes = []
    for cls, cx, cy, w, h in all_boxes:
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        pixel_boxes.append([cls, x1, y1, x2, y2])

    output_images_dir = Path(output_dir) / 'images'
    output_labels_dir = Path(output_dir) / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem
    patch_count = 0

    # Sliding window
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # Ensure patch doesn't go out of bounds (optional: could pad)
            # For simplicity, we just clip or skip. 
            # Better strategy for training: if remaining width < patch_size, shift back
            
            x_start = x
            y_start = y
            
            if x_start + patch_size > img_w:
                x_start = max(0, img_w - patch_size)
            if y_start + patch_size > img_h:
                y_start = max(0, img_h - patch_size)
            
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Extract patch
            patch = img[y_start:y_end, x_start:x_end]
            
            # Find boxes in this patch
            patch_boxes = []
            for cls, bx1, by1, bx2, by2 in pixel_boxes:
                # Calculate intersection
                ix1 = max(bx1, x_start)
                iy1 = max(by1, y_start)
                ix2 = min(bx2, x_end)
                iy2 = min(by2, y_end)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                
                if iw > 0 and ih > 0:
                    # Check if enough of the object is in the patch
                    box_area = (bx2 - bx1) * (by2 - by1)
                    inter_area = iw * ih
                    
                    if inter_area / box_area >= min_area_ratio:
                        # Convert to patch-relative coordinates
                        new_x1 = max(0, bx1 - x_start)
                        new_y1 = max(0, by1 - y_start)
                        new_x2 = min(patch_size, bx2 - x_start)
                        new_y2 = min(patch_size, by2 - y_start)
                        
                        # Convert back to YOLO format for the patch
                        p_w = patch_size
                        p_h = patch_size
                        
                        ncx = ((new_x1 + new_x2) / 2) / p_w
                        ncy = ((new_y1 + new_y2) / 2) / p_h
                        nw = (new_x2 - new_x1) / p_w
                        nh = (new_y2 - new_y1) / p_h
                        
                        # Clip to be safe
                        ncx = min(max(ncx, 0), 1)
                        ncy = min(max(ncy, 0), 1)
                        nw = min(max(nw, 0), 1)
                        nh = min(max(nh, 0), 1)
                        
                        patch_boxes.append([cls, ncx, ncy, nw, nh])

            # Save if patch has labels or (optionally) save empty backgrounds
            # For this task (single image), we probably want mostly positive samples
            if len(patch_boxes) > 0:
                patch_name = f"{base_name}_{y_start}_{x_start}"
                
                cv2.imwrite(str(output_images_dir / f"{patch_name}.jpg"), patch)
                save_yolo_labels(patch_boxes, output_labels_dir / f"{patch_name}.txt")
                patch_count += 1
                
    print(f"Created {patch_count} patches from {image_path}")

def main():
    # Detect raw data
    raw_dir = Path('raw_data') / 'images'
    # Fallback or specific paths
    if not raw_dir.exists():
        # Check current dir structure
        if Path('worm_annotation_tool.html').exists():
           # Assume we might be in valid root, check specifically for user provided files
           # User said "raw_data/labels/dish_001.txt" is active, so I assume raw_data/images exists or image is somewhere
           pass 
            
    # For now, let's assume standard structure or create it
    # We will search for ANY images in raw_data matching label names
    
    label_dir = Path('raw_data') / 'labels'
    image_dir = Path('raw_data') / 'images' # Try standard
    
    if not label_dir.exists():
        print("Could not find raw_data/labels. Please ensure structure.")
        return

    output_dir = 'processed_data_sliced'
    
    # Process all matched pairs
    images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    if not images:
        # Fallback: check if image is next to labels or elsewhere
        # User has dish_001.txt open.
        pass

    if len(images) == 0:
        print("No images found in raw_data/images. Please place your source image there.")
        # Create directory to help user
        image_dir.mkdir(parents=True, exist_ok=True)
        return

    print(f"Found {len(images)} images. Starting slicing...")
    
    for img_path in images:
        # Find corresponding label
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            slice_image(img_path, label_path, output_dir)
        else:
            print(f"Warning: No label found for {img_path.name}")

if __name__ == "__main__":
    main()
