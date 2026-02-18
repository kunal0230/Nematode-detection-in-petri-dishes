#!/usr/bin/env python3
"""
Prepare YOLOv8 Dataset for Kaggle
==================================
Creates a properly structured YOLOv8 dataset from raw frames + labels.
Performs slicing and augmentation, then packages everything into a zip.
"""

import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path

def load_yolo_labels(label_path):
    """Load YOLO-format labels from a file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append({
                    'class': int(parts[0]),
                    'cx': float(parts[1]),
                    'cy': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
    return boxes

def slice_image(img, boxes, patch_size, stride, img_w, img_h):
    """Slice image into overlapping patches with corresponding labels."""
    patches = []
    
    x_steps = list(range(0, img_w - patch_size + 1, stride))
    if len(x_steps) == 0 or x_steps[-1] + patch_size < img_w:
        x_steps.append(max(0, img_w - patch_size))
    
    y_steps = list(range(0, img_h - patch_size + 1, stride))
    if len(y_steps) == 0 or y_steps[-1] + patch_size < img_h:
        y_steps.append(max(0, img_h - patch_size))
    
    for y in y_steps:
        for x in x_steps:
            # Extract patch
            patch = img[y:y+patch_size, x:x+patch_size]
            
            # Find boxes that overlap with this patch
            patch_boxes = []
            for box in boxes:
                # Convert normalized to pixel coords
                bx = box['cx'] * img_w
                by = box['cy'] * img_h
                bw = box['w'] * img_w
                bh = box['h'] * img_h
                
                # Box edges
                bx1 = bx - bw/2
                by1 = by - bh/2
                bx2 = bx + bw/2
                by2 = by + bh/2
                
                # Patch edges
                px1 = x
                py1 = y
                px2 = x + patch_size
                py2 = y + patch_size
                
                # Check overlap
                ix1 = max(bx1, px1)
                iy1 = max(by1, py1)
                ix2 = min(bx2, px2)
                iy2 = min(by2, py2)
                
                if ix1 < ix2 and iy1 < iy2:
                    # Calculate overlap ratio
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    box_area = bw * bh
                    overlap_ratio = inter_area / box_area if box_area > 0 else 0
                    
                    # Only include if >50% of the box is in the patch
                    if overlap_ratio > 0.5:
                        # Clip to patch and convert to patch-relative normalized
                        new_cx = ((ix1 + ix2) / 2 - x) / patch_size
                        new_cy = ((iy1 + iy2) / 2 - y) / patch_size
                        new_w = (ix2 - ix1) / patch_size
                        new_h = (iy2 - iy1) / patch_size
                        
                        patch_boxes.append({
                            'class': box['class'],
                            'cx': new_cx,
                            'cy': new_cy,
                            'w': new_w,
                            'h': new_h
                        })
            
            patches.append({
                'image': patch,
                'boxes': patch_boxes
            })
    
    return patches

def augment_patch(img, boxes, aug_idx):
    """Apply augmentation to a patch."""
    h, w = img.shape[:2]
    aug_img = img.copy()
    aug_boxes = list(boxes)
    
    if aug_idx == 0:
        # Horizontal flip
        aug_img = cv2.flip(aug_img, 1)
        aug_boxes = [{'class': b['class'], 'cx': 1.0 - b['cx'], 'cy': b['cy'], 'w': b['w'], 'h': b['h']} for b in boxes]
    elif aug_idx == 1:
        # Vertical flip
        aug_img = cv2.flip(aug_img, 0)
        aug_boxes = [{'class': b['class'], 'cx': b['cx'], 'cy': 1.0 - b['cy'], 'w': b['w'], 'h': b['h']} for b in boxes]
    elif aug_idx == 2:
        # Brightness jitter
        factor = random.uniform(0.7, 1.3)
        aug_img = np.clip(aug_img * factor, 0, 255).astype(np.uint8)
    elif aug_idx == 3:
        # Gaussian blur
        aug_img = cv2.GaussianBlur(aug_img, (5, 5), 0)
    elif aug_idx == 4:
        # Rotate 90
        aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
        aug_boxes = [{'class': b['class'], 'cx': 1.0 - b['cy'], 'cy': b['cx'], 'w': b['h'], 'h': b['w']} for b in boxes]
    elif aug_idx == 5:
        # Contrast adjustment
        factor = random.uniform(0.8, 1.4)
        mean = np.mean(aug_img)
        aug_img = np.clip((aug_img - mean) * factor + mean, 0, 255).astype(np.uint8)
    elif aug_idx == 6:
        # Add noise
        noise = np.random.normal(0, 10, aug_img.shape).astype(np.int16)
        aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif aug_idx == 7:
        # Both flips
        aug_img = cv2.flip(aug_img, -1)
        aug_boxes = [{'class': b['class'], 'cx': 1.0 - b['cx'], 'cy': 1.0 - b['cy'], 'w': b['w'], 'h': b['h']} for b in boxes]
    
    return aug_img, aug_boxes

def save_yolo_label(boxes, path):
    """Save boxes to YOLO format."""
    with open(path, 'w') as f:
        for b in boxes:
            f.write(f"{b['class']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

def main():
    RAW_IMAGES = 'raw_data/images'
    RAW_LABELS = 'raw_data/labels'
    OUTPUT_DIR = 'yolov8_dataset'
    PATCH_SIZE = 416
    STRIDE = 208  # 50% overlap
    NUM_AUGMENTATIONS = 8
    TRAIN_SPLIT = 0.85
    
    # Clean output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Create YOLOv8 directory structure
    for split in ['train', 'val']:
        os.makedirs(f'{OUTPUT_DIR}/{split}/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/{split}/labels', exist_ok=True)
    
    # Process each frame
    all_patches = []
    image_files = sorted([f for f in os.listdir(RAW_IMAGES) if f.endswith('.png')])
    
    print(f"Processing {len(image_files)} frames...")
    
    for img_file in image_files:
        base_name = img_file.replace('.png', '')
        label_file = base_name + '.txt'
        
        img_path = os.path.join(RAW_IMAGES, img_file)
        label_path = os.path.join(RAW_LABELS, label_file)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping {img_file} (cannot read)")
            continue
        
        h, w = img.shape[:2]
        boxes = load_yolo_labels(label_path)
        
        print(f"  {img_file}: {w}x{h}, {len(boxes)} annotations")
        
        # Slice into patches
        patches = slice_image(img, boxes, PATCH_SIZE, STRIDE, w, h)
        
        # Keep patches that have at least one annotation
        annotated_patches = [p for p in patches if len(p['boxes']) > 0]
        
        # Also keep some empty patches (negative examples) — 20%
        empty_patches = [p for p in patches if len(p['boxes']) == 0]
        n_neg = max(1, len(annotated_patches) // 5)
        if len(empty_patches) > n_neg:
            random.shuffle(empty_patches)
            empty_patches = empty_patches[:n_neg]
        
        print(f"    Patches: {len(annotated_patches)} with worms, {len(empty_patches)} negative")
        
        for patch in annotated_patches:
            all_patches.append((base_name, patch, True))
            
            # Augment annotated patches
            for aug_idx in range(NUM_AUGMENTATIONS):
                aug_img, aug_boxes = augment_patch(patch['image'], patch['boxes'], aug_idx)
                all_patches.append((f"{base_name}_aug{aug_idx}", {'image': aug_img, 'boxes': aug_boxes}, True))
        
        for patch in empty_patches:
            all_patches.append((base_name, patch, False))
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_patches)
    
    split_idx = int(len(all_patches) * TRAIN_SPLIT)
    train_patches = all_patches[:split_idx]
    val_patches = all_patches[split_idx:]
    
    print(f"\nTotal patches: {len(all_patches)}")
    print(f"  Train: {len(train_patches)}")
    print(f"  Val: {len(val_patches)}")
    
    # Save patches
    for split_name, patches in [('train', train_patches), ('val', val_patches)]:
        for idx, (name, patch, _) in enumerate(patches):
            img_path = f'{OUTPUT_DIR}/{split_name}/images/{name}_{idx:04d}.jpg'
            lbl_path = f'{OUTPUT_DIR}/{split_name}/labels/{name}_{idx:04d}.txt'
            
            cv2.imwrite(img_path, patch['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_yolo_label(patch['boxes'], lbl_path)
    
    # Create data.yaml
    data_yaml = f"""# Worm Detection Dataset
path: /kaggle/working/yolov8_dataset
train: train/images
val: val/images

nc: 1
names: ['worm']
"""
    with open(f'{OUTPUT_DIR}/data.yaml', 'w') as f:
        f.write(data_yaml)
    
    print(f"\nDataset saved to {OUTPUT_DIR}/")
    print(f"  data.yaml created")
    
    # Create zip
    print("Creating zip file...")
    shutil.make_archive('worm_dataset_yolov8', 'zip', '.', OUTPUT_DIR)
    zip_size = os.path.getsize('worm_dataset_yolov8.zip') / (1024 * 1024)
    print(f"Dataset zip: worm_dataset_yolov8.zip ({zip_size:.1f} MB)")

if __name__ == '__main__':
    main()
