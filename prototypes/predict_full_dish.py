#!/usr/bin/env python3
"""
Full Dish Inference (Sliding Window)
====================================
Runs inference on large images by slicing them into overlapping patches,
detecting worms in each patch, and stitching the results back together
using Non-Maximum Suppression (NMS).
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import transforms
from train_optimized import get_mobilenet_model

def apply_nms(boxes, scores, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    """
    if len(boxes) == 0:
        return [], []
        
    # Convert to tensor for torchvision NMS
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)
    
    return boxes[keep_indices], scores[keep_indices]

def predict_full_dish(image_path, model_path='worm_detector_mobilenet_m1.pth', 
                     patch_size=416, stride=208, conf_threshold=0.5):
    
    # 1. Setup
    print(f"Processing full dish: {image_path}")
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Load Model
    print(f"Loading model from {model_path}...")
    model = get_mobilenet_model(num_classes=2)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Image
    full_image = cv2.imread(image_path)
    if full_image is None:
        print(f"Error loading {image_path}")
        return
        
    img_h, img_w = full_image.shape[:2]
    print(f"Image Size: {img_w}x{img_h}")
    
    # 2. Sliding Window Inference
    all_boxes = []
    all_scores = []
    
    transform = transforms.ToTensor()
    
    # Calculate grid
    x_steps = list(range(0, img_w - patch_size + 1, stride))
    if x_steps[-1] + patch_size < img_w:
        x_steps.append(img_w - patch_size) # Add last patch
        
    y_steps = list(range(0, img_h - patch_size + 1, stride))
    if y_steps[-1] + patch_size < img_h:
        y_steps.append(img_h - patch_size) # Add last patch
        
    total_patches = len(x_steps) * len(y_steps)
    print(f"Scanning {total_patches} patches...")
    
    patch_count = 0
    
    for y in y_steps:
        for x in x_steps:
            patch = full_image[y:y+patch_size, x:x+patch_size]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            patch_tensor = transform(patch_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(patch_tensor)[0]
                
            # Filter by confidence
            p_boxes = predictions['boxes'].cpu().numpy()
            p_scores = predictions['scores'].cpu().numpy()
            
            keep = p_scores >= conf_threshold
            p_boxes = p_boxes[keep]
            p_scores = p_scores[keep]
            
            # Offset boxes to global coordinates
            for box in p_boxes:
                # box is [x1, y1, x2, y2] relative to patch
                global_box = [
                    box[0] + x,
                    box[1] + y,
                    box[2] + x,
                    box[3] + y
                ]
                all_boxes.append(global_box)
                
            all_scores.extend(p_scores)
            
            patch_count += 1
            print(f"\rProgress: {patch_count}/{total_patches} | Found {len(p_boxes)} worms in patch", end="")
            
    print("\nInference complete. Stitching results...")
    
    # 3. Apply NMS (Remove duplicates)
    if len(all_boxes) > 0:
        final_boxes, final_scores = apply_nms(np.array(all_boxes), np.array(all_scores), iou_threshold=0.3)
        print(f"Found {len(final_boxes)} unique worms after NMS.")
    else:
        final_boxes = []
        final_scores = []
        print("No worms detected.")
        
    # 4. Visualize
    vis_img = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(vis_img)
    ax = plt.gca()
    
    for box, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-10, f'{score:.2f}', color='black', fontsize=8, weight='bold', 
                bbox=dict(facecolor='lime', alpha=0.7, pad=1))
        
    plt.axis('off')
    plt.title(f'Full Dish Detection: {len(final_boxes)} Worms (Thresh={conf_threshold})')
    
    output_filename = os.path.join(output_dir, f"full_dish_prediction_{os.path.basename(image_path)}")
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Result saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_full_dish(image_path)
    else:
        print("Usage: python3 predict_full_dish.py <image_path>")
