#!/usr/bin/env python3
"""
Test Worm Detector
==================
Run inference on a single image and visualize results.
"""

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from train_optimized import get_mobilenet_model
from torchvision import transforms

# Enable MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def predict_image(image_path, model_path, threshold=0.5):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Load model
    print(f"Loading model from {model_path}...")
    num_classes = 2
    model = get_mobilenet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    transform = transforms.ToTensor()
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Filter results
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    print(f"Found {len(boxes)} worms (confidence >= {threshold})")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='cyan', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.2f}', color='cyan', fontsize=10, weight='bold')
        
    plt.axis('off')
    plt.title(f'Detections (Thresh={threshold})')
    output_path = 'prediction_result.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    # Test on one of the augmented images (since we don't have a separate test set)
    # We'll pick one that wasn't likely in the very last batch, or just random.
    import glob
    test_images = glob.glob('augmented_data_final/images/*.jpg')
    
    if test_images:
        test_img = test_images[0]
        predict_image(test_img, 'worm_detector_mobilenet_m1.pth')
    else:
        print("No images found to test.")
