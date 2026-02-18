#!/usr/bin/env python3
import torch
import cv2
import matplotlib.pyplot as plt
import sys
import os
from train_optimized import get_mobilenet_model
from torchvision import transforms

def predict_custom(image_path, model_path='worm_detector_mobilenet_m1.pth', threshold=0.5):
    print(f"Processing {image_path}...")
    
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
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

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
        
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.2f}', color='black', fontsize=10, weight='bold', bbox=dict(facecolor='lime', alpha=0.5))
        
    plt.axis('off')
    plt.title(f'Detections on {os.path.basename(image_path)} (Thresh={threshold})')
    
    output_filename = f"prediction_{os.path.basename(image_path)}"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Result saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_custom(image_path)
    else:
        print("Usage: python3 predict_custom.py <image_path>")
