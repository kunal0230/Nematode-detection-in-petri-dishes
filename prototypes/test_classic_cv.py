#!/usr/bin/env python3
"""
Classic CV Worm Detector
========================
Attempts to detect worms using standard image processing techniques 
(Adaptive Thresholding + Morphology) instead of Deep Learning.

This simulates "Background Subtraction" concepts for static images
by attempting to separate foreground (worms) from background (agar)
based on contrast/intensity.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def detect_classic_cv(image_path):
    print(f"Processing {image_path} with Classic CV...")
    
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Blurring to reduce noise (agar texture)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 4. Adaptive Thresholding
    # This acts like a local background subtraction
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # Invert so worms are white
        blockSize=25, # Area to look at for local threshold
        C=10 # Constant subtracted from mean (sensitivity)
    )
    
    # 5. Morphological Operations (Clean up)
    kernel = np.ones((3,3), np.uint8)
    
    # Open: Remove small noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Close: Connect broken parts of worms
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 6. Find Contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 7. Filter Contours (Size/Shape analysis)
    min_area = 100  # Minimum pixel area to be a worm
    max_area = 5000 # Max area (avoid full dish borders etc)
    
    worm_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            worm_contours.append(cnt)
            
    print(f"Classic CV found {len(worm_contours)} potential candidates.")
    
    # 8. Visualize
    result = img.copy()
    cv2.drawContours(result, worm_contours, -1, (0, 0, 255), 2)
    
    # Plot steps for comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(thresh, cmap='gray')
    axes[1].set_title("Adaptive Threshold")
    axes[1].axis('off')
    
    axes[2].imshow(closing, cmap='gray')
    axes[2].set_title("Morphed (Noise Removed)")
    axes[2].axis('off')
    
    axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Result ({len(worm_contours)} found)")
    axes[3].axis('off')
    
    output_path = f"classic_cv_result_{os.path.basename(image_path)}"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    import os
    if len(sys.argv) > 1:
        detect_classic_cv(sys.argv[1])
    else:
        print("Usage: python3 test_classic_cv.py <image_path>")
