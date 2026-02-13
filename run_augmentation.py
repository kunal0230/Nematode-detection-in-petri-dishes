#!/usr/bin/env python3
"""
Run Augmentation
================
Applies augmentation to all images in the input directory.
"""

import glob
import os
from pathlib import Path
from data_augmentation import WormDataAugmentor

def main():
    input_dir = Path('processed_data_sliced')
    output_dir = Path('augmented_data_final')
    
    # Clean/Create output
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    augmentor = WormDataAugmentor(output_dir=str(output_dir))
    
    images = list(input_dir.glob('images/*.jpg')) + list(input_dir.glob('images/*.png'))
    
    print(f"Found {len(images)} patches to augment.")
    
    for img_path in images:
        label_path = input_dir / 'labels' / f"{img_path.stem}.txt"
        
        if label_path.exists():
            # Generate 20 versions per patch -> ~500 images total
            augmentor.generate_augmentations(
                image_path=str(img_path),
                label_path=str(label_path),
                num_augmentations=20,
                format='yolo'
            )
            
    print(f"Augmentation complete. Data in {output_dir}")

if __name__ == "__main__":
    main()
