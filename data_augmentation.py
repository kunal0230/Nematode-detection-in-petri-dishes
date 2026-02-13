#!/usr/bin/env python3
"""
Data Augmentation Pipeline for Worm Detection
==============================================
Generates augmented training data from limited samples
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import random
from scipy import ndimage


class WormDataAugmentor:
    """
    Advanced data augmentation for worm detection with:
    - Geometric transformations
    - Color/brightness adjustments
    - Noise injection
    - Elastic deformations
    - Background variations
    """
    
    def __init__(self, output_dir='augmented_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        (self.output_dir / 'coco_annotations').mkdir(exist_ok=True)
    
    def load_coco_annotations(self, json_path):
        """Load COCO format annotations."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def load_yolo_annotations(self, txt_path):
        """Load YOLO format annotations."""
        boxes = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, cx, cy, w, h = map(float, parts)
                    boxes.append({
                        'class_id': int(class_id),
                        'center_x': cx,
                        'center_y': cy,
                        'width': w,
                        'height': h
                    })
        return boxes
    
    def yolo_to_bbox(self, yolo_box, img_width, img_height):
        """Convert YOLO format to pixel coordinates."""
        cx = yolo_box['center_x'] * img_width
        cy = yolo_box['center_y'] * img_height
        w = yolo_box['width'] * img_width
        h = yolo_box['height'] * img_height
        
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        return x1, y1, x2, y2
    
    def bbox_to_yolo(self, bbox, img_width, img_height):
        """Convert pixel bbox to YOLO format."""
        x1, y1, x2, y2 = bbox
        
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        
        return cx, cy, w, h
    
    # ============================================================================
    # GEOMETRIC AUGMENTATIONS
    # ============================================================================
    
    def rotate(self, image, boxes, angle):
        """Rotate image and adjust bounding boxes."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(128, 128, 128))
        
        # Rotate boxes
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = self.yolo_to_bbox(box, w, h)
            
            # Get corners
            corners = np.array([
                [x1, y1, 1], [x2, y1, 1],
                [x2, y2, 1], [x1, y2, 1]
            ]).T
            
            # Transform corners
            transformed = M @ corners
            
            # Get new bbox
            xs = transformed[0, :]
            ys = transformed[1, :]
            new_x1, new_x2 = int(min(xs)), int(max(xs))
            new_y1, new_y2 = int(min(ys)), int(max(ys))
            
            # Convert back to YOLO
            cx, cy, bw, bh = self.bbox_to_yolo([new_x1, new_y1, new_x2, new_y2], 
                                                new_w, new_h)
            
            new_boxes.append({
                'class_id': box['class_id'],
                'center_x': cx,
                'center_y': cy,
                'width': bw,
                'height': bh
            })
        
        return rotated, new_boxes
    
    def flip_horizontal(self, image, boxes):
        """Horizontal flip."""
        flipped = cv2.flip(image, 1)
        
        new_boxes = []
        for box in boxes:
            new_boxes.append({
                'class_id': box['class_id'],
                'center_x': 1.0 - box['center_x'],
                'center_y': box['center_y'],
                'width': box['width'],
                'height': box['height']
            })
        
        return flipped, new_boxes
    
    def flip_vertical(self, image, boxes):
        """Vertical flip."""
        flipped = cv2.flip(image, 0)
        
        new_boxes = []
        for box in boxes:
            new_boxes.append({
                'class_id': box['class_id'],
                'center_x': box['center_x'],
                'center_y': 1.0 - box['center_y'],
                'width': box['width'],
                'height': box['height']
            })
        
        return flipped, new_boxes
    
    def random_crop(self, image, boxes, crop_ratio=0.8):
        """Random crop while maintaining boxes."""
        h, w = image.shape[:2]
        
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        cropped = image[top:top+new_h, left:left+new_w]
        
        # Adjust boxes
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = self.yolo_to_bbox(box, w, h)
            
            # Check if box is in cropped region
            if x2 > left and x1 < left + new_w and y2 > top and y1 < top + new_h:
                # Clip to crop boundaries
                new_x1 = max(0, x1 - left)
                new_y1 = max(0, y1 - top)
                new_x2 = min(new_w, x2 - left)
                new_y2 = min(new_h, y2 - top)
                
                # Only keep if significant overlap
                if (new_x2 - new_x1) > 10 and (new_y2 - new_y1) > 10:
                    cx, cy, bw, bh = self.bbox_to_yolo([new_x1, new_y1, new_x2, new_y2],
                                                        new_w, new_h)
                    new_boxes.append({
                        'class_id': box['class_id'],
                        'center_x': cx,
                        'center_y': cy,
                        'width': bw,
                        'height': bh
                    })
        
        return cropped, new_boxes
    
    def zoom(self, image, boxes, zoom_factor=1.2):
        """Zoom in/out."""
        h, w = image.shape[:2]
        
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)
        
        # Calculate crop region (centered)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        cropped = image[top:top+new_h, left:left+new_w]
        zoomed = cv2.resize(cropped, (w, h))
        
        # Adjust boxes
        new_boxes = []
        for box in boxes:
            # Convert to pixel coords in original
            x1, y1, x2, y2 = self.yolo_to_bbox(box, w, h)
            
            # Adjust for crop
            x1_crop = x1 - left
            y1_crop = y1 - top
            x2_crop = x2 - left
            y2_crop = y2 - top
            
            # Check if in crop region
            if x1_crop < new_w and x2_crop > 0 and y1_crop < new_h and y2_crop > 0:
                # Clip to boundaries
                x1_crop = max(0, x1_crop)
                y1_crop = max(0, y1_crop)
                x2_crop = min(new_w, x2_crop)
                y2_crop = min(new_h, y2_crop)
                
                # Scale back to original size
                x1_scaled = int(x1_crop * zoom_factor)
                y1_scaled = int(y1_crop * zoom_factor)
                x2_scaled = int(x2_crop * zoom_factor)
                y2_scaled = int(y2_crop * zoom_factor)
                
                cx, cy, bw, bh = self.bbox_to_yolo([x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                                                    w, h)
                
                new_boxes.append({
                    'class_id': box['class_id'],
                    'center_x': cx,
                    'center_y': cy,
                    'width': bw,
                    'height': bh
                })
        
        return zoomed, new_boxes
    
    # ============================================================================
    # COLOR/BRIGHTNESS AUGMENTATIONS
    # ============================================================================
    
    def adjust_brightness(self, image, factor):
        """Adjust brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """Adjust contrast."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        l = np.clip((l - 128) * factor + 128, 0, 255)
        lab[:, :, 0] = l
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def adjust_hue(self, image, shift):
        """Adjust hue."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_saturation(self, image, factor):
        """Adjust saturation."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # ============================================================================
    # NOISE AND BLUR
    # ============================================================================
    
    def add_gaussian_noise(self, image, std=15):
        """Add Gaussian noise."""
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def add_salt_pepper_noise(self, image, amount=0.01):
        """Add salt and pepper noise."""
        noisy = image.copy()
        
        # Salt
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255
        
        # Pepper
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0
        
        return noisy
    
    def gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def motion_blur(self, image, kernel_size=15):
        """Apply motion blur."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
    
    # ============================================================================
    # ELASTIC DEFORMATION
    # ============================================================================
    
    def elastic_transform(self, image, alpha=30, sigma=5):
        """Apply elastic deformation."""
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), 
                                     sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), 
                                     sigma, mode="constant", cval=0) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
        
        if len(image.shape) == 3:
            distorted = np.zeros_like(image)
            for i in range(image.shape[2]):
                distorted[:, :, i] = ndimage.map_coordinates(
                    image[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            distorted = ndimage.map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
        
        return distorted
    
    # ============================================================================
    # AUGMENTATION PIPELINE
    # ============================================================================
    
    def augment_image(self, image, boxes, augmentation_config):
        """
        Apply a series of augmentations based on config.
        
        augmentation_config: dict with boolean flags for each augmentation
        """
        aug_image = image.copy()
        aug_boxes = [b.copy() for b in boxes]
        
        # Geometric transformations
        if augmentation_config.get('rotate'):
            angle = random.uniform(-30, 30)
            aug_image, aug_boxes = self.rotate(aug_image, aug_boxes, angle)
        
        if augmentation_config.get('flip_h') and random.random() > 0.5:
            aug_image, aug_boxes = self.flip_horizontal(aug_image, aug_boxes)
        
        if augmentation_config.get('flip_v') and random.random() > 0.5:
            aug_image, aug_boxes = self.flip_vertical(aug_image, aug_boxes)
        
        if augmentation_config.get('crop') and random.random() > 0.5:
            aug_image, aug_boxes = self.random_crop(aug_image, aug_boxes, 
                                                     random.uniform(0.7, 0.95))
        
        if augmentation_config.get('zoom') and random.random() > 0.5:
            aug_image, aug_boxes = self.zoom(aug_image, aug_boxes, 
                                              random.uniform(0.9, 1.3))
        
        # Color augmentations
        if augmentation_config.get('brightness'):
            aug_image = self.adjust_brightness(aug_image, random.uniform(0.7, 1.3))
        
        if augmentation_config.get('contrast'):
            aug_image = self.adjust_contrast(aug_image, random.uniform(0.8, 1.2))
        
        if augmentation_config.get('hue'):
            aug_image = self.adjust_hue(aug_image, random.uniform(-10, 10))
        
        if augmentation_config.get('saturation'):
            aug_image = self.adjust_saturation(aug_image, random.uniform(0.8, 1.2))
        
        # Noise and blur
        if augmentation_config.get('gaussian_noise') and random.random() > 0.7:
            aug_image = self.add_gaussian_noise(aug_image, random.uniform(5, 20))
        
        if augmentation_config.get('blur') and random.random() > 0.7:
            aug_image = self.gaussian_blur(aug_image, random.choice([3, 5, 7]))
        
        if augmentation_config.get('elastic') and random.random() > 0.8:
            aug_image = self.elastic_transform(aug_image, 
                                               alpha=random.uniform(20, 40),
                                               sigma=random.uniform(4, 7))
        
        return aug_image, aug_boxes
    
    def generate_augmentations(self, image_path, label_path, num_augmentations=10,
                               format='yolo'):
        """
        Generate multiple augmented versions of a single image.
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        label_path : str
            Path to label file (YOLO txt or COCO json)
        num_augmentations : int
            Number of augmented versions to generate
        format : str
            'yolo' or 'coco'
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Load annotations
        if format == 'yolo':
            boxes = self.load_yolo_annotations(label_path)
        else:
            coco_data = self.load_coco_annotations(label_path)
            # Convert COCO to YOLO format for processing
            # (implementation depends on your COCO structure)
            boxes = []
        
        base_name = Path(image_path).stem
        
        # Define augmentation strategies (progressive difficulty)
        augmentation_strategies = [
            # Light augmentations
            {'rotate': True, 'brightness': True},
            {'flip_h': True, 'contrast': True},
            {'flip_v': True, 'saturation': True},
            
            # Medium augmentations
            {'rotate': True, 'flip_h': True, 'brightness': True, 'contrast': True},
            {'crop': True, 'hue': True, 'saturation': True},
            {'zoom': True, 'brightness': True, 'gaussian_noise': True},
            
            # Heavy augmentations
            {'rotate': True, 'crop': True, 'brightness': True, 'contrast': True, 
             'hue': True, 'gaussian_noise': True},
            {'flip_h': True, 'zoom': True, 'saturation': True, 'blur': True},
            {'rotate': True, 'flip_v': True, 'crop': True, 'brightness': True, 
             'elastic': True},
            {'rotate': True, 'flip_h': True, 'zoom': True, 'contrast': True, 
             'hue': True, 'blur': True, 'gaussian_noise': True}
        ]
        
        print(f"\nGenerating {num_augmentations} augmentations for {base_name}...")
        
        for i in range(num_augmentations):
            # Select augmentation strategy
            strategy = augmentation_strategies[i % len(augmentation_strategies)]
            
            # Apply augmentations
            aug_image, aug_boxes = self.augment_image(image, boxes, strategy)
            
            # Skip if no boxes remain
            if len(aug_boxes) == 0:
                print(f"  Skipping aug_{i}: no boxes remaining after augmentation")
                continue
            
            # Save augmented image
            aug_image_path = self.output_dir / 'images' / f'{base_name}_aug_{i}.jpg'
            cv2.imwrite(str(aug_image_path), aug_image)
            
            # Save augmented labels (YOLO format)
            aug_label_path = self.output_dir / 'labels' / f'{base_name}_aug_{i}.txt'
            with open(aug_label_path, 'w') as f:
                for box in aug_boxes:
                    f.write(f"{box['class_id']} {box['center_x']:.6f} "
                           f"{box['center_y']:.6f} {box['width']:.6f} "
                           f"{box['height']:.6f}\n")
            
            print(f"  Generated aug_{i} with {len(aug_boxes)} worms")
        
        print(f"Augmentation complete for {base_name}!")


def main():
    """Example usage."""
    print("="*80)
    print("WORM DETECTION DATA AUGMENTATION PIPELINE")
    print("="*80)
    print()
    
    augmentor = WormDataAugmentor(output_dir='augmented_worm_data')
    
    # Example: augment a single image
    # augmentor.generate_augmentations(
    #     image_path='path/to/image.jpg',
    #     label_path='path/to/labels.txt',
    #     num_augmentations=20,
    #     format='yolo'
    # )
    
    print("Data augmentation pipeline ready!")
    print()
    print("Usage:")
    print("  augmentor.generate_augmentations(")
    print("      image_path='your_image.jpg',")
    print("      label_path='your_labels.txt',")
    print("      num_augmentations=20,")
    print("      format='yolo'")
    print("  )")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
