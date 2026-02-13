#!/usr/bin/env python3
"""
Custom ML Model for Worm Detection
===================================
Transfer learning + Custom architecture
Supports: YOLOv5, Faster R-CNN, or custom CNN

NOTE: This requires PyTorch. Install with:
pip install torch torchvision --break-system-packages

Or use this as a template for your ML framework of choice.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATASET CLASS
# ============================================================================

class WormDetectionDataset(Dataset):
    """Custom dataset for worm detection."""
    
    def __init__(self, image_dir, label_dir, transforms=None, format='yolo'):
        """
        Parameters:
        -----------
        image_dir : str
            Directory containing images
        label_dir : str
            Directory containing labels
        transforms : callable
            Optional transforms to apply
        format : str
            'yolo' or 'coco'
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.format = format
        
        # Get all image files
        self.images = sorted(list(self.image_dir.glob('*.jpg')) + 
                           list(self.image_dir.glob('*.png')))
        
        print(f"Found {len(self.images)} images in dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # Convert YOLO to pixel coordinates
                        img_h, img_w = image.shape[:2]
                        x1 = int((cx - w/2) * img_w)
                        y1 = int((cy - h/2) * img_h)
                        x2 = int((cx + w/2) * img_w)
                        y2 = int((cy + h/2) * img_h)
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id) + 1)  # +1 because 0 is background
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transforms:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, target


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def get_fasterrcnn_model(num_classes):
    """
    Get Faster R-CNN model with ResNet-50 backbone.
    Pre-trained on COCO, fine-tuned for worm detection.
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


class CustomWormCNN(nn.Module):
    """
    Custom CNN for worm detection.
    Lighter weight alternative to Faster R-CNN.
    """
    
    def __init__(self, num_classes=2):
        super(CustomWormCNN, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Detection head (simplified)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # Output heads
        self.bbox_head = nn.Linear(1024, 4)  # x, y, w, h
        self.class_head = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        features = self.features(x)
        features = self.classifier(features)
        
        bbox = self.bbox_head(features)
        classification = self.class_head(features)
        
        return bbox, classification


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class WormDetectionTrainer:
    """Training pipeline for worm detection models."""
    
    def __init__(self, model_type='fasterrcnn', num_classes=2, device=None):
        """
        Parameters:
        -----------
        model_type : str
            'fasterrcnn' or 'custom'
        num_classes : int
            Number of classes (including background)
        device : str
            'cuda' or 'cpu'
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Initialize model
        if model_type == 'fasterrcnn':
            self.model = get_fasterrcnn_model(num_classes)
        else:
            self.model = CustomWormCNN(num_classes)
        
        self.model.to(self.device)
        
        print(f"Model: {model_type}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {num_classes}")
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        """
        Train the model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int
            Number of training epochs
        lr : float
            Learning rate
        """
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=lr)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_loss': float('inf'),
            'best_epoch': 0
        }
        
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc='Training')
            for images, targets in train_pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
                train_pbar.set_postfix({'loss': losses.item()})
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc='Validation')
                for images, targets in val_pbar:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    val_loss += losses.item()
                    val_pbar.set_postfix({'loss': losses.item()})
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < history['best_loss']:
                history['best_loss'] = val_loss
                history['best_epoch'] = epoch
                self.save_model('best_worm_detector.pth')
                print(f"New best model saved (val_loss: {val_loss:.4f})")
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best epoch: {history['best_epoch']+1}")
        print(f"Best validation loss: {history['best_loss']:.4f}")
        print("="*80)
        
        return history
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
    
    def predict(self, image, confidence_threshold=0.5):
        """
        Make predictions on a single image.
        
        Parameters:
        -----------
        image : numpy array or str
            Image array or path to image
        confidence_threshold : float
            Minimum confidence for detections
        
        Returns:
        --------
        dict : Predictions with boxes, labels, and scores
        """
        self.model.eval()
        
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Filter by confidence
        pred = predictions[0]
        keep = pred['scores'] > confidence_threshold
        
        boxes = pred['boxes'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }
    
    def visualize_predictions(self, image_path, save_path=None, confidence_threshold=0.5):
        """Visualize predictions on an image."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        predictions = self.predict(image, confidence_threshold)
        
        # Draw predictions
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        for box, label, score in zip(predictions['boxes'], 
                                      predictions['labels'],
                                      predictions['scores']):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor='lime', linewidth=2)
            ax.add_patch(rect)
            
            ax.text(x1, y1-5, f'Worm {score:.2f}',
                   bbox=dict(facecolor='lime', alpha=0.7),
                   fontsize=10, color='black')
        
        ax.axis('off')
        ax.set_title(f'Detected {len(predictions["boxes"])} worms', 
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
        
        return predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline."""
    print("="*80)
    print("WORM DETECTION ML TRAINING PIPELINE")
    print("="*80)
    print()
    
    # Configuration
    config = {
        'image_dir': 'augmented_worm_data/images',
        'label_dir': 'augmented_worm_data/labels',
        'model_type': 'fasterrcnn',  # 'fasterrcnn' or 'custom'
        'num_classes': 2,  # background + worm
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'train_split': 0.8,
        'confidence_threshold': 0.5
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataset
    full_dataset = WormDetectionDataset(
        config['image_dir'],
        config['label_dir']
    )
    
    # Split train/val
    train_size = int(config['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Initialize trainer
    trainer = WormDetectionTrainer(
        model_type=config['model_type'],
        num_classes=config['num_classes']
    )
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate']
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history saved to training_history.png")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("="*80)
        print("ERROR: PyTorch not installed")
        print("="*80)
        print()
        print("This script requires PyTorch. Install with:")
        print("  pip install torch torchvision --break-system-packages")
        print()
        print("Or use this as a template for TensorFlow/Keras:")
        print("  - Replace PyTorch Dataset with tf.data.Dataset")
        print("  - Replace model with TensorFlow implementation")
        print("  - Replace training loop with model.fit()")
        print()
        print("="*80)
