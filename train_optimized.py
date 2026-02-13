#!/usr/bin/env python3
"""
Optimized Training Script for M1 Mac
====================================
Uses MobileNetV3-SSDLite for efficient training on Apple Silicon.
Targeted for 8GB RAM devices.
"""

import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection import _utils as det_utils
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Import dataset from existing script
from ml_model_training import WormDetectionDataset
import os

# Enable MPS fallback for features not yet implemented (like hardsigmoid)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

def get_mobilenet_model(num_classes):
    """
    Get SSDLite MobileNetV3 Large model.
    Using SSDLite because it's much faster and lighter than Faster R-CNN.
    """
    # Load pre-trained model (weights=DEFAULT is best practice now, equivalent to pretrained=True)
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    
    # Modify the head for our number of classes
    # SSDLite has multiple heads. The standard way is to rebuild the head.
    # However, for simplicity using torchvision's helper if possible or just manual replacement.
    
    # We need to recreate the predictor
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # Recreate head
    model.head = SSDHead(in_channels, num_anchors, num_classes)
    
    return model

def main():
    print("="*80)
    print("OPTIMIZED M1 TRAINING (MobileNetV3 SSDLite)")
    print("="*80)

    # config
    config = {
        'image_dir': 'augmented_data_final/images',
        'label_dir': 'augmented_data_final/labels',
        'batch_size': 8, # SSDLite is small, we might fit 8. If fails, drop to 4.
        'num_epochs': 20,
        'lr': 0.0005, # SSD usually needs lower LR or warmup
        'num_classes': 2, # background + worm
    }
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("! Using CPU (Slow)")

    # Data
    print("\nLoading Dataset...")
    full_dataset = WormDetectionDataset(config['image_dir'], config['label_dir'])
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0 # MPS sometimes has issues with multiprocessing
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # Model
    print("\nInitializing Model...")
    model = get_mobilenet_model(config['num_classes'])
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=0.0005)
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    
    history = {'train_loss': [], 'val_loss': []}
    
    try:
        for epoch in range(config['num_epochs']):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
                pbar.set_postfix({'loss': f"{losses.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation (optional optimization: skip every other epoch if slow)
            # Evaluating SSD loss is tricky because model() in eval mode returns detections, not loss.
            # We must keep model in train mode but with no_grad to get loss? 
            # Actually, standard torchvision models return predictions in eval mode.
            # To get validation loss, we typically keep 'train' mode but don't optimize.
            
            # Note: For efficient validation metric (mAP) we'd use coco_eval, 
            # but for simple loss tracking we can just do a pass in 'train' mode without backward().
            
            # Validation pass
            with torch.no_grad():
                val_loss_accum = 0.0
                # Keep training mode behavior for loss calculation
                # (FasterRCNN/SSD return loss only in train() mode)
                model.train() 
                
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    val_loss_accum += sum(loss for loss in loss_dict.values()).item()
                
                avg_val_loss = val_loss_accum / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                print(f"  Val Loss: {avg_val_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")

    # Save model
    torch.save(model.state_dict(), 'worm_detector_mobilenet_m1.pth')
    print("\nModel saved to 'worm_detector_mobilenet_m1.pth'")
    
    # Plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig('loss_curve_m1.png')
    print("Loss curve saved")

if __name__ == "__main__":
    main()
