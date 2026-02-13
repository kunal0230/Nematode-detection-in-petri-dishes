#!/usr/bin/env python3
"""
Complete Worm Detection Workflow Example
=========================================
Demonstrates the full pipeline from annotation to inference
"""

import os
import sys
from pathlib import Path

print("="*80)
print("WORM DETECTION ML PIPELINE - COMPLETE WORKFLOW")
print("="*80)
print()

# ============================================================================
# PHASE 1: ANNOTATION
# ============================================================================

print("PHASE 1: DATA ANNOTATION")
print("-"*80)
print()
print("1. Open worm_annotation_tool.html in your web browser")
print("2. Load your petri dish images")
print("3. Draw bounding boxes around each worm")
print("4. Export annotations as YOLO format (.txt)")
print()
print("Organize your data as:")
print("  raw_data/")
print("    ├── images/")
print("    │   ├── dish_001.jpg")
print("    │   ├── dish_002.jpg")
print("    │   └── ...")
print("    └── labels/")
print("        ├── dish_001.txt")
print("        ├── dish_002.txt")
print("        └── ...")
print()
input("Press Enter after you've annotated your images...")
print()

# ============================================================================
# PHASE 2: DATA AUGMENTATION
# ============================================================================

print("\nPHASE 2: DATA AUGMENTATION")
print("-"*80)
print()

try:
    from data_augmentation import WormDataAugmentor
    
    # Initialize augmentor
    augmentor = WormDataAugmentor(output_dir='augmented_worm_data')
    
    # Check if raw data exists
    raw_image_dir = Path('raw_data/images')
    raw_label_dir = Path('raw_data/labels')
    
    if not raw_image_dir.exists():
        print("Warning: raw_data/images directory not found")
        print("Creating example directory structure...")
        raw_image_dir.mkdir(parents=True, exist_ok=True)
        raw_label_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("Please place your:")
        print("  - Images in: raw_data/images/")
        print("  - Labels in: raw_data/labels/")
        print()
        print("Then run this script again.")
        sys.exit(1)
    
    # Get all images
    image_files = list(raw_image_dir.glob('*.jpg')) + list(raw_image_dir.glob('*.png'))
    
    if len(image_files) == 0:
        print("No images found in raw_data/images/")
        print("Please add your annotated images and run again.")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to augment")
    print()
    
    # Augment each image
    total_generated = 0
    for img_file in image_files:
        label_file = raw_label_dir / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            print(f"Skipping {img_file.name} - no label file found")
            continue
        
        print(f"Processing {img_file.name}...")
        
        augmentor.generate_augmentations(
            image_path=str(img_file),
            label_path=str(label_file),
            num_augmentations=20,
            format='yolo'
        )
        
        total_generated += 20
    
    print()
    print(f"Data augmentation complete!")
    print(f"Generated {total_generated} augmented samples")
    print(f"Output directory: augmented_worm_data/")
    print()

except Exception as e:
    print(f"Error during augmentation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

input("Press Enter to continue to training phase...")
print()

# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================

print("\nPHASE 3: MODEL TRAINING")
print("-"*80)
print()

print("Training requires PyTorch. Checking installation...")

try:
    import torch
    import torchvision
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchVision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    from ml_model_training import WormDetectionTrainer, WormDetectionDataset
    from torch.utils.data import DataLoader, random_split
    
    # Configuration
    config = {
        'image_dir': 'augmented_worm_data/images',
        'label_dir': 'augmented_worm_data/labels',
        'model_type': 'fasterrcnn',
        'num_classes': 2,
        'batch_size': 2,  # Small batch size for limited GPU memory
        'num_epochs': 30,  # Reduced for demo
        'learning_rate': 0.001,
        'train_split': 0.8
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = WormDetectionDataset(
        config['image_dir'],
        config['label_dir']
    )
    
    if len(full_dataset) < 10:
        print(f"Warning: Only {len(full_dataset)} samples found")
        print("Consider generating more augmentations for better results")
    
    # Split train/val
    train_size = int(config['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Initialize trainer
    print("Initializing model...")
    trainer = WormDetectionTrainer(
        model_type=config['model_type'],
        num_classes=config['num_classes']
    )
    print()
    
    # Train
    print("Starting training...")
    print("(This may take a while depending on your hardware)")
    print()
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate']
    )
    
    print()
    print("Training complete!")
    print(f"Best model saved to: best_worm_detector.pth")
    print()

except ImportError:
    print("PyTorch not installed")
    print()
    print("To install PyTorch:")
    print("  pip install torch torchvision --break-system-packages")
    print()
    print("OR use the CPU version:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("Once installed, run this script again to train the model.")
    print()
    sys.exit(1)

except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PHASE 4: INFERENCE
# ============================================================================

print("\nPHASE 4: INFERENCE & TESTING")
print("-"*80)
print()

try:
    # Load the best model
    print("Loading best model...")
    trainer.load_model('best_worm_detector.pth')
    print()
    
    # Test on validation images
    print("Running inference on test images...")
    
    test_images = list(Path(config['image_dir']).glob('*.jpg'))[:5]
    
    for img_path in test_images:
        print(f"\nTesting on: {img_path.name}")
        
        # Predict
        predictions = trainer.predict(str(img_path), confidence_threshold=0.5)
        
        print(f"  Detected {len(predictions['boxes'])} worms")
        
        for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
            print(f"    Worm {i+1}: Confidence={score:.2f}")
        
        # Visualize
        output_path = f"results_{img_path.stem}.png"
        trainer.visualize_predictions(
            str(img_path),
            save_path=output_path,
            confidence_threshold=0.5
        )
        print(f"  Saved visualization to: {output_path}")
    
    print()
    print("Inference complete!")
    print()

except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WORKFLOW COMPLETE!")
print("="*80)
print()
print("Summary:")
print("  Data annotated and organized")
print(f"  Generated {total_generated} augmented samples")
print("  Model trained and saved")
print("  Inference demonstrated")
print()
print("Next Steps:")
print("  1. Use the trained model on new images")
print("  2. Fine-tune with more data if needed")
print("  3. Adjust confidence threshold for your use case")
print("  4. Consider active learning for continuous improvement")
print()
print("Files Generated:")
print("  - augmented_worm_data/         (augmented dataset)")
print("  - best_worm_detector.pth       (trained model)")
print("  - training_history.png         (training curves)")
print("  - results_*.png                (inference visualizations)")
print()
print("="*80)
