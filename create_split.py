"""
Create train/validation split for dataset
Split dataset into 2 parts:
- Train: 90%
- Validation: 10%
"""

import os
import random
from pathlib import Path

def create_train_val_split(data_path, train_ratio=0.9, seed=42):
    """
    Create train.txt and val.txt files containing image list
    
    Args:
        data_path: path to dataset directory
        train_ratio: training ratio (0.9 = 90%)
        seed: random seed for reproducibility
    """
    random.seed(seed)
    
    images_dir = os.path.join(data_path, 'images')
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    print(f"Total images: {len(image_files)}")
    
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)} images ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Validation: {len(val_files)} images ({len(val_files)/len(image_files)*100:.1f}%)")
    
    train_path = os.path.join(data_path, 'train.txt')
    with open(train_path, 'w') as f:
        for fname in sorted(train_files):
            f.write(f"{fname}\n")
    print(f"\nCreated: {train_path}")
    
    val_path = os.path.join(data_path, 'val.txt')
    with open(val_path, 'w') as f:
        for fname in sorted(val_files):
            f.write(f"{fname}\n")
    print(f"Created: {val_path}")
    
    all_path = os.path.join(data_path, 'all.txt')
    with open(all_path, 'w') as f:
        for fname in sorted(image_files):
            f.write(f"{fname}\n")
    print(f"Created: {all_path}")

def main():
    data_path = r"c:\code\LINGMI-MR\data\blender_colon"
    
    if not os.path.exists(data_path):
        print(f"Error: Directory not found {data_path}")
        print("Please run prepare_dataset.py first!")
        return
    
    print("=" * 60)
    print("CREATE TRAIN/VALIDATION SPLIT")
    print("=" * 60)
    print()
    
    create_train_val_split(data_path, train_ratio=0.9, seed=42)
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
