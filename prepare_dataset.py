"""
Dataset preparation script for cd2rtzm23r-1/UnityCam/Colon
Convert to format compatible with BlenderDataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_depth_png_to_npy(depth_png_path, output_npy_path, focal_length=20.0):
    """
    Convert depth PNG to depth NPY
    
    Args:
        depth_png_path: path to PNG depth file
        output_npy_path: output NPY file path
        focal_length: camera focal length (mm)
    """
    depth_img = cv2.imread(depth_png_path, cv2.IMREAD_UNCHANGED)
    
    if depth_img is None:
        print(f"Cannot read file: {depth_png_path}")
        return False
    
    if len(depth_img.shape) == 3:
        depth_img = depth_img[:, :, 0]
    
    depth_float = depth_img.astype(np.float32)
    
    if depth_float.max() < 100:
        depth_float = depth_float / 255.0
        near = 0.01 * focal_length
        far = 100 * focal_length
        depth_float = near + depth_float * (far - near)
    
    # Lưu depth dưới dạng numpy array
    np.save(output_npy_path, depth_float)
    return True

def copy_and_rename_images(src_folder, dst_folder):
    """
    Copy and rename images from image_XXXX.png to XXXX.png
    
    Args:
        src_folder: source image folder
        dst_folder: destination folder
    """
    image_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
    
    print(f"Copying {len(image_files)} images from {src_folder}...")
    
    for img_file in tqdm(image_files):
        if img_file.startswith('image_'):
            num = img_file.replace('image_', '').replace('.png', '')
            new_name = f"{num}.png"
        else:
            new_name = img_file
        
        src_path = os.path.join(src_folder, img_file)
        dst_path = os.path.join(dst_folder, new_name)
        
        shutil.copy2(src_path, dst_path)

def convert_depth_images(src_folder, dst_folder, focal_length=20.0):
    """
    Convert all depth PNG to NPY
    
    Args:
        src_folder: source depth PNG folder
        dst_folder: destination folder
        focal_length: camera focal length
    """
    depth_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
    
    print(f"Converting {len(depth_files)} depth files from {src_folder}...")
    
    for depth_file in tqdm(depth_files):
        if depth_file.startswith('aov_image_'):
            num = depth_file.replace('aov_image_', '').replace('.png', '')
            new_name = f"{num.zfill(8)}_depth.npy"
        else:
            num = depth_file.replace('.png', '')
            new_name = f"{num.zfill(8)}_depth.npy"
        
        src_path = os.path.join(src_folder, depth_file)
        dst_path = os.path.join(dst_folder, new_name)
        
        convert_depth_png_to_npy(src_path, dst_path, focal_length)

def main():
    base_path = r"c:\code\LINGMI-MR"
    source_frames = os.path.join(base_path, "cd2rtzm23r-1", "UnityCam", "Colon", "Frames", "Frames")
    source_depths = os.path.join(base_path, "cd2rtzm23r-1", "UnityCam", "Colon", "Depths", "Pixelwise Depths")
    
    dest_images = os.path.join(base_path, "data", "blender_colon", "images")
    dest_depths = os.path.join(base_path, "data", "blender_colon", "depth")
    
    if not os.path.exists(source_frames):
        print(f"Error: Directory not found {source_frames}")
        print("Please extract Frames.zip first!")
        return
    
    if not os.path.exists(source_depths):
        print(f"Error: Directory not found {source_depths}")
        print("Please extract Pixelwise Depths.zip first!")
        return
    
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_depths, exist_ok=True)
    
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    
    print("\n[STEP 1/2] Copy and rename RGB images...")
    copy_and_rename_images(source_frames, dest_images)
    
    print("\n[STEP 2/2] Convert depth maps...")
    convert_depth_images(source_depths, dest_depths, focal_length=20.0)
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"Dataset prepared at: {os.path.join(base_path, 'data', 'blender_colon')}")
    print(f"  - RGB images: {dest_images}")
    print(f"  - Depth maps: {dest_depths}")
    
    num_images = len([f for f in os.listdir(dest_images) if f.endswith('.png')])
    num_depths = len([f for f in os.listdir(dest_depths) if f.endswith('.npy')])
    print(f"\nStatistics:")
    print(f"  - Number of RGB images: {num_images}")
    print(f"  - Number of depth maps: {num_depths}")

if __name__ == "__main__":
    main()
