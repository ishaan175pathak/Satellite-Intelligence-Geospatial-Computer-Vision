import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from tqdm import tqdm

# Define the path to the dataset
CWD: str = os.getcwd()
DATASET_PATH: str = os.path.join(CWD, "data", "raw", "whu_building")

# Function to check image-mask alignment and mask format
def check_alignment_and_format(split):
    SPLIT_DIR: str = os.path.join(DATASET_PATH, split)
    IMAGE_DIR: str = os.path.join(SPLIT_DIR, "Image")
    MASK_DIR: str = os.path.join(SPLIT_DIR, "Mask")

    # Check that both Image and Mask directories exist
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
        print(f"[ERROR] Missing Image or Mask directory for split: {split}")
        return

    # Get list of image files
    image_files: List[str] = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    mask_files: List[str] = [f for f in os.listdir(MASK_DIR) if f.endswith(".png")]

    # Check that the number of images and masks match
    if len(image_files) != len(mask_files):
        print(f"[ERROR] Mismatched number of images and masks for split: {split}")
        print(f"  Images: {len(image_files)} | Masks: {len(mask_files)}")
        return

    # Check that the filenames match
    for image_file, mask_file in zip(image_files, mask_files):
        if image_file != mask_file:
            print(f"[ERROR] Mismatched filenames for split: {split}")
            print(f"  Image: {image_file} | Mask: {mask_file}")
            return

    # Check that the mask is a binary mask (black and white)
    for image_file, mask_file in tqdm(zip(image_files, mask_files), desc=f"Checking {split} masks", total=len(image_files)):
        image_path: str = os.path.join(IMAGE_DIR, image_file)
        mask_path: str = os.path.join(MASK_DIR, mask_file)
    
        # Load the mask

        mask_array: np.ndarray = np.array(Image.open(mask_path).convert("L"))  # Convert to grayscale
        mask_unique_values: np.ndarray = np.unique(mask_array)
        
        # Check if all unique values are within the allowed set {0, 255}
        if not np.all(np.isin(mask_unique_values, [0, 255])):
            print(f"\n[ERROR] Mask contains non-binary values for split: {split}")
            print(f"  File: {mask_file} | Unique values: {mask_unique_values}")
            return

    print(f"[SUCCESS] All masks for split: {split} are binary and aligned with images.")

# Check alignment and format for each split
check_alignment_and_format(split = "train")