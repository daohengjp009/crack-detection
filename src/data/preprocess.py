#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess the Concrete Crack Images dataset and prepare it for model training.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels from this file
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Dataset specific paths
DATASET_EXTRACT_DIR = RAW_DATA_DIR / "concrete_crack_images_temp"
POSITIVE_DIR = DATASET_EXTRACT_DIR / "Positive"
NEGATIVE_DIR = DATASET_EXTRACT_DIR / "Negative"

# Processed data directories
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Target image size for the model
TARGET_SIZE = (224, 224)  # Standard size for many CNN architectures

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def setup_directories() -> None:
    """Create the necessary directories for processed data."""
    # Create main directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create class subdirectories
    (TRAIN_DIR / "positive").mkdir(exist_ok=True)
    (TRAIN_DIR / "negative").mkdir(exist_ok=True)
    (VAL_DIR / "positive").mkdir(exist_ok=True)
    (VAL_DIR / "negative").mkdir(exist_ok=True)
    (TEST_DIR / "positive").mkdir(exist_ok=True)
    (TEST_DIR / "negative").mkdir(exist_ok=True)


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    return list(directory.glob("*.jpg")) + list(directory.glob("*.png"))


def split_dataset(
    positive_files: List[Path], 
    negative_files: List[Path]
) -> Dict[str, Dict[str, List[Path]]]:
    """Split dataset into train, validation and test sets."""
    # Shuffle files to ensure random distribution
    random.seed(42)  # For reproducibility
    random.shuffle(positive_files)
    random.shuffle(negative_files)
    
    # Calculate split indices
    pos_train_idx = int(len(positive_files) * TRAIN_RATIO)
    pos_val_idx = pos_train_idx + int(len(positive_files) * VAL_RATIO)
    
    neg_train_idx = int(len(negative_files) * TRAIN_RATIO)
    neg_val_idx = neg_train_idx + int(len(negative_files) * VAL_RATIO)
    
    # Split the files
    pos_train = positive_files[:pos_train_idx]
    pos_val = positive_files[pos_train_idx:pos_val_idx]
    pos_test = positive_files[pos_val_idx:]
    
    neg_train = negative_files[:neg_train_idx]
    neg_val = negative_files[neg_train_idx:neg_val_idx]
    neg_test = negative_files[neg_val_idx:]
    
    return {
        "train": {"positive": pos_train, "negative": neg_train},
        "val": {"positive": pos_val, "negative": neg_val},
        "test": {"positive": pos_test, "negative": neg_test}
    }


def preprocess_image(img_path: Path) -> Tuple[np.ndarray, str]:
    """Preprocess a single image."""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Convert from BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Get filename for saving
    output_filename = img_path.name
    
    return img, output_filename


def save_processed_image(img: np.ndarray, filename: str, output_dir: Path) -> None:
    """Save the processed image."""
    # Convert back to uint8 for saving
    img_to_save = (img * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_to_save)
    
    # Save the image
    output_path = output_dir / filename
    pil_img.save(output_path)


def process_dataset(split_files: Dict[str, Dict[str, List[Path]]]) -> None:
    """Process all images in the dataset."""
    # Process each split
    for split_name, classes in split_files.items():
        print(f"Processing {split_name} set...")
        
        # Process each class
        for class_name, files in classes.items():
            if split_name == "train":
                output_dir = TRAIN_DIR / class_name
            elif split_name == "val":
                output_dir = VAL_DIR / class_name
            else:  # test
                output_dir = TEST_DIR / class_name
            
            # Process each file
            for img_path in tqdm(files, desc=f"{split_name}/{class_name}"):
                try:
                    img, filename = preprocess_image(img_path)
                    save_processed_image(img, filename, output_dir)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


def generate_dataset_stats(split_files: Dict[str, Dict[str, List[Path]]]) -> None:
    """Generate and save dataset statistics."""
    stats = {
        "train": {
            "positive": len(split_files["train"]["positive"]),
            "negative": len(split_files["train"]["negative"]),
            "total": len(split_files["train"]["positive"]) + len(split_files["train"]["negative"])
        },
        "val": {
            "positive": len(split_files["val"]["positive"]),
            "negative": len(split_files["val"]["negative"]),
            "total": len(split_files["val"]["positive"]) + len(split_files["val"]["negative"])
        },
        "test": {
            "positive": len(split_files["test"]["positive"]),
            "negative": len(split_files["test"]["negative"]),
            "total": len(split_files["test"]["positive"]) + len(split_files["test"]["negative"])
        },
        "total": {
            "positive": len(split_files["train"]["positive"]) + len(split_files["val"]["positive"]) + len(split_files["test"]["positive"]),
            "negative": len(split_files["train"]["negative"]) + len(split_files["val"]["negative"]) + len(split_files["test"]["negative"]),
            "total": len(split_files["train"]["positive"]) + len(split_files["val"]["positive"]) + len(split_files["test"]["positive"]) +
                    len(split_files["train"]["negative"]) + len(split_files["val"]["negative"]) + len(split_files["test"]["negative"])
        }
    }
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Training set: {stats['train']['total']} images ({stats['train']['positive']} positive, {stats['train']['negative']} negative)")
    print(f"Validation set: {stats['val']['total']} images ({stats['val']['positive']} positive, {stats['val']['negative']} negative)")
    print(f"Test set: {stats['test']['total']} images ({stats['test']['positive']} positive, {stats['test']['negative']} negative)")
    print(f"Total dataset: {stats['total']['total']} images ({stats['total']['positive']} positive, {stats['total']['negative']} negative)")
    
    # Save statistics to file
    stats_file = PROCESSED_DATA_DIR / "dataset_stats.txt"
    with open(stats_file, "w") as f:
        f.write("Dataset Statistics:\n")
        f.write(f"Training set: {stats['train']['total']} images ({stats['train']['positive']} positive, {stats['train']['negative']} negative)\n")
        f.write(f"Validation set: {stats['val']['total']} images ({stats['val']['positive']} positive, {stats['val']['negative']} negative)\n")
        f.write(f"Test set: {stats['test']['total']} images ({stats['test']['positive']} positive, {stats['test']['negative']} negative)\n")
        f.write(f"Total dataset: {stats['total']['total']} images ({stats['total']['positive']} positive, {stats['total']['negative']} negative)\n")
    
    print(f"\nStatistics saved to {stats_file}")


def main():
    """Main function to preprocess the dataset."""
    print("Starting data preprocessing...")
    
    # Make sure the extract directory exists
    if not DATASET_EXTRACT_DIR.exists() or not POSITIVE_DIR.exists() or not NEGATIVE_DIR.exists():
        print("Error: Dataset not found. Please run download_dataset.py first.")
        return
    
    # Setup directories
    setup_directories()
    
    # Get image files
    positive_files = get_image_files(POSITIVE_DIR)
    negative_files = get_image_files(NEGATIVE_DIR)
    
    print(f"Found {len(positive_files)} positive examples and {len(negative_files)} negative examples.")
    
    # Split the dataset
    split_files = split_dataset(positive_files, negative_files)
    
    # Process the dataset
    process_dataset(split_files)
    
    # Generate statistics
    generate_dataset_stats(split_files)
    
    print("Data preprocessing completed!")


if __name__ == "__main__":
    main() 