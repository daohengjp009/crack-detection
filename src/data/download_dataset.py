#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and prepare the Concrete Crack Images dataset from IBM Cognitive Class.
"""

import os
import shutil
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm

# URL for the dataset
DATASET_URL = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels from this file
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a URL to a file with a progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_dataset():
    """Download and prepare the dataset."""
    # Create directories if they don't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set the path for the downloaded zip file
    zip_path = RAW_DATA_DIR / "concrete_crack_images.zip"
    
    # Download the dataset if it doesn't exist
    if not zip_path.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        download_url(DATASET_URL, zip_path)
    else:
        print(f"Dataset already downloaded at {zip_path}")
    
    # Extract the dataset
    extract_dir = RAW_DATA_DIR / "concrete_crack_images_temp"
    extract_dir.mkdir(exist_ok=True)
    
    print(f"Extracting dataset to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # The dataset contains Positive and Negative folders
    # Organize into processed directory with train/val/test splits
    positive_dir = extract_dir / "Positive"
    negative_dir = extract_dir / "Negative"
    
    if not positive_dir.exists() or not negative_dir.exists():
        print("Error: Expected directory structure not found in the downloaded data.")
        print(f"Contents of {extract_dir}:")
        for item in extract_dir.iterdir():
            print(f"  {item}")
        return
    
    print("Dataset extracted successfully!")
    print(f"Positive samples: {len(list(positive_dir.glob('*.jpg')))}")
    print(f"Negative samples: {len(list(negative_dir.glob('*.jpg')))}")
    
    print("\nDataset is ready for processing!")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    
    # Note: Further processing like train/val/test split will be implemented in a separate script


if __name__ == "__main__":
    prepare_dataset() 