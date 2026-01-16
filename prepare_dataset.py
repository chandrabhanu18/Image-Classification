"""
Download and prepare Dogs vs Cats dataset from Kaggle.
Requires kaggle API credentials in ~/.kaggle/kaggle.json
"""
import os
import shutil
import zipfile
from pathlib import Path
import random

def download_dataset(data_dir: str = "./data"):
    """Download Dogs vs Cats from Kaggle."""
    print("Downloading Dogs vs Cats dataset from Kaggle...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download using Kaggle API
    os.system(f"kaggle competitions download -c dogs-vs-cats -p {data_dir}")
    
    # Extract the main zip
    zip_path = os.path.join(data_dir, "dogs-vs-cats.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
    
    # Extract train.zip
    train_zip = os.path.join(data_dir, "train.zip")
    if os.path.exists(train_zip):
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(train_zip)
    
    # Clean up test zip if exists
    test_zip = os.path.join(data_dir, "test1.zip")
    if os.path.exists(test_zip):
        os.remove(test_zip)
    
    print("Dataset downloaded and extracted.")


def organize_dataset(data_dir: str = "./data", train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Organize images into train/val/test splits with class folders."""
    source_dir = os.path.join(data_dir, "train")
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found!")
        return
    
    # Get all images
    all_images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)
    random.shuffle(all_images)
    
    # Split images by class
    dog_images = [f for f in all_images if f.startswith('dog.')]
    cat_images = [f for f in all_images if f.startswith('cat.')]
    
    print(f"Found {len(dog_images)} dog images and {len(cat_images)} cat images")
    
    # Calculate splits
    def split_list(lst, train_r, val_r):
        n = len(lst)
        train_n = int(n * train_r)
        val_n = int(n * val_r)
        return lst[:train_n], lst[train_n:train_n+val_n], lst[train_n+val_n:]
    
    dog_train, dog_val, dog_test = split_list(dog_images, train_ratio, val_ratio)
    cat_train, cat_val, cat_test = split_list(cat_images, train_ratio, val_ratio)
    
    # Create directory structure
    splits = {
        'train': {'dog': dog_train, 'cat': cat_train},
        'val': {'dog': dog_val, 'cat': cat_val},
        'test': {'dog': dog_test, 'cat': cat_test},
    }
    
    for split, classes in splits.items():
        for cls, images in classes.items():
            dest_dir = os.path.join(data_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in images:
                src = os.path.join(source_dir, img)
                dst = os.path.join(dest_dir, img)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    
    print(f"\nDataset organized:")
    print(f"Train: {len(dog_train)} dogs, {len(cat_train)} cats")
    print(f"Val: {len(dog_val)} dogs, {len(cat_val)} cats")
    print(f"Test: {len(dog_test)} dogs, {len(cat_test)} cats")
    
    # Clean up original train folder
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    print("Original train folder cleaned up.")


if __name__ == "__main__":
    data_path = "./data"
    download_dataset(data_path)
    organize_dataset(data_path, train_ratio=0.7, val_ratio=0.15)
    print("\nDataset ready at:", os.path.abspath(data_path))
