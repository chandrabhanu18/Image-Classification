import os
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and eval transforms with augmentation for train."""
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], list]:
    """Create train/val/test dataloaders and metadata."""
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_tfms, eval_tfms = build_transforms(image_size)

    datasets_map = {
        "train": datasets.ImageFolder(train_dir, transform=train_tfms),
        "val": datasets.ImageFolder(val_dir, transform=eval_tfms),
        "test": datasets.ImageFolder(test_dir, transform=eval_tfms),
    }

    class_names = datasets_map["train"].classes

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, ds in datasets_map.items()
    }

    sizes = {split: len(ds) for split, ds in datasets_map.items()}
    return loaders, sizes, class_names
