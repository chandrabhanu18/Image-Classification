Transfer Learning Image Classifier
-------------------------------------
Project Overview:
---------------------
A production-ready PyTorch implementation demonstrating transfer learning for image classification. This project trains a high-performance classifier using ResNet50 pre-trained on ImageNet, adapted to a custom Dogs vs. Cats dataset through a two-phase training strategy.

Key Features:
- Transfer learning with ResNet50 (ImageNet pre-trained)
- Two-phase training: feature extraction → fine-tuning
- Baseline CNN comparison from scratch
- Comprehensive evaluation: accuracy, precision, recall, F1
- Grad-CAM visualizations for model interpretability
- Reproducible setup with deterministic seeding
- Modular code structure with utility scripts

Project Structure
-------------------
week-8 mandatory task/
├── data/                    # Dataset (train/val/test splits)
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   ├── val/
│   └── test/
├── models/                  # Saved model checkpoints
├── visualizations/          # Training curves, confusion matrices, Grad-CAMs
├── utils/                   # Utility modules
│   ├── data.py             # Data loading and augmentation
│   ├── models.py           # Model builders (ResNet50, baseline CNN)
│   └── gradcam.py          # Grad-CAM implementation
├── train.py                # Command-line training script
├── transfer_learning.ipynb # Complete notebook workflow
├── config.yaml             # Hyperparameter configuration
├── requirements.txt        # Dependencies
├── create_sample_data.py   # Synthetic dataset generator
└── README.md              # This file


Quick Start
-----------

1. Environment Setup
---------------------
Run these commands in git bash:

pip install -r requirements.txt


2. Dataset Preparation
--------------------------
Option A: Use Synthetic Data (for testing):
python create_sample_data.py


Option B: Real Dataset:
- Download Dogs vs. Cats from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- Organize into `data/train/`, `data/val/`, `data/test/` with class subfolders

3. Training:
Command-line:

python train.py --config config.yaml

Requirements
-----------------
- Python 3.8+
- PyTorch 2.1.0+
- torchvision 0.16.0+
- NumPy, Matplotlib, Seaborn
- scikit-learn
- PIL/OpenCV

