# Transfer Learning Image Classification

A production-ready PyTorch implementation demonstrating transfer learning for binary image classification. This project achieves **97.8% test accuracy** using ResNet50 pre-trained on ImageNet, fine-tuned on a Dogs vs. Cats dataset through a two-phase training strategy.

## 🎯 Key Features

- **Transfer Learning:** ResNet50 backbone with ImageNet pre-trained weights
- **Two-Phase Training:** Feature extraction (frozen backbone) → Fine-tuning (differential learning rates)
- **Data Augmentation:** Random flips, rotations, and color jittering for robustness
- **Comprehensive Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix
- **Model Interpretability:** Grad-CAM visualizations showing attention regions
- **Baseline Comparison:** Custom CNN trained from scratch for performance benchmarking
- **Docker Support:** Containerized environment for reproducible deployment
- **Modular Design:** Clean separation of data, models, and training logic

## 📁 Project Structure

```
week-8/
├── data/                          # Dataset organized in train/val/test splits
│   ├── train/cat/  train/dog/
│   ├── val/cat/    val/dog/
│   └── test/cat/   test/dog/
├── models/                        # Saved model checkpoints
│   ├── resnet50_head.pth         # Phase 1: Feature extraction
│   ├── resnet50_finetuned.pth    # Phase 2: Fine-tuned (BEST)
│   └── resnet50_final.pth        # Final with metadata
├── visualizations/                # Generated plots and visualizations
│   ├── resnet_head_curves.png    # Phase 1 training curves
│   ├── resnet_ft_curves.png      # Phase 2 training curves
│   ├── cm_resnet50_ft.png        # Confusion matrix
│   └── gradcam_test.png          # Grad-CAM attention maps
├── utils/                         # Utility modules
│   ├── data.py                   # Data loading and augmentation
│   ├── models.py                 # ResNet50 and baseline CNN builders
│   └── gradcam.py                # Grad-CAM implementation
├── train.py                       # Two-phase training pipeline
├── predict.py                     # Inference script with Grad-CAM support
├── create_sample_data.py          # Generate synthetic dataset
├── transfer_learning.ipynb        # Complete Jupyter notebook workflow
├── config.yaml                    # Hyperparameter configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker orchestration
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

**Option A: Synthetic Data (for quick testing)**
```bash
python create_sample_data.py
```

**Option B: Real Dataset**
- Download from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
- Organize into `data/train/`, `data/val/`, `data/test/` with `cat/` and `dog/` subfolders

### 3. Training

```bash
# Run two-phase training (feature extraction + fine-tuning)
python train.py
```

**Expected Output:**
- Phase 1: ~99.6% validation accuracy (3 epochs)
- Phase 2: ~99.6% validation accuracy (5 epochs)
- Test: ~97.8% accuracy with 99% precision/recall

### 4. Inference

```bash
# Basic prediction
python predict.py --checkpoint models/resnet50_finetuned.pth --image data/test/cat/cat.5.jpg

# With Grad-CAM visualization
python predict.py --checkpoint models/resnet50_finetuned.pth --image data/test/cat/cat.5.jpg --gradcam
```

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t image-classification .

# Run inference in container
docker run --rm -v "${PWD}:/app" image-classification python predict.py --checkpoint models/resnet50_finetuned.pth --image data/test/cat/cat.5.jpg
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.8% |
| Precision | 99.0% |
| Recall | 99.0% |
| F1-Score | 99.0% |
| Training Time | ~10-15 min (CPU) |

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.1.0+
- torchvision 0.16.0+
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Pillow, OpenCV
- PyYAML

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step setup guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Detailed project overview
- **[CONCEPTUAL_UNDERSTANDING.md](CONCEPTUAL_UNDERSTANDING.md)** - Theory and concepts
- **[EVALUATION_ANSWERS_SHORT.md](EVALUATION_ANSWERS_SHORT.md)** - Key evaluation insights
- **[DOCKER.md](DOCKER.md)** - Docker deployment guide

## 📝 Configuration

Edit [config.yaml](config.yaml) to customize hyperparameters:

```yaml
batch_size: 32
num_epochs_head: 3        # Phase 1 epochs
num_epochs_ft: 5          # Phase 2 epochs
lr_head: 0.0003          # Head learning rate
lr_backbone: 0.00001     # Backbone learning rate
early_stop_patience: 5
trainable_layers_ft: 10  # Top N layers to unfreeze
```

## 🐳 Docker

See [DOCKER.md](DOCKER.md) for complete Docker setup and usage instructions.

## 📄 License

MIT License - Feel free to use this project for educational purposes.

## 👤 Author

**Chandrabhanu** - [GitHub](https://github.com/chandrabhanu18/Image-Classification)

