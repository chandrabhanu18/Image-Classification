# Quick Start Guide - Transfer Learning Image Classifier

## 🚀 5-Minute Quickstart

### Option 1: Test with Synthetic Data (Fastest)
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Create test dataset
python create_sample_data.py

# Step 3: Train (takes ~15-30 min on CPU)
python train.py --config config.yaml

# Step 4: Check results
ls models/       # Model checkpoints
ls visualizations/   # Training curves & confusion matrices
```

### Option 2: Run Notebook (Interactive)
```bash
# Step 1: Install
pip install -r requirements.txt jupyter

# Step 2: Create data
python create_sample_data.py

# Step 3: Launch notebook
jupyter notebook transfer_learning.ipynb

# Step 4: Run all cells (Kernel → Restart & Run All)
```

## 📊 What You'll Get

After training completes, you'll have:

### 1. Model Checkpoints (`models/`)
- `baseline_best.pth` - Best baseline CNN (from scratch)
- `resnet50_head.pth` - After Phase 1 (feature extraction)
- `resnet50_finetuned.pth` - After Phase 2 (fine-tuning) **← BEST MODEL**
- `resnet50_final.pth` - Final with metadata
- `run_results.json` - Training history and metrics

### 2. Visualizations (`visualizations/`)
- `baseline_curves.png` - Baseline training progress
- `resnet_head_curves.png` - Phase 1 training curves
- `resnet_ft_curves.png` - Phase 2 training curves
- `cm_baseline.png` - Baseline confusion matrix
- `cm_resnet50_ft.png` - Transfer learning confusion matrix
- `gradcam_*.png` - Model attention heatmaps

### 3. Console Output
```
Using device: cpu
Class names: ['cat', 'dog']
Sizes: {'train': 1000, 'val': 200, 'test': 200}

Phase 1: Feature extraction (backbone frozen)
[Head] Epoch 1/3 - train_acc: 0.750, val_acc: 0.800
[Head] Epoch 2/3 - train_acc: 0.850, val_acc: 0.880
[Head] Epoch 3/3 - train_acc: 0.900, val_acc: 0.920

Phase 2: Fine-tuning (top layers unfrozen)
[FT] Epoch 1/5 - train_acc: 0.920, val_acc: 0.940
[FT] Epoch 2/5 - train_acc: 0.950, val_acc: 0.960
...

resnet50_ft - loss: 0.1234, acc: 0.9600, precision: 0.9580, recall: 0.9620, f1: 0.9600
              precision    recall  f1-score   support
         cat       0.96      0.95      0.96       100
         dog       0.96      0.97      0.96       100
    accuracy                           0.96       200
```

## 🎯 Using Trained Model for Predictions

### Single Image Prediction
```bash
python predict.py \
    --checkpoint models/resnet50_finetuned.pth \
    --image path/to/your/image.jpg \
    --top-k 2
```

**Output:**
```
Using device: cpu
Loading model from models/resnet50_finetuned.pth
Classes: ['cat', 'dog']
Processing image: path/to/your/image.jpg

Top-2 Predictions:
  1. dog: 95.23%
  2. cat: 4.77%
```

### With Grad-CAM Visualization
```bash
python predict.py \
    --checkpoint models/resnet50_finetuned.pth \
    --image path/to/your/image.jpg \
    --gradcam \
    --output heatmap.png
```

This saves `heatmap.png` showing which parts of the image influenced the prediction.

## ⚙️ Configuration Options

Edit `config.yaml` to customize training:

```yaml
# Dataset
data_root: "./data"           # Path to data folder
image_size: 224               # Input image size

# Training
batch_size: 32                # Batch size (reduce if out of memory)
num_epochs_head: 3            # Phase 1 epochs
num_epochs_ft: 5              # Phase 2 epochs

# Learning Rates
lr_head: 0.0003               # Learning rate for new head
lr_backbone: 0.00001          # Learning rate for backbone (fine-tuning)

# Regularization
weight_decay: 0.0001          # L2 regularization
early_stop_patience: 5        # Stop if no improvement for N epochs

# Fine-tuning
trainable_layers_ft: 10       # Number of top layers to unfreeze
```

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size in `config.yaml`
```yaml
batch_size: 16  # or even 8
```

### Issue: Training Too Slow on CPU
**Solutions:**
1. Reduce dataset size in `create_sample_data.py`:
   ```python
   train_per_class=200,  # Instead of 500
   val_per_class=50,
   test_per_class=50
   ```

2. Reduce epochs in `config.yaml`:
   ```yaml
   num_epochs_head: 2
   num_epochs_ft: 3
   ```

3. Use GPU if available (automatic detection)

### Issue: Import Errors
**Solution:** Reinstall requirements
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Data Not Found
**Solution:** Check data directory structure
```bash
data/
  train/
    cat/  (contains .jpg files)
    dog/  (contains .jpg files)
  val/
    cat/
    dog/
  test/
    cat/
    dog/
```

Run `python create_sample_data.py` if missing.

## 📈 Expected Performance

### With Synthetic Data (1400 images)
- **Baseline CNN:** 60-75% accuracy
- **ResNet50 Transfer:** 85-95% accuracy
- **Training Time (CPU):** 15-30 minutes

### With Real Kaggle Data (25,000 images)
- **Baseline CNN:** 75-85% accuracy
- **ResNet50 Transfer:** 95-98% accuracy
- **Training Time (GPU):** 10-20 minutes

## 🎓 Understanding the Output

### Training Curves
- **Loss going down:** ✅ Model is learning
- **Validation > Train:** ⚠️ Possible overfitting
- **Both plateaued:** ✅ Converged

### Confusion Matrix
```
         Pred Cat  Pred Dog
True Cat    95        5      ← 5 cats misclassified as dogs
True Dog     3       97      ← 3 dogs misclassified as cats
```

**Interpretation:**
- Diagonal (95, 97): Correct predictions
- Off-diagonal: Errors
- High diagonal values = Good model

### Grad-CAM Heatmap
- **Red/Hot regions:** Model focuses here for prediction
- **Blue/Cool regions:** Ignored by model

**Good sign:** Model highlights faces, bodies (relevant features)  
**Bad sign:** Model highlights backgrounds, watermarks (spurious)

## 🔄 Retraining Tips

### To improve accuracy:
1. **More data:** Download full Kaggle dataset
2. **More augmentation:** Add in `utils/data.py`
3. **Longer training:** Increase epochs
4. **Better architecture:** Try EfficientNet instead of ResNet50

### To speed up training:
1. **Use GPU:** Training will be 5-10x faster
2. **Smaller model:** Use ResNet18 instead of ResNet50
3. **Less data:** Reduce dataset size (trades accuracy for speed)
4. **Mixed precision:** Enable AMP (automatic mixed precision)

## 📚 Learning Resources

### Understanding the Code
1. **Data Pipeline:** `utils/data.py` - See augmentation transforms
2. **Models:** `utils/models.py` - ResNet50 vs Baseline architecture
3. **Training:** `train.py` - Two-phase training logic
4. **Grad-CAM:** `utils/gradcam.py` - Interpretability implementation

### Key Concepts
- **Transfer Learning:** Using pre-trained weights (lines in `utils/models.py:47`)
- **Freezing Layers:** `param.requires_grad = False` (line 49)
- **Two-Phase Training:** `train_resnet()` in `train.py` (lines 144-206)
- **Differential LR:** `param_groups()` in `utils/models.py` (lines 59-71)

## 🎯 Next Steps

### For Experimentation
1. Try different pre-trained models (VGG, EfficientNet)
2. Experiment with augmentation strength
3. Tune learning rates and epochs
4. Try different optimizers (SGD with momentum)

### For Production
1. Train on full dataset
2. Create REST API for predictions
3. Deploy with Docker
4. Add monitoring and logging

### For Learning
1. Read `CONCEPTUAL_UNDERSTANDING.md` for theory
2. Modify `config.yaml` and observe changes
3. Add new augmentations in `utils/data.py`
4. Implement other interpretability methods (LIME, SHAP)

---

## ✅ Verification Checklist

Before submitting, verify:
- [ ] All files present (README, notebook, scripts, utils/)
- [ ] `requirements.txt` includes all dependencies
- [ ] Training runs without errors
- [ ] Model checkpoints saved in `models/`
- [ ] Visualizations generated in `visualizations/`
- [ ] README explains project clearly
- [ ] Code is well-commented
- [ ] Reproducibility: fixed seeds, pinned versions

---

**Need Help?**
1. Check `README.md` for detailed documentation
2. Review `PROJECT_SUMMARY.md` for complete feature list
3. Read `CONCEPTUAL_UNDERSTANDING.md` for theory
4. Examine notebook for step-by-step walkthrough

**Happy Training! 🚀**
