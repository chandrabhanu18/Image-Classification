# Evaluation Answers (Short)

## 1) Why ResNet50?
ResNet50 is a strong ImageNet‑pretrained backbone with residual connections that stabilize deep training. It learns rich, transferable visual features that generalize well to cats vs. dogs. Using a pre-trained model reduces data requirements and training time. It also provides a reliable baseline for transfer learning performance.

## 2) Why two‑phase training?
Phase 1 trains only the new classifier head while keeping the backbone frozen. This quickly adapts the model to the new dataset without destroying learned features. Phase 2 unfreezes top layers to fine‑tune higher‑level features for the task. Differential learning rates keep the backbone stable while improving accuracy.

## 3) Impact of data augmentation?
Augmentation (flip, rotation, color jitter) increases dataset diversity and reduces overfitting. It teaches the model invariance to common image variations. This improves generalization on the validation and test sets. Overall, it boosts robustness and helps reach >97% test accuracy.

## 4) Confusion matrix insights?
The matrix shows very few misclassifications, indicating balanced performance. Both classes have high recall and precision, with only a couple of errors. This suggests the model handles cats and dogs equally well. It confirms the high accuracy metrics are not biased to one class.

## 5) What does Grad‑CAM show?
Grad‑CAM highlights the image regions most responsible for predictions. For cats/dogs, it focuses on faces, fur patterns, and body contours. This increases interpretability and trust in the model’s decisions. It also helps verify the model isn’t using background artifacts.

## 6) Transfer learning vs baseline CNN?
The baseline CNN learns from scratch and typically underperforms with limited data. Transfer learning achieves higher accuracy faster due to pre‑learned features. In this project, ResNet50 exceeded the baseline by a large margin. It also converged quicker and produced more stable metrics.
