"""
Inference script for trained image classifier.
Load a saved model and predict on new images.
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.models import build_resnet50, build_baseline
from utils.gradcam import GradCAM, overlay_heatmap


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    
    # Determine model type from path
    if "baseline" in checkpoint_path:
        model = build_baseline(num_classes)
    else:
        model = build_resnet50(num_classes, pretrained=False)
    
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    return model, class_names


def preprocess_image(image_path: str, image_size: int = 224):
    """Load and preprocess single image."""
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)


def predict(model, image_tensor, class_names, device, top_k: int = 2):
    """Run inference and return top-k predictions."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append((class_names[idx.item()], prob.item()))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference with trained classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top predictions")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualization")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output path for Grad-CAM")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, class_names = load_model(args.checkpoint, device)
    print(f"Classes: {class_names}")
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Predict
    results = predict(model, image_tensor, class_names, device, args.top_k)
    print(f"\nTop-{args.top_k} Predictions:")
    for i, (cls, prob) in enumerate(results, 1):
        print(f"  {i}. {cls}: {prob*100:.2f}%")
    
    # Grad-CAM
    if args.gradcam and "resnet" in args.checkpoint.lower():
        print(f"\nGenerating Grad-CAM...")
        cam = GradCAM(model, target_layer=model.layer4[-1])
        pred_idx = class_names.index(results[0][0])
        heatmap = cam(image_tensor.to(device), class_idx=pred_idx)
        original, overlay_img = overlay_heatmap(image_tensor.squeeze(0), heatmap)
        overlay_img.save(args.output)
        print(f"Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
