import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from utils.data import create_dataloaders
from utils.models import build_baseline, build_resnet50, unfreeze_top_layers, param_groups


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum=1):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    optimizer.zero_grad()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        acc = accuracy_from_logits(logits, y)
        total_loss += loss.item() * grad_accum
        total_acc += acc
        n_batches += 1
    return total_loss / n_batches, total_acc / n_batches


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            acc = accuracy_from_logits(logits, y)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_targets.append(y.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return total_loss / n_batches, total_acc / n_batches, preds, targets


def plot_curves(history: Dict[str, list], title: str, path: str):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_and_report(model, loader, class_names, device, name: str, viz_dir: str):
    criterion = nn.CrossEntropyLoss()
    loss, acc, preds, targets = eval_model(model, loader, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted")
    print(f"{name} - loss: {loss:.4f}, acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    print(classification_report(targets, preds, target_names=class_names))
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    out_path = os.path.join(viz_dir, f"cm_{name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm_path": out_path,
    }


def train_baseline(cfg, device, loaders, class_names):
    model = build_baseline(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0
    best_path = os.path.join(cfg["output_dir"], "baseline_best.pth")

    for epoch in range(cfg["baseline_epochs"]):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, cfg["grad_accum_steps"])
        val_loss, val_acc, _, _ = eval_model(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[Baseline] Epoch {epoch+1}/{cfg['baseline_epochs']} - train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "class_names": class_names}, best_path)

    plot_curves(history, "Baseline CNN", os.path.join(cfg["viz_dir"], "baseline_curves.png"))
    return model, history


def train_resnet(cfg, device, loaders, class_names):
    model = build_resnet50(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: head only
    optimizer_head = AdamW(model.fc.parameters(), lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
    history_head = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0
    head_ckpt = os.path.join(cfg["output_dir"], "resnet50_head.pth")
    print("Phase 1: Feature extraction (backbone frozen)")
    for epoch in range(cfg["num_epochs_head"]):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer_head, device, cfg["grad_accum_steps"])
        val_loss, val_acc, _, _ = eval_model(model, loaders["val"], criterion, device)
        history_head["train_loss"].append(train_loss)
        history_head["val_loss"].append(val_loss)
        history_head["train_acc"].append(train_acc)
        history_head["val_acc"].append(val_acc)
        print(f"[Head] Epoch {epoch+1}/{cfg['num_epochs_head']} - train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "class_names": class_names}, head_ckpt)

    plot_curves(history_head, "ResNet50 - Head Training", os.path.join(cfg["viz_dir"], "resnet_head_curves.png"))

    # Phase 2: fine-tune top layers
    unfreeze_top_layers(model, trainable_layers=cfg["trainable_layers_ft"])
    optimizer_ft = AdamW(param_groups(model, base_lr=cfg["lr_backbone"], head_lr=cfg["lr_head"]), weight_decay=cfg["weight_decay"])
    scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode="min", factor=0.5, patience=2)
    history_ft = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_ft = 0.0
    ft_ckpt = os.path.join(cfg["output_dir"], "resnet50_finetuned.pth")
    patience = cfg["early_stop_patience"]
    wait = 0
    print("Phase 2: Fine-tuning (top layers unfrozen)")
    for epoch in range(cfg["num_epochs_ft"]):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer_ft, device, cfg["grad_accum_steps"])
        val_loss, val_acc, _, _ = eval_model(model, loaders["val"], criterion, device)
        scheduler_ft.step(val_loss)
        history_ft["train_loss"].append(train_loss)
        history_ft["val_loss"].append(val_loss)
        history_ft["train_acc"].append(train_acc)
        history_ft["val_acc"].append(val_acc)
        print(f"[FT] Epoch {epoch+1}/{cfg['num_epochs_ft']} - train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")
        if val_acc > best_val_ft:
            best_val_ft = val_acc
            torch.save({"model_state": model.state_dict(), "class_names": class_names}, ft_ckpt)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    plot_curves(history_ft, "ResNet50 - Fine-tuning", os.path.join(cfg["viz_dir"], "resnet_ft_curves.png"))
    return model, history_head, history_ft


def main():
    parser = argparse.ArgumentParser(description="Train image classifier with transfer learning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["viz_dir"]).mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loaders, sizes, class_names = create_dataloaders(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        image_size=cfg["image_size"],
    )
    print("Class names:", class_names)
    print("Sizes:", sizes)

    results = {}

    if cfg.get("model_type", "resnet50") == "baseline":
        model, hist = train_baseline(cfg, device, loaders, class_names)
        results["history_baseline"] = hist
        torch.save({"model_state": model.state_dict(), "class_names": class_names}, os.path.join(cfg["output_dir"], "baseline_final.pth"))
        results["baseline_test"] = evaluate_and_report(model, loaders["test"], class_names, device, "baseline", cfg["viz_dir"])
    else:
        model, hist_head, hist_ft = train_resnet(cfg, device, loaders, class_names)
        results["history_head"] = hist_head
        results["history_ft"] = hist_ft
        torch.save({"model_state": model.state_dict(), "class_names": class_names, "config": cfg}, os.path.join(cfg["output_dir"], "resnet50_final.pth"))
        results["resnet_test"] = evaluate_and_report(model, loaders["test"], class_names, device, "resnet50_ft", cfg["viz_dir"])

    # Save histories/metrics
    with open(os.path.join(cfg["output_dir"], "run_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to run_results.json")


if __name__ == "__main__":
    main()
