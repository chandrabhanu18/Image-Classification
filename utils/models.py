from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_baseline(num_classes: int) -> nn.Module:
    return BaselineCNN(num_classes=num_classes)


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_top_layers(model: nn.Module, trainable_layers: int = 10) -> None:
    """Unfreeze top K layers (counted from the end) for fine-tuning."""
    layers = list(model.children())
    # Flatten nested layers for more control
    params = list(model.parameters())
    for p in params[-trainable_layers:]:
        p.requires_grad = True


def param_groups(model: nn.Module, base_lr: float, head_lr: float) -> Tuple[list, list]:
    """Return parameter groups for differential learning rates."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": head_lr},
    ]
