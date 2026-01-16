from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


def _normalize(t: torch.Tensor) -> torch.Tensor:
    t_min, t_max = t.min(), t.max()
    if (t_max - t_min) == 0:
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.eval()
        input_tensor.requires_grad = True
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Ensure target_layer is valid.")
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = _normalize(cam)
        return cam.squeeze(0).squeeze(0)  # H, W


def overlay_heatmap(image_tensor: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4) -> Tuple:
    """Return PIL image and heatmap overlay. Assumes image_tensor is normalized CHW."""
    # Denormalize using ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = _normalize(img)

    heatmap_resized = heatmap.unsqueeze(0)
    heatmap_color = torch.zeros_like(img)
    heatmap_color[0] = heatmap_resized
    heatmap_color[1] = heatmap_resized * 0.5

    overlay = (1 - alpha) * img + alpha * heatmap_color
    return to_pil_image(img), to_pil_image(overlay)
