"""
gradcam.py — Extension: Interpretability (Grad-CAM / Attention Rollout)
Generates visual explanations for model predictions.

For CNNs (ResNet, EfficientNet): uses Grad-CAM on the last conv layer.
For ViTs: uses Attention Rollout (averages attention weights across heads/layers).

Usage:
    python gradcam.py --model resnet50 \
        --checkpoint checkpoints/resnet50_finetune_bs32_auglight_seed42_best.pth \
        --output_dir results/gradcam \
        --n_samples 16
"""

import os
import argparse
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from models import get_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denormalize(tensor):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


# ─────────────────────────────────────────────
# Grad-CAM for CNNs
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits[0, class_idx]
        score.backward()

        # Global average pool gradients → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx.item(), torch.softmax(logits, dim=1)[0]

    def remove(self):
        for h in self._hooks:
            h.remove()


def get_cnn_target_layer(model, model_name):
    """Return the last conv layer for common architectures."""
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'efficientnet' in model_name:
        return model.conv_head
    else:
        raise ValueError(f"Unknown CNN model: {model_name}")


# ─────────────────────────────────────────────
# Attention Rollout for ViTs
# ─────────────────────────────────────────────
class AttentionRollout:
    """
    Attention Rollout (Abnar & Zuidema 2020) for ViT models in timm.
    Averages across heads, multiplies layer by layer with residual weight 0.5.
    """
    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attention_maps = []
        self._hooks = []
        for block in model.blocks:
            self._hooks.append(
                block.attn.register_forward_hook(self._store_attn))

    def _store_attn(self, module, inp, output):
        # timm attention output is just the projected result, not the matrix.
        # We re-run the internal attention computation in no-grad mode.
        # Instead, we hook into the SOFTMAX output via a trick:
        # Just capture the attention weights stored by timm's Attention module.
        if hasattr(module, 'attn_drop'):
            pass  # older timm: attn stored internally
        # For newer timm: we'll get it from the module's last forward data
        # (timm stores it in module._attn if return_attention=True flag is exposed)
        # Simpler: re-derive from Q,K in the stored inputs
        self._attention_maps.append(None)  # placeholder, see __call__

    def __call__(self, x):
        self._attention_maps = []
        # Use timm's built-in attention capture via monkey-patching
        attention_maps = []

        def hook_fn(module, inp, output):
            # timm Attention: inp[0] is (B, N, C)
            B, N, C = inp[0].shape
            qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu())

        hooks = [block.attn.register_forward_hook(hook_fn) for block in self.model.blocks]

        with torch.no_grad():
            logits = self.model(x)

        for h in hooks:
            h.remove()

        # Rollout
        result = torch.eye(attention_maps[0].shape[-1])
        for attn in attention_maps:
            if self.head_fusion == 'mean':
                attn = attn.mean(dim=1)    # (B, N, N)
            elif self.head_fusion == 'max':
                attn = attn.max(dim=1)[0]
            attn = attn[0]  # first batch item

            # Discard low-attention weights
            flat = attn.flatten()
            threshold = flat.kthvalue(int(flat.numel() * self.discard_ratio))[0]
            attn[attn < threshold] = 0

            # Add residual, normalise rows
            attn = attn + torch.eye(attn.shape[0])
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = attn @ result

        # CLS token row → patch attention
        mask = result[0, 1:]  # skip CLS column
        grid = int(mask.numel() ** 0.5)
        mask = mask.reshape(grid, grid).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask, logits.argmax(dim=1).item(), torch.softmax(logits, dim=1)[0]

    def remove(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────
# Overlay CAM onto image
# ─────────────────────────────────────────────
def overlay_cam(img_tensor, cam, alpha=0.45):
    img_np = (denormalize(img_tensor).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    h, w   = img_np.shape[:2]
    cam_up = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return img_np, overlay


# ─────────────────────────────────────────────
# Plot grid
# ─────────────────────────────────────────────
def plot_cam_grid(results, class_names, output_dir, title, filename):
    n = len(results)
    cols = 4  # Original | CAM | Original | CAM
    rows = math.ceil(n / 2)  # 2 samples per row (each sample = 2 cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, (img_np, overlay, pred_cls, conf) in enumerate(results):
        row = i // 2
        col = (i % 2) * 2
        axes[row, col].imshow(img_np)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"{class_names[pred_cls]}\n({conf*100:.1f}%)", fontsize=8)
        axes[row, col+1].imshow(overlay)
        axes[row, col+1].axis('off')
        axes[row, col+1].set_title("Grad-CAM / Attention", fontsize=8)

    # Hide empty subplots
    for i in range(n, rows * 2):
        row = i // 2; col = (i % 2) * 2
        axes[row, col].axis('off')
        axes[row, col+1].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, required=True,
                        choices=['resnet50', 'efficientnet_b0', 'vit_b_16', 'deit_s'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir',   type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='results/gradcam')
    parser.add_argument('--n_samples',  type=int, default=16,
                        help='Number of images to visualise')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=1, seed=args.seed)
    num_classes = len(class_names)

    model = get_model(args.model, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    is_vit = args.model in ('vit_b_16', 'deit_s')
    correct_results, wrong_results = [], []

    if is_vit:
        cam_engine = AttentionRollout(model)
    else:
        target_layer = get_cnn_target_layer(model, args.model)
        cam_engine = GradCAM(model, target_layer)

    n_want_each = args.n_samples // 2

    for imgs, labels in tqdm(test_loader, desc="Generating CAM"):
        if len(correct_results) >= n_want_each and len(wrong_results) >= n_want_each:
            break
        img = imgs[0:1].to(device)
        lbl = labels[0].item()

        if is_vit:
            cam, pred_idx, probs = cam_engine(img)
        else:
            cam, pred_idx, probs = cam_engine(img, class_idx=None)

        conf = probs[pred_idx].item()
        img_np, overlay = overlay_cam(imgs[0], cam)

        entry = (img_np, overlay, pred_idx, conf)
        if pred_idx == lbl and len(correct_results) < n_want_each:
            correct_results.append(entry)
        elif pred_idx != lbl and len(wrong_results) < n_want_each:
            wrong_results.append(entry)

    if not is_vit:
        cam_engine.remove()

    method_name = "Attention Rollout" if is_vit else "Grad-CAM"
    plot_cam_grid(correct_results, class_names, args.output_dir,
                  f"{method_name} — Correct Predictions ({args.model})",
                  f"{args.model}_cam_correct.png")
    plot_cam_grid(wrong_results, class_names, args.output_dir,
                  f"{method_name} — Wrong Predictions ({args.model})",
                  f"{args.model}_cam_wrong.png")

    print(f"Generation complete. Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
