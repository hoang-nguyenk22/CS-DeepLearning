"""
error_analysis.py — Extension: Error Analysis
Loads a trained checkpoint, runs inference on the test set, and produces:
  1. A grid of misclassified images with true/pred labels
  2. Per-class accuracy bar chart (highlights worst classes)
  3. Top-K most confused class pairs heatmap
Usage:
    python error_analysis.py --model resnet50 \
        --checkpoint checkpoints/resnet50_finetune_bs32_auglight_seed42_best.pth \
        --output_dir results/error_analysis
"""

import os
import argparse
import random
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from dataset import get_dataloaders
from models import get_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denormalize(tensor):
    """Undo ImageNet normalisation for display."""
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)

def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_imgs, all_probs = [], [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_imgs.append(imgs.cpu())
            all_probs.append(probs.cpu())
    return (torch.cat(all_labels), torch.cat(all_preds),
            torch.cat(all_imgs),  torch.cat(all_probs))

# ─────────────────────────────────────────────
# 1.  Misclassified Image Grid
# ─────────────────────────────────────────────
def plot_misclassified(labels, preds, imgs, probs, class_names, output_dir, n=20, seed=42):
    random.seed(seed)
    wrong_idx = (labels != preds).nonzero(as_tuple=True)[0].tolist()
    if not wrong_idx:
        print("No misclassified samples — model is perfect on test set?")
        return
    sample = random.sample(wrong_idx, min(n, len(wrong_idx)))

    cols = 5
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
    fig.suptitle("Misclassified Samples", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for ax_i, idx in enumerate(sample):
        img  = denormalize(imgs[idx]).permute(1, 2, 0).numpy()
        true = class_names[labels[idx]]
        pred = class_names[preds[idx]]
        conf = probs[idx][preds[idx]].item()
        axes[ax_i].imshow(img)
        axes[ax_i].axis('off')
        axes[ax_i].set_title(
            f"True: {true}\nPred: {pred}\n({conf*100:.1f}%)",
            fontsize=7.5, color='red' if true != pred else 'green'
        )

    for ax_i in range(len(sample), len(axes)):
        axes[ax_i].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "misclassified_grid.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────
# 2.  Per-class Accuracy
# ─────────────────────────────────────────────
def plot_per_class_accuracy(labels, preds, class_names, output_dir, top_k=30):
    n_classes = len(class_names)
    per_class_correct = np.zeros(n_classes)
    per_class_total   = np.zeros(n_classes)

    for t, p in zip(labels.numpy(), preds.numpy()):
        per_class_total[t]   += 1
        per_class_correct[t] += int(t == p)

    acc = per_class_correct / np.maximum(per_class_total, 1)

    # Show bottom-K (worst) classes
    sorted_idx = np.argsort(acc)[:top_k]
    names_k    = [class_names[i] for i in sorted_idx]
    accs_k     = acc[sorted_idx]

    plt.figure(figsize=(14, 8))
    colors = ['#e74c3c' if a < 0.5 else '#f39c12' if a < 0.75 else '#2ecc71' for a in accs_k]
    bars = plt.barh(names_k, accs_k, color=colors)
    plt.xlabel("Accuracy")
    plt.title(f"Bottom-{top_k} Classes by Test Accuracy")
    plt.xlim(0, 1)
    plt.axvline(acc.mean(), color='navy', linestyle='--', label=f'Mean={acc.mean():.2f}')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_accuracy.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")
    return acc

# 3. Top Confused Pairs Heatmap
def plot_top_confused_pairs(labels, preds, class_names, output_dir, top_k=20):
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    np.fill_diagonal(cm, 0)

    flat = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j]
    flat.sort(reverse=True)
    top = flat[:top_k]

    pairs  = [f"{class_names[i]}\n→ {class_names[j]}" for _, i, j in top]
    counts = [v for v, _, _ in top]

    plt.figure(figsize=(14, 7))
    plt.barh(pairs[::-1], counts[::-1], color='#e74c3c')
    plt.xlabel("Times Confused")
    plt.title(f"Top-{top_k} Most Confused Class Pairs")
    plt.tight_layout()
    path = os.path.join(output_dir, "top_confused_pairs.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

    involved = sorted(set([i for _, i, _ in top] + [j for _, _, j in top]))
    sub_cm   = cm[np.ix_(involved, involved)]
    sub_names = [class_names[i] for i in involved]
    plt.figure(figsize=(max(10, len(involved)), max(8, len(involved)-2)))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=sub_names, yticklabels=sub_names)
    plt.title("Confusion Sub-Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path2 = os.path.join(output_dir, "confusion_submatrix.png")
    plt.savefig(path2, dpi=200)
    plt.close()
    print(f"Saved: {path2}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, required=True,
                        choices=['resnet50', 'efficientnet_b0', 'vit_b_16', 'deit_s'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir',   type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='results/error_analysis')
    parser.add_argument('--n_examples', type=int, default=20,
                        help='Number of misclassified images to show')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed)
    num_classes = len(class_names)

    print(f"Loading {args.model}: Adaptation to {num_classes} classes")
    model = get_model(args.model, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)

    labels, preds, imgs, probs = run_inference(model, test_loader, device)

    acc = (labels == preds).float().mean().item()
    print(f"\nOverall Test Accuracy: {acc*100:.2f}%  |  Errors: {(labels!=preds).sum().item()}")

    plot_misclassified(labels, preds, imgs, probs, class_names,
                       args.output_dir, n=args.n_examples, seed=args.seed)
    plot_per_class_accuracy(labels, preds, class_names, args.output_dir)
    plot_top_confused_pairs(labels, preds, class_names, args.output_dir)

    print("\nError analysis complete. Results saved to:", args.output_dir)

if __name__ == '__main__':
    main()
