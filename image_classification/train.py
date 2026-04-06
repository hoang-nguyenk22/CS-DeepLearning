import os
import argparse
import time
import random
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from models import get_model, get_layer_wise_optimizer

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping for ViT stability and general good practice
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train image classification models on FGVC Aircraft")
    parser = argparse.ArgumentParser(description="Train image classification models")
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'vit_b_16', 'deit_s'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--augment_level', type=str, default='light', choices=['light', 'strong'])
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze backbone for linear probing")
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--llrd', action='store_true', help='Use layer-wise learning rate decay')
    parser.add_argument('--exp_tag', type=str, default='', help='Experiment tag')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Seed: {args.seed}")
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size,
        augment_level=args.augment_level,
        seed=args.seed
    )
    num_classes = len(class_names)
    
    # -----------------------------
    # 3. Model & Optimizer
    # -----------------------------
    model = get_model(args.model, num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.llrd and not args.freeze_backbone:
        # Layer-wise Learning Rate Decay — third strategy in extension
        param_groups = get_layer_wise_optimizer(model, args.model, base_lr=args.lr, decay=0.65)
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"Using LLRD optimizer with {len(param_groups)} param groups (decay=0.65)")
    else:
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Filenames encode strategy, seed, optional experiment tag
    phase_lbl = 'probing' if args.freeze_backbone else ('llrd' if args.llrd else 'finetune')
    tag_part = f'_{args.exp_tag}' if args.exp_tag else ''
    csv_filename = f"{args.model}_{phase_lbl}{tag_part}_bs{args.batch_size}_aug{args.augment_level}_seed{args.seed}.csv"
    csv_path = os.path.join(args.log_dir, csv_filename)
    
    save_prefix = csv_filename.replace('.csv', '')
    save_path = os.path.join(args.output_dir, f"{save_prefix}_best.pth")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'time_sec', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # -----------------------------
    # 4. Training Loop
    # -----------------------------
    print(f"Starting {phase_lbl.upper()} for {args.model} (Epochs: {args.epochs}, BS: {args.batch_size}, Aug: {args.augment_level}, LR: {args.lr})")
    start_time = time.time()
    
    try:
        for epoch in range(args.epochs):
            ep_start = time.time()
            
            # Record current LR before step
            current_lr = optimizer.param_groups[0]['lr']
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            scheduler.step()
            
            ep_end = time.time()
            ep_time_sec = ep_end - ep_start
            mins, secs = divmod(ep_time_sec, 60)
            
            print(f"Epoch {epoch+1}/{args.epochs} [{int(mins)}m {int(secs)}s] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - LR: {current_lr:.2e}")
            
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, ep_time_sec, train_loss, train_acc, val_loss, val_acc, current_lr])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"WARNING: Out of Memory error with Batch Size {args.batch_size}.")
        else:
            raise e
            
    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)
    print(f"\nTraining completed in {int(mins)}m {int(secs)}s. Best Val Acc: {best_val_acc:.4f}")

    # -----------------------------
    # 5. Final Test Evaluation & Efficiency Metrics
    # -----------------------------
    print("\nStarting final test evaluation...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    else:
        print("Warning: Best checkpoint not found. Evaluating on last epoch weights.")
        
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Final Test Acc (Ground Truth): {test_acc:.4f}")

    # Measure inference time with CUDA Events (GPU-accurate)
    print("Measuring Inference Time (CUDA Events)...")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            model(dummy)
        if device.type == 'cuda':
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(100):
                model(dummy)
            ender.record()
            torch.cuda.synchronize()
            inference_ms = starter.elapsed_time(ender) / 100
        else:
            # CPU fallback
            t0 = time.perf_counter()
            for _ in range(100):
                model(dummy)
            inference_ms = (time.perf_counter() - t0) / 100 * 1000

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test_csv = os.path.join(args.log_dir, "test_summaries.csv")
    write_header = not os.path.exists(test_csv)
    with open(test_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['experiment', 'seed', 'best_val_acc', 'test_acc', 'total_train_sec', 'inference_ms', 'total_params_M', 'trainable_params_M'])
        writer.writerow([csv_filename, args.seed, best_val_acc, test_acc, 
                         f"{total_time:.1f}", f"{inference_ms:.2f}", 
                         f"{total_params/1e6:.2f}", f"{trainable_params/1e6:.2f}"])

if __name__ == "__main__":
    main()
