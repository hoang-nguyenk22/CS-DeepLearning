from model.trans import *
from loader.trans_loader import *

import torch.optim as optim
from sklearn.metrics import f1_score, hamming_loss
import numpy as np
from tqdm import tqdm

import torch, os, gc
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from model.config.trans import Trans_config

from loader.lb_prep import get_mlb

def init_trans(dataset="cs"):
    # Config
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset == "cs":
        mlb, num_classes = get_mlb()
    else:
        mlb, num_classes = get_mlb('data/eurlex/mlb.pktl')
    from transformers import AutoTokenizer


    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    config = Trans_config()
    config.dev = dev
    # Model Init
    model = Trans(device=dev)
    model.add_head(config.head, num_classes)
    # Loss & Optimizer
    
    return model, mlb, tokenizer,config

import torch, os, gc
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

def train_transformer(model, train_loader, val_loader, mlb, device, head_name='l3', epochs=10, criterion=None, train_with_wu=True, acc_steps=2, name="trans_lwa"):
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))
    scaler = GradScaler()
    best_f1 = 0.0
    warm_up = max(int(epochs*3/4)+1, 0) if train_with_wu else 0
    print("\n=====================================================================================================================")
    print(f"--- Starting Transformer Training on {device} | Warm-up Epochs: {warm_up} | Finetune Epochs: {epochs - warm_up} ---")

    optimizer = AdamW([
        {'params': model.transformer.parameters(), 'lr': 0.0}, # Mặc định là 0
        {'params': model.heads[head_name].parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)

    for epoch in range(epochs):
        if epoch < warm_up and train_with_wu:
            model.transformer.requires_grad_(False)
            lr_backbone = 0.0
            lr_head = 1e-3
            desc = f"Warmup {epoch+1}/{epochs}"
        else:
            model.transformer.requires_grad_(True)
            lr_backbone = 1e-6
            lr_head = 5e-4
            desc = f"Fine-tune {epoch+1}/{epochs}"

        optimizer.param_groups[0]['lr'] = lr_backbone
        optimizer.param_groups[1]['lr'] = lr_head

        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=desc)
        
        for i, batch in enumerate(pbar):
            batch = [b.to(device) for b in batch]
            inputs_list = [
                (batch[j].long(), batch[j+1].long()) 
                for j in range(0, len(batch) - 1, 2)
            ]
            y = batch[-1].float()

            # Scale back from F32 --> F16 to reduce Mem cost
            with autocast(device_type='cuda'): # We avoid auto cast in LSTM since gradients can be overhead with F16, but for transformer it's more stable with less epochs
                outputs = model(inputs_list, active_head=head_name)[head_name]
                loss = criterion(outputs, y) / acc_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % acc_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({'loss': f"{loss.item() * acc_steps:.4f}"})
            del inputs_list, y, outputs, loss

        cur_t, tab, curr_f1 = best_thres(model, val_loader, mlb, device, active_head=head_name)
        print(f"================== Epoch {epoch + 1} | t = {cur_t} ==================")
        print(tab)
        
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            os.makedirs("model/weight/eurlex", exist_ok=True)
            save_path = f"model/weight/eurlex/{name}_{head_name}_best.pt"
            torch.save({
                'model_state': model.state_dict(),
                'best_threshold': cur_t,
                'macro_f1': best_f1
            }, save_path)
            print(f"--> Saved: {save_path} (F1: {best_f1:.4f})")
        else:
            print(f"====> F1: {curr_f1:.4f} (Best: {best_f1:.4f})")
            
        torch.cuda.empty_cache()
        gc.collect()

    return model