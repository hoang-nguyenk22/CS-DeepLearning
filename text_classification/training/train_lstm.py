from model.lstm import *
from loader.lstm_loader import *

import torch.optim as optim
from sklearn.metrics import f1_score, hamming_loss
import numpy as np
from tqdm import tqdm

from model.config.lstm import LSTM_config

def init_lstm(src = "model/weight/lstm_attention.pt"):
    # Config
    config = LSTM_config() # please fix in config file
    mlb, num_classes = get_mlb()

    dev = config.dev


    # Model Init
    if src is None:
        model = BiLSTM(input_dim=config.input_dim, hidden_dim=config.hidden_dim).add_head(config.head, num_classes)
    else:
        model = BiLSTM.load_model(
            path=src, 
            heads_config={config.head:num_classes }, 
            device=dev,      
            input_dim=config.input_dim, 
            hidden_dim=config.hidden_dim     
        )
            
    model.to(dev)
    
    # Loss & Optimizer
    opt = optim.AdamW(model.parameters(), lr= config.lr , weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()


    return model, mlb, opt, crit, config

import torch.optim as optim
from sklearn.metrics import f1_score, hamming_loss
import numpy as np
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc="Training Batch", leave=True)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # Forward & Backward
        outputs = model(x, head_names=['l3'])['l3']
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        preds = (torch.sigmoid(outputs) > 0.5).cpu().detach().numpy()
        all_preds.append(preds)
        all_targets.append(y.cpu().detach().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return {
        'loss': epoch_loss / len(loader),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'hamming': hamming_loss(all_targets, all_preds)
    }

def full_training(model, loader, optimizer, criterion, config : LSTM_config ): # a training function with simplified status tracking
    print(f"--- Starting Training on {device} ---")

    device =  config.dev
    path = config.path
    epochs=config.epochs

    for epoch in range(epochs):
        metrics = train_epoch(model, loader, optimizer, criterion, device)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Macro-F1: {metrics['f1']:.4f} | "
              f"Hamming: {metrics['hamming']:.4f}")
    

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


