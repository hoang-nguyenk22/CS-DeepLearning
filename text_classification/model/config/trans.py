import torch

import torch
from model.config.emb import Emb_conf
from dataclasses import dataclass



class Trans_config:
    def __init__(self, dataset="eurlex"):
        self.dataset = dataset
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.head = "l3"
        self.loss = "aslcb"
        self.epochs = 30
        
        if dataset == "eurlex":
            self.typ = "max"
            self.name = f"trans_{self.loss}{'_lwa' if self.typ == 'lwa' else ''}_{self.head}_best.pt"
            self.thres = 0.55
        else:
            self.typ = "lwa"
            self.name = f"trans_lwa_{self.head}_best.pt"
            self.thres = 0.65
            
        self.path = f'model/weight/{dataset}/{self.name}'

# --- TẠO INSTANCE ---
trans_eurlex_conf = Trans_config(dataset="eurlex", )
trans_cs_conf = Trans_config(dataset="cs")