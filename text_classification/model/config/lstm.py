import torch
from model.config.emb import Emb_conf
from dataclasses import dataclass

@dataclass
class LSTM_config:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epochs=30
    head= 'l3'

    name = "lstm_attention_aslcb.pt"

    path = f'model/weight/{name}'


    input_dim=Emb_conf().dim, 
    hidden_dim=256
    
    lr=1e-3
    thres= 0.65

    hub = name