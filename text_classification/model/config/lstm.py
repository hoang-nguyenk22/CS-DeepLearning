import torch
from model.config.emb import Emb_conf
class LSTM_config:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'model/weight/lstm_attention_aslcb.pt'
    epochs=30
    head= 'l3'

    input_dim=Emb_conf().dim, 
    hidden_dim=256
    
    lr=1e-3
    thres= 0.65