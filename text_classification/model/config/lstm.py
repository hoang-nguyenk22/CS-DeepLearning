import torch
from model.config.emb import Emb_conf
from dataclasses import dataclass, field


@dataclass
class LSTM_config:
    dataset: str = "eurlex" # "eurlex" hoặc "cs"
    dev: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    epochs: int = 30
    head: str = 'l3'
    hidden_dim: int = 256
    lr: float = 1e-3
    thres: float = 0.65
    
    name: str = field(init=False)
    path: str = field(init=False)
    hub: str = field(init=False)

    def __post_init__(self):
  
        if self.dataset == "eurlex":
            self.name = "lstm_attention_aslcb.pt"
            self.thres = 0.65
        else:
            self.name = "lstm_attention.pt" 
            self.thres = 0.45 
            
        self.path = f'model/weight/{self.dataset}/{self.name}'
        self.hub = self.name

lstm_eurlex_conf = LSTM_config(dataset="eurlex")
lstm_cs_conf = LSTM_config(dataset="cs")