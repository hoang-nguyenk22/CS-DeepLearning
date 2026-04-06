import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

from model.config.emb import Emb_conf
import os
class EmbeddingExtractor:
    def __init__(self, device=None):
        self.conf = Emb_conf()
        model_name = self.conf.model_name
        self.dim = self.conf.dim
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        
        

    def extract(self, df, batch_size=64, save_path=None):
        texts = df['text'].tolist()
        if len(texts) < 5000:
            print(f"Warning: Dataset size ({len(texts)}) is below the assignment requirement.")
        else:
            print(f"Extracting embeddings for {len(texts)} samples...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(embeddings.cpu(), save_path)
            print(f"Embeddings saved to {save_path}. Dimension: {self.dim}")
            
        return embeddings

    @staticmethod
    def load_embeddings(path = 'data/cs_embeddings.pt'):
        if os.path.exists(path):
            return torch.load(path)
        raise FileNotFoundError(f"No embeddings found at {path}")

    def get_embeddings(self, text):
        return self.model.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
