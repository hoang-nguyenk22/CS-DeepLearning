import torch
import torch.nn as nn
import numpy as np
import time
from transformers import AutoTokenizer


from model.config.lstm import lstm_cs_conf, lstm_eurlex_conf
from model.config.trans import trans_cs_conf, trans_eurlex_conf


from model.config.emb import emb_eur, emb_cs
from loader.lb_prep import get_mlb
from model.lstm import BiLSTM
from model.trans import Trans
from model.embedding import EmbeddingExtractor

import joblib
from huggingface_hub import hf_hub_download
from torch.amp import autocast
from eda.preprocess import clean, clean_legal

from typing import Dict, Union

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

def get_resource_path(relative_path):
    return os.path.join(project_root, relative_path)

    return os.path.join(base_path, relative_path)
class InferenceEngine:
    def __init__(self, device=None, dataset= "eurlex", local=False):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = dataset
      
        if local:
            path = get_resource_path(f'data/{dataset}/mlb.{"pktl" if dataset == "eurlex" else "pkl"}')
            self.mlb , self.num_classes= get_mlb(path)
            if dataset != "cs":
                self.mlb_lstm, self.n_lstm = get_mlb(get_resource_path(f'data/{dataset}/mlb_lstm.pkl'))
            else:
                self.mlb_lstm, self.n_lstm = self.mlb, self.num_classes
        else:
            if dataset == "cs":
                self.mlb = hf_hub_download(repo_id="TungDKS/XMC", filename="mlb.pktl")
                self.mlb = joblib.load(self.mlb)
                self.num_classes = len(self.mlb.classes_)
                self.mlb_lstm = self.mlb
            else:
                self.mlb = hf_hub_download(repo_id="TungDKS/XMC", filename="mlb.pktl")
                self.mlb = joblib.load(self.mlb)
                self.num_classes = len(self.mlb.classes_)
                self.mlb_lstm = hf_hub_download(repo_id="TungDKS/XMC", filename="mlb_lstm.pkl")
                self.mlb_lstm = joblib.load(self.mlb_lstm)
                self.n_lstm = len(self.mlb_lstm.classes_)
        
        self.trans_conf = trans_eurlex_conf if dataset == "eurlex" else trans_cs_conf
        self.lstm_conf = lstm_cs_conf if dataset == "cs" else lstm_eurlex_conf
        self.emb_conf = emb_cs if dataset == "cs" else emb_eur 
        
        self.prep_func = clean if dataset == "cs" else clean_legal 
        self.lstm_to_trans_map = self._build_label_mapping()

        self.tokenizer = AutoTokenizer.from_pretrained(self.emb_conf.model_name)
        self.extractor = EmbeddingExtractor(device=self.device)
        
        self.model_lstm, self.lstm_thres = self._init_lstm()
        self.model_trans, self.trans_thres = self._init_trans()

    def _build_label_mapping(self):
        trans_class_to_idx = {cls: i for i, cls in enumerate(self.mlb.classes_)}
        mapping = np.zeros(self.n_lstm, dtype=int)
        for i, cls in enumerate(self.mlb_lstm.classes_):
            mapping[i] = trans_class_to_idx.get(cls, 0)
        return mapping

    def _init_lstm(self):
        model = BiLSTM(input_dim=self.emb_conf.dim, hidden_dim=self.lstm_conf.hidden_dim)
        model.add_head('l3',self.n_lstm )
        try:
            weight_path = get_resource_path(self.lstm_conf.path)
        except:
            weight_path = hf_hub_download(repo_id="TungDKS/XMC", filename=self.lstm_conf.name)
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint)
        thres = checkpoint.get('best_threshold', self.lstm_conf.thres) if isinstance(checkpoint, dict) else self.lstm_conf.thres
        return model.to(self.device).eval(), thres

    def _init_trans(self):
        model = Trans(model_name=self.emb_conf.model_name, device=self.device)
        model.add_head(name='l3', head_type=self.trans_conf.typ, num_labels=self.num_classes)
        try:
            weight_path = get_resource_path(self.trans_conf.path)
        except:
            weight_path = hf_hub_download(repo_id="TungDKS/XMC", filename=self.trans_conf.name)
        thres = model.load_checkpoint(weight_path)
        return model.eval(), thres

    def _prepare_trans_input(self, data: Union[Dict, str]):
        if isinstance(data, str):
            c1 = data
            c2 = ""
        else:
            c1 = f"{data.get('title', '')} [SEP] {data.get('main_body', '')}"
            c2 = data.get('recitals', None)

        enc1 = self.tokenizer(c1, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.device)
        enc2 = self.tokenizer(c2, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.device) if c2 else None
        
        inp =  [
            (enc1['input_ids'], enc1['attention_mask'])            
        ]
        if enc2:
            inp.append((enc2['input_ids'], enc1['attention_mask']))
        return inp

    def predict(self, data, model_type='trans', thres=None):
        start_time = time.time()
        
        if model_type == 'lstm':
            t = thres if thres is not None else self.lstm_thres
            current_mlb = self.mlb_lstm
            current_num_classes = self.n_lstm
        else:
            t = thres if thres is not None else self.trans_thres
            current_mlb = self.mlb
            current_num_classes = self.num_classes

        data['title'] = self.prep_func(data.get('title', ''))
        data['main_body'] = self.prep_func(data.get('main_body', ''))
        data['recitals'] = self.prep_func(data.get('recitals', '')) if self.dataset == "eurlex" else None
        full_text = f"{data.get('title', '')} {data.get('main_body', '')}{data.get('recitals', '')}"


        print(f"Title: {data.get('title', '')}")
        print(f"Body: {data.get('main_body', '')}")
        with torch.no_grad():
            if model_type == 'lstm':
                
                emb = self.extractor.get_embeddings(full_text)
                emb = emb.unsqueeze(0).unsqueeze(0) if emb.dim() == 1 else emb.unsqueeze(1)
                print(full_text)
                out = self.model_lstm(emb.float().to(self.device))
            else:
                trans_input = self._prepare_trans_input(data)
                with autocast(device_type='cuda'):
                    out = self.model_trans(trans_input, active_head='l3')
            
            if isinstance(out, dict):
                logits = out.get('l3', out.get('logits', next(iter(out.values()))))
            else:
                logits = out
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        indices = np.where(probs >= t)[0]
        prediction_mask = np.zeros((1, current_num_classes))
        prediction_mask[0, indices] = 1
        labels = current_mlb.inverse_transform(prediction_mask)[0]
        
        return {
            "labels": list(labels),
            "confidence": probs[indices].tolist(),
            "probs_raw": probs,
            "inference_time": f"{time.time() - start_time:.4f}s",
            "attention": out.get('attention') if isinstance(out, dict) else None,
            "model_used": model_type
        }

    def ensemble_predict(self, data, w_trans=0.7, threshold=0.3):
        start_time = time.time()
        res_trans = self.predict(data, model_type='trans', thres=0)
        res_lstm = self.predict(data, model_type='lstm', thres=0)
        
        aligned_lstm_probs = np.zeros(self.num_classes)
        aligned_lstm_probs[self.lstm_to_trans_map] = res_lstm['probs_raw']
        
        combined_probs = (w_trans * res_trans['probs_raw']) + ((1 - w_trans) * aligned_lstm_probs)
        
        indices = np.where(combined_probs >= threshold)[0]
        prediction_mask = np.zeros((1, self.num_classes))
        prediction_mask[0, indices] = 1
        labels = self.mlb.inverse_transform(prediction_mask)[0]

        return {
            "labels": list(labels),
            "confidence": combined_probs[indices].tolist(),
            "inference_time": f"{time.time() - start_time:.4f}s",
            "probs_raw": combined_probs
        }
    def visualize_attention(self, data, model_choice='trans'):
        if model_choice == 'trans':
            trans_input = self._prepare_trans_input(data)
            with torch.no_grad():
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                with autocast(device_type=device_type):
                    out = self.model_trans(trans_input, active_head='l3')
            
            attn_weights = out.get('attention')
            if attn_weights is None: return None

            text = f"{data.get('title', '')} [SEP] {data.get('main_body', '')}"
            inputs = self.tokenizer(text, truncation=True, max_length=512)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'])

            weights = attn_weights.cpu().numpy()[0]
            return [{"token": t.replace(' ', ''), "weight": float(weights[i])} 
                    for i, t in enumerate(tokens) if i < len(weights)]

        elif model_choice == 'lstm':
            full_text = f"{data.get('title', '')} {data.get('main_body', '')} {data.get('recitals', '')}"
            emb = self.extractor.get_embeddings(full_text)
            emb = emb.unsqueeze(0).unsqueeze(1) if emb.dim() == 1 else emb.unsqueeze(1)
            
            with torch.no_grad():
                out = self.model_lstm(emb.float().to(self.device))
            
            attn_weights = out.get('attention_weights')
            if attn_weights is None: return None

            tokens = full_text.split()[:512]
            weights = attn_weights.cpu().numpy().flatten()
            return [{"token": t, "weight": float(weights[i])} 
                    for i, t in enumerate(tokens) if i < len(weights)]
        return None