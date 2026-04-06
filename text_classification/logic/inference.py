import torch
import torch.nn as nn
import numpy as np
import time
from transformers import AutoTokenizer
from model.config.lstm import LSTM_config
from model.config.trans import Trans_config
from model.config.emb import Emb_conf
from loader.lb_prep import get_mlb
from model.lstm import BiLSTM
from model.trans import Trans
from model.embedding import EmbeddingExtractor

from torch.amp import autocast
from eda.preprocess import clean, clean_legal

from typing import Dict, Union
class InferenceEngine:
    def __init__(self, device=None, dataset= "eurlex"):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dataset == "cs":
             self.mlb, self.num_classes = get_mlb()
             self.mlb_lstm = self.mlb
        else:
            self.mlb, self.num_classes = get_mlb("data/eurlex/mlb.pktl")
            self.mlb_lstm, self.n_lstm = get_mlb("data/eurlex/mlb_lstm.pkl")
        
        self.trans_conf = Trans_config()
        self.lstm_conf = LSTM_config()
        self.emb_conf = Emb_conf()
        

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
        checkpoint = torch.load(self.lstm_conf.path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint)
        thres = checkpoint.get('best_threshold', self.lstm_conf.thres) if isinstance(checkpoint, dict) else self.lstm_conf.thres
        return model.to(self.device).eval(), thres

    def _init_trans(self):
        model = Trans(model_name=self.emb_conf.model_name, device=self.device)
        model.add_head(name='l3', head_type=self.trans_conf.typ, num_labels=self.num_classes)
        thres = model.load_checkpoint(self.trans_conf.path)
        return model.eval(), thres

    def _prepare_trans_input(self, data: Union[Dict, str]):
        if isinstance(data, str):
            c1 = data
            c2 = ""
        else:
            c1 = f"{data.get('title', '')} [SEP] {data.get('main_body', '')}"
            c2 = f"{data.get('recitals', '')}"

        enc1 = self.tokenizer(c1, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.device)
        enc2 = self.tokenizer(c2, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.device) 
        
        return [
            (enc1['input_ids'], enc1['attention_mask']),
            (enc2['input_ids'], enc2['attention_mask']) 
        ]

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

        data['title'] = clean_legal(data.get('title', ''))
        data['main_body'] = clean_legal(data.get('main_body', ''))
        data['recitals'] = clean_legal(data.get('recitals', ''))

        with torch.no_grad():
            if model_type == 'lstm':
                full_text = f"{data.get('title', '')} {data.get('main_body', '')} {data.get('recitals', '')}"
                emb = self.extractor.get_embeddings(full_text)
                emb = emb.unsqueeze(0).unsqueeze(0) if emb.dim() == 1 else emb.unsqueeze(1)
                out = self.model_lstm(emb.float().to(self.device))
            else:
                trans_input = self._prepare_trans_input(data)
                with autocast(device_type='cuda'):
                    out = self.model_trans(trans_input, active_head='l3')
            
            logits = out['l3'] if isinstance(out, dict) else out
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
            "model_used": model_type
        }

    def ensemble_predict(self, data, w_trans=0.7, threshold=0.3):
        start_time = time.time()
        res_trans = self.predict(data, model_type='trans', override_thres=0)
        res_lstm = self.predict(data, model_type='lstm', override_thres=0)
        
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