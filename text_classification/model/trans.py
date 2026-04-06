import torch
import torch.nn as nn
from transformers import AutoModel

from typing import List, Tuple


from model.attention import LabelWiseAttention, GlobalAttention, MaxHead

class Trans(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L12-v2', device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dim = self.transformer.config.hidden_size
        self.heads = nn.ModuleDict()
        self.to(self.device)

    def add_head(self, name, head_type, num_labels):
        if head_type == 'lwa':
            self.heads[name] = LabelWiseAttention(self.dim, num_labels)
        elif head_type == 'global':
            self.heads[name] = GlobalAttention(self.dim, num_labels)
        elif head_type == 'max':
            self.heads[name] = MaxHead(self.dim, num_labels)
        else:
            raise ValueError(f"Head type {head_type} is not supported!")
        self.heads[name].to(self.device)

    def forward(self, inputs: List[Tuple], active_head='l3'):
        outs = []
        for ids, mask in inputs:
            out = self.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
            outs.append(out)

        lhs = torch.cat(outs, dim=1)
        
        if active_head == 'all':
            return {name: head(lhs) for name, head in self.heads.items()}
        
        return {active_head: self.heads[active_head](lhs)}

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.load_state_dict(checkpoint['model_state'])
            thres = checkpoint.get('best_threshold', 0.3)
            f1 = checkpoint.get('f1', checkpoint.get('macro_f1', 0.0))
            print(f"--> [SUCCESS] Loaded Model. Best Threshold: {thres} | Best F1: {f1:.4f}")
            return thres 
        else:
            self.load_state_dict(checkpoint)
            print(f"--> [SUCCESS] Loaded Raw State Dict. Threshold defaults to 0.3")
            return 0.3
        
    @torch.no_grad()
    def semantic_init(self, mlb, tokenizer, head_name='l3',id_to_name=None, normalize = 0.05, init_bias=-2.0,batch_size=64):

        if id_to_name is None:
                label = pd.read_csv('data/eurlex/train_labels.csv')
                id_to_name = dict(zip(label['concept_id'], label['title']))

        label_ids = [str(c) for c in mlb.classes_]
        label_names = [id_to_name.get(cid, cid) for cid in label_ids]

        print(f"--> Convert {len(label_names)} label to text. Eg: {label_names[:3]}")
        self.eval()
        all_label_feats = []
        
        pbar = tqdm(range(0, len(label_names), batch_size), desc=f"Semantic Init [{head_name}]")
        for i in pbar:
            batch = label_names[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(self.device)
            
            outputs = self.transformer(**inputs)
            batch_feats = outputs.last_hidden_state[:, 0, :] 
            all_label_feats.append(batch_feats)
            
        semantic_weights = torch.cat(all_label_feats, dim=0) 
        semantic_weights = torch.nn.functional.normalize(semantic_weights, p=2, dim=1)
        
        target_head = self.heads[head_name]
        found = False
        for module in target_head.modules():
            if isinstance(module, nn.Linear):
                if module.weight.shape == semantic_weights.shape:
                    module.weight.data = semantic_weights * normalize# Scale down to prevent large initial loss, slow start 
                    module.bias.data.fill_(init_bias) # Slow start 
                    found = True
                    break
        
        if found:
            print(f"--> [SUCCESS] {head_name} init label vector done. Shape: {semantic_weights.shape}")
        else:
            print(f"--> [ERROR]  {head_name}")