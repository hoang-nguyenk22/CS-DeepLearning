import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import ndcg_score, classification_report, f1_score
from tabulate import tabulate
import gc

def find_best_thres(y_true, y_score, start=0.2, end=0.95, step=0.01):
    best_t = 0
    best_avg_f1 = 0
    history = []

    y_true = y_true.astype(np.uint8)
    
    print(f"--> Searching best threshold from {start} to {end}...")
    
    for t in np.arange(start, end, step):
        t = round(t, 3)
        y_pred = (y_score > t).astype(np.uint8)
        
        mi_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        ma_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        avg_f1 = (mi_f1 + ma_f1) / 2
        
        history.append([t, mi_f1, ma_f1, avg_f1])
        
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_t = t
            
        del y_pred
        gc.collect()

    headers = ["Threshold", "Micro F1", "Macro F1", "AVG F1"]
    print("\n" + tabulate(history, headers=headers, tablefmt='fancy_grid', floatfmt=".4f"))
    
    print(f"\n==> Best Threshold: {best_t} | Best AVG F1: {best_avg_f1:.4f}")
    return best_t


@torch.no_grad()
def base_model_eval(model, loader, mlb, tokenizer, device, id_to_name=None, filename="result/zero_shot_base.csv"):
    model.eval()
    model.to(device)
    
    if id_to_name is None:
        label_df = pd.read_csv('data/eurlex/train_labels.csv')
        id_to_name = dict(zip(label_df['concept_id'].astype(str), label_df['title']))

    label_names = [id_to_name.get(str(c), str(c)) for c in mlb.classes_]
    
    print(f"--> Encoding {len(label_names)} label names...")
    all_label_feats = []
    for i in range(0, len(label_names), 64):
        batch_names = label_names[i:i+64]
        inputs = tokenizer(batch_names, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        outputs = model.transformer(**inputs)
        all_label_feats.append(outputs.last_hidden_state[:, 0, :])
            
    label_embeddings = torch.cat(all_label_feats, dim=0)
    label_embeddings = F.normalize(label_embeddings, p=2, dim=1) # Chuẩn hóa L2

    all_scores, all_targets = [], []

    for batch in tqdm(loader, desc="Base Similarity Inference"):
        y_true = batch[-1].numpy()
        
        input_ids = batch[0].to(device).long()
        attention_mask = batch[1].to(device).long()

        outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        similarity = torch.matmul(text_embeddings, label_embeddings.t())
        
        all_scores.append(similarity.cpu().numpy())
        all_targets.append(y_true)

    y_score = np.vstack(all_scores)
    y_true = np.vstack(all_targets)

    t = find_best_thres(y_true, y_score, start=0.2, end=0.9, step=0.01)
    y_pred = (y_score > t).astype(np.uint8)

    

    def get_pk(y_t, y_s, k):
        order = np.argsort(y_s, axis=1)[:, ::-1][:, :k]
        hits = np.take_along_axis(y_t, order, axis=1)
        return np.mean(np.sum(hits, axis=1) / k)

    pk = {f"P@{k}": get_pk(y_true, y_score, k) for k in [1, 3, 5]}
    ndcg_5 = ndcg_score(y_true, y_score, k=5)
    
    report = classification_report(y_true, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)
    
    metrics_to_show = ['micro avg', 'macro avg', 'weighted avg']
    global_data = [[m.upper()] + [report[m][k] for k in ['precision', 'recall', 'f1-score', 'support']] for m in metrics_to_show]
    
    xmc_data = [
        ["P@1 / P@3 / P@5", f"{pk['P@1']:.4f} / {pk['P@3']:.4f} / {pk['P@5']:.4f}"],
        ["nDCG@5", f"{ndcg_5:.4f}"]
    ]

    print("\n" + tabulate(global_data, headers=['GLOBAL', 'PREC', 'RECALL', 'F1', 'SUPPORT'], tablefmt='fancy_grid'))
    print("\n" + tabulate(xmc_data, headers=['XMC METRIC', 'VALUE'], tablefmt='fancy_grid'))

    if filename:
        pd.DataFrame(report).transpose().to_csv(filename)

    return y_score