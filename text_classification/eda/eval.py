import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, hamming_loss, ndcg_score, f1_score
from tabulate import tabulate
import os
import pandas as pd

def precision_at_k(y_true, y_score, k=5):
    order = np.argsort(y_score, axis=1)[:, ::-1]
    top_k = order[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1)
    return np.mean(np.sum(hits, axis=1) / k)

def recall_at_k(y_true, y_score, k=5):
    order = np.argsort(y_score, axis=1)[:, ::-1]
    top_k = order[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1)
    relevant = np.sum(y_true, axis=1)
    recall = np.sum(hits, axis=1) / np.maximum(relevant, 1)
    return np.mean(recall)

def eval(model, loader, mlb, device, train_counter, test_counter, thres=0.3, active_head='l3', fs_thresh=10, filename="result/report.csv"):
    model.eval()
    all_scores, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if len(batch) == 3:
                ids, y = batch[0].to(device).float(), batch[2].to(device).float()
            else:
                ids, y = batch[0].to(device).float(), batch[1].to(device).float()
            
            out = model(ids)
            outputs = out['l3'] if isinstance(out, dict) else out
            all_scores.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    y_score = np.vstack(all_scores)
    y_true = np.vstack(all_targets)
    y_pred = (y_score > thres).astype(int)
    
    pk = {f"P@{k}": precision_at_k(y_true, y_score, k=k) for k in [1, 3, 5]}
    rk = {f"R@{k}": recall_at_k(y_true, y_score, k=k) for k in [1, 3, 5]}
    ndcg_5 = ndcg_score(y_true, y_score, k=5)
    
    report_dict = classification_report(y_true, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)
    
    name2idx = {name: i for i, name in enumerate(mlb.classes_)}
    cat_metrics = []
    label_groups = {"Zero-shot": [], "Few-shot": [], "Normal": []}
    
    for label in test_counter.keys():
        train_freq = train_counter.get(label, 0)
        label_str = str(label)
        if label_str not in name2idx: continue
        
        if train_freq == 0:
            label_groups["Zero-shot"].append(label_str)
        elif train_freq < fs_thresh:
            label_groups["Few-shot"].append(label_str)
        else:
            label_groups["Normal"].append(label_str)

    for group_name, labels in label_groups.items():
        if not labels:
            cat_metrics.append([group_name, 0, 0, 0, 0, 0, 0, 0])
            continue
            
        idxs = [name2idx[l] for l in labels]
        group_true = y_true[:, idxs]
        group_pred = y_pred[:, idxs]
        
        tp = np.sum((group_true == 1) & (group_pred == 1))
        fp = np.sum((group_true == 0) & (group_pred == 1))
        fn = np.sum((group_true == 1) & (group_pred == 0))
        
        micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
        
        macro_p = np.mean([report_dict[l]['precision'] for l in labels])
        macro_r = np.mean([report_dict[l]['recall'] for l in labels])
        macro_f1 = np.mean([report_dict[l]['f1-score'] for l in labels])
        support = np.sum(group_true)
        
        cat_metrics.append([
            group_name, 
            f"{macro_f1:.4f}", f"{micro_f1:.4f}", 
            f"{macro_p:.4f}", f"{micro_p:.4f}", 
            f"{macro_r:.4f}", f"{micro_r:.4f}", 
            int(support)
        ])

    metrics_to_show = ['micro avg', 'macro avg', 'weighted avg']
    global_data = [[m.upper()] + [report_dict[m][k] for k in ['precision', 'recall', 'f1-score', 'support']] for m in metrics_to_show]
    basic_tab = tabulate(global_data, headers=['GLOBAL', 'PREC', 'RECALL', 'F1', 'SUPPORT'], tablefmt='fancy_grid')
    
    xmc_data = [
        ["P@1 / P@3 / P@5", f"{pk['P@1']:.4f} / {pk['P@3']:.4f} / {pk['P@5']:.4f}"],
        ["R@1 / R@3 / R@5", f"{rk['R@1']:.4f} / {rk['R@3']:.4f} / {rk['R@5']:.4f}"],
        ["nDCG@5", f"{ndcg_5:.4f}"],
        ["Hamming Loss", f"{hamming_loss(y_true, y_pred):.5f}"]
    ]
    extra_tab = tabulate(xmc_data, headers=['XMC METRIC', 'VALUE'], tablefmt='fancy_grid')
    
    cat_headers = ['CATEGORY', 'MA-F1', 'MI-F1', 'MA-PREC', 'MI-PREC', 'MA-REC', 'MI-REC', 'SUPPORT']
    cat_tab = tabulate(cat_metrics, headers=cat_headers, tablefmt='fancy_grid')

    df_report = pd.DataFrame(report_dict).transpose()
    for k, v in {**pk, **rk, "ndcg@5": ndcg_5}.items():
        df_report.loc[k] = [v] * 3 + [y_true.shape[0]]
        
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_report.to_csv(filename, index=True, encoding='utf-8-sig')
    
    return df_report, basic_tab, extra_tab, cat_tab

import gc 
from sklearn.metrics import classification_report, f1_score
from tabulate import tabulate


def best_thres(model, val_loader, mlb, device, active_head='l3', max_t=0.85):
    model.eval()
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            inputs = [
                (batch[j].to(device).long(), batch[j+1].to(device).long()) 
                for j in range(0, len(batch) - 1, 2)
            ]
            y = batch[-1].numpy()
            
            with autocast(device_type='cuda'):
                outputs = model(inputs, active_head=active_head)[active_head]
            all_probs.append(torch.sigmoid(outputs).cpu().numpy().astype(np.float32))
            all_targets.append(y.astype(np.uint8))
            
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    
    del all_probs, all_targets
    gc.collect() 
    
    best_t, best_f1 = 0.3, 0
    for t in np.round(np.arange(0.1, max_t + 0.05, 0.05), 2):
        y_pred = (y_probs > t).astype(np.uint8)
        ma_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if ma_f1 > best_f1:
            best_f1, best_t = ma_f1, t
        
        del y_pred
        
    gc.collect() 
    
    report = classification_report(y_true, (y_probs > best_t).astype(np.uint8), 
                                   target_names=mlb.classes_, output_dict=True, zero_division=0)
    
    metrics = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
    table_data = [[m.upper()] + [report[m][k] for k in ['precision', 'recall', 'f1-score', 'support']] for m in metrics]
    basic_tab = tabulate(table_data, headers=['GLOBAL METRIC', 'PRECISION', 'RECALL', 'F1-SCORE', 'SUPPORT'], tablefmt='fancy_grid')
    
    return best_t, basic_tab, best_f1