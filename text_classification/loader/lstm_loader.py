from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import f1_score, hamming_loss
import torch
import numpy as np
from tqdm import tqdm
import joblib


import ast

from loader.lb_prep import get_mlb


def full_loader(df, embeddings, mlb = None):
    #  label encode
    if mlb is None:
        mlb, num_cl = get_mlb()

    if isinstance(df['filtered_tags'].iloc[0], str):
        print("Phát hiện dữ liệu dạng string, đang convert sang list...")
        tags_list = df['filtered_tags'].apply(ast.literal_eval).tolist()
    else:
        tags_list = df['filtered_tags'].tolist()
    y_l = mlb.transform(tags_list)
    y = torch.tensor(y_l, dtype=torch.float32)
    
    df = df.reset_index(drop=True)
    X = embeddings[df.index]

    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])
    train_indices, val_indices = next(stratifier.split(X=df.index.values.reshape(-1, 1), y=y_l))

    X_train = torch.tensor(embeddings[train_indices], dtype=torch.float32)
    y_train = torch.tensor(y_l[train_indices], dtype=torch.float32)

    X_val = torch.tensor(embeddings[val_indices], dtype=torch.float32)
    y_val = torch.tensor(y_l[val_indices], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

    print(f"Sync Success | Train: {len(X_train)} | Val: {len(X_val)} | Classes: {len(mlb.classes_)}")

    return train_loader, val_loader    


def check_alignment(dataset_split, df):
    uids_ds = set(dataset_split['uid'])
    uids_df = set(df['uid'])
    if uids_ds != uids_df:
        print(f"UID mismatch detected! DS UIDs: {len(uids_ds)}, DF UIDs: {len(uids_df)}")
        print(f"UIDs in DS but not in DF: {len(uids_ds - uids_df)}")
        print(f"UIDs in DF but not in DS: {len(uids_df - uids_ds)}")
    else:
        print("=====================")
        print("UIDs are perfectly aligned between dataset and DataFrame.")
        return dataset_split, df, dataset_split['uid']
    
    common_uids = uids_ds.intersection(uids_df)
    dataset_split = dataset_split.filter(lambda x: x['uid'] in common_uids)
    df = df[df['uid'].isin(common_uids)].copy()
    return check_alignment(dataset_split, df)

def prepare_loader_eurlex(split_name, dataset_dict, df, mlb, batch_size=128, shuffle=False):
    ds_split = dataset_dict[split_name]
    uids = ds_split['uid']
    ds_split, df ,uids  = check_alignment(ds_split, df)  
    print("==============================================")
    print("Prepare Loader for split:", split_name)
    x_tensor = torch.tensor(np.array(ds_split['embedding_all-MiniLM-L12-v2']), dtype=torch.float32)
    

    meta_indexed = df.set_index('uid')
    relevant_tags = meta_indexed.loc[uids]['filtered_tags']
    print("Debug", relevant_tags.head(5))
    y_tensor = torch.tensor(mlb.transform(relevant_tags), dtype=torch.float32)
    
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)