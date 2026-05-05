
import torch
from torch.utils.data import TensorDataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import f1_score, hamming_loss
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd
import ast
from transformers import AutoTokenizer
from eda.analyze import clean_tags
from loader.lb_prep import get_mlb


def prepare_transformer_loader(df, mlb, tokenizer, batch_size=16, shuffle=False):

    if mlb is None:
        mlb, num_cl = get_mlb()
    if isinstance(df, str):
        df = pd.read_csv(df)
   
    tags_list = df['filtered_tags'].apply(clean_tags)
    y = torch.tensor(mlb.transform(tags_list), dtype=torch.float32)
    
    encoded = tokenizer(
        df['text'].fillna("").tolist(), # Xử lý trường hợp text bị rỗng
        padding='max_length', 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    
    dataset = TensorDataset(
        encoded['input_ids'], 
        encoded['attention_mask'], 
        y
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split_and_tokenize(df, tokenizer, max_len=512):
        chunk1_texts = []
        chunk2_texts = []
        
        for _, row in df.iterrows():
            c1 = f"{row['title']} [SEP] {row['main_body']}" # Chunk 1 = title + main_body
            c2 = f"{row['recitals']}" # Chunk 2 = recitals only (often contains key legal points)
            chunk1_texts.append(c1)
            chunk2_texts.append(c2)
        
        enc1 = tokenizer(chunk1_texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        enc2 = tokenizer(chunk2_texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        
        return enc1, enc2


def map_ids_to_titles(tags_list, mapping_dict):
        return [mapping_dict.get(str(t), str(t)) for t in tags_list]

def trans_loader_eur(model_name = 'sentence-transformers/all-MiniLM-L12-v2', path = "data/eurlex"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv(f'{path}/train_processed.csv')
    test_df = pd.read_csv(f'{path}/test_processed.csv')
    label = pd.read_csv(f'{path}/train_labels.csv')

    id_to_title = dict(zip(label['concept_id'].astype(str), label['title']))

    



    mlb, num_cl = get_mlb('data/eurlex/mlb.pktl')

    y_train_raw = mlb.transform(train_df['filtered_tags'].apply(clean_tags))
    y_test_raw = mlb.transform(test_df['filtered_tags'].apply(clean_tags))

    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32)

    train_enc1, train_enc2 = split_and_tokenize(train_df, tokenizer)
    test_enc1, test_enc2 = split_and_tokenize(test_df, tokenizer)

    train_ds = TensorDataset(
        train_enc1['input_ids'], train_enc1['attention_mask'],
        train_enc2['input_ids'], train_enc2['attention_mask'],
        y_train
    )

    test_ds = TensorDataset(
        test_enc1['input_ids'], test_enc1['attention_mask'],
        test_enc2['input_ids'], test_enc2['attention_mask'],
        y_test
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"Ready: Train {len(train_ds)} | Test {len(test_ds)}")
    print(f"Labels: {len(mlb.classes_)}")

    return train_loader, test_loader, mlb, tokenizer, id_to_title