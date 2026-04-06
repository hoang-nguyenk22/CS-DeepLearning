from typing import Union
import os
import re

from eda.analyze import *

def filter_tags(df, ratio=0.001, filter=None):
    N = len(df)
    if ratio:
        threshold = int(N * ratio)
        
        all_tags = [t.strip() for tags in df['tags'] for t in (tags if isinstance(tags, list) else str(tags).split('|'))]
        tag_counts = pd.Series(all_tags).value_counts()
        valid_tags = set(tag_counts[tag_counts >= threshold].index)
        print(f"--- FLAT BASELINE SETUP (Ratio: {ratio}) ---")
        print(f"Calculated Threshold: {threshold} samples")
        print(f"Unique Tags Remaining: {len(valid_tags)} (Dropped {len(tag_counts) - len(valid_tags)})")
        print(f"Samples Maintained: {len(df)} / {N}")
    elif filter is not None:    
        valid_tags = filter
        
    def clean_func(tags):
            t_list = tags if isinstance(tags, list) else str(tags).split('|')
            return [t.strip() for t in t_list if t.strip() in valid_tags]
    
    df['filtered_tags'] = df['tags'].apply(clean_func)
    

    df = df[df['filtered_tags'].map(len) > 0].copy()
    
    
    
    return df, sorted(list(valid_tags))


def clean(text):
    text = re.sub(r'<code>(.*?)</code>', r' [CODE_START] \1 [CODE_END] ', str(text), flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()
import re
import unicodedata
import ast
import pandas as pd

def clean_legal(text):

    if not text or pd.isna(text):
        return ""
    # String list from recitals or main_body
    if isinstance(text, str) and text.startswith('['):
        try:
            items = ast.literal_eval(text)
            text = " ".join(items) if isinstance(items, list) else str(items)
        except (ValueError, SyntaxError):
            pass
    elif isinstance(text, list):
        text = " ".join(text)

    text = str(text)
    
    # Strip HTML tags (If yes)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 3. Unicode Normalization (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # 4. Remove Footnotes [1], [2] or (1), (2)

    text = re.sub(r'\[\s*\d+\s*\]', ' ', text)
    text = re.sub(r'\(\s*\d+\s*\)', ' ', text)
    
    # 5. Normalize special punctuation
    text = text.replace('–', '-').replace('—', '-').replace('“', '"').replace('”', '"')
    

    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lowercase 
    return text.lower()

from typing import Union
import os
import re

def filter_tags(df, ratio=0.001, filter=None):
    N = len(df)
    if ratio:
        threshold = int(N * ratio)
        
        all_tags = [t.strip() for tags in df['tags'] for t in (tags if isinstance(tags, list) else str(tags).split('|'))]
        tag_counts = pd.Series(all_tags).value_counts()
        valid_tags = set(tag_counts[tag_counts >= threshold].index)
        print(f"--- FLAT BASELINE SETUP (Ratio: {ratio}) ---")
        print(f"Calculated Threshold: {threshold} samples")
        print(f"Unique Tags Remaining: {len(valid_tags)} (Dropped {len(tag_counts) - len(valid_tags)})")
        print(f"Samples Maintained: {len(df)} / {N}")
    elif filter is not None:    
        valid_tags = filter
    else:
        df['filtered_tags'] = df['tags']
        all_tags = [t.strip() for tags in df['tags'] for t in (tags if isinstance(tags, list) else str(tags).split('|'))]
        tag_counts = pd.Series(all_tags).value_counts()
        valid_tags = set(tag_counts[tag_counts > 0].index)
        
        return df, sorted(list(valid_tags))
        
    def clean_func(tags):
            t_list = tags if isinstance(tags, list) else str(tags).split('|')
            return [t.strip() for t in t_list if t.strip() in valid_tags]
    
    df['filtered_tags'] = df['tags'].apply(clean_func)
    

    df = df[df['filtered_tags'].map(len) > 0].copy()
    
    
    
    return df, sorted(list(valid_tags))

def get_comm(df):
    sub_df, co_m, u_tags = get_subsumption(df)
    partition = get_communities(co_m, u_tags, min_weight=5)
    
    v = np.sqrt(np.diag(co_m))
    v[v == 0] = 1e-9
    corr_m = co_m / np.outer(v, v)
    
    unique_ids = set(partition.values())
    rep_names = {}
    for c_id in unique_ids:
        indices = [i for i, t in enumerate(u_tags) if partition.get(t) == c_id]
        if not indices: continue
        sub_corr = corr_m[np.ix_(indices, indices)]
        centrality = sub_corr.sum(axis=1)
        rep_names[c_id] = u_tags[indices[np.argmax(centrality)]]
        
    return partition, rep_names
def get_title(mapper, concept_id):
    title = mapper[mapper["concept_id"] == concept_id]["title"]
    if not title.empty:
            return title
    print(f"Warning: Concept ID {concept_id} not found in mapper.")
    return concept_id

def prep(src, dst=None, mapper=None, ratio=0.001, filter = None):
    if isinstance(src, str):
        df = pd.read_csv(src)
    else:
        df = src.copy()

    df['text'] = df['text'].apply(clean_legal)
    df['main_body'] = df['main_body'].apply(clean_legal)
    df['title'] = df['title'].apply(clean_legal)
    df['recitals'] = df['recitals'].apply(clean_legal)

    if mapper is not None:
        df['tag_names'] = df['tags'].apply(lambda x: get_title(mapper, x))

    if ratio:
        df, val_tags = filter_tags(df, ratio=ratio)
    else:
        df, val_tags = filter_tags(df, ratio=None, filter = filter)
    
    # Get Community Mapping
    mapping, names = get_comm(df)
    
    def map_to_multi_l1(tags):
        l1_ids = {mapping.get(t) for t in tags if mapping.get(t) is not None}
        return sorted(list(l1_ids))

    df['l1'] = df['filtered_tags'].apply(map_to_multi_l1)
    
    df['l1_name'] = df['l1'].apply(lambda ids: [names.get(i) for i in ids])

    if dst:
        if os.path.dirname(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        df.to_csv(dst, index=False)
        
    return df, mapping, names, val_tags