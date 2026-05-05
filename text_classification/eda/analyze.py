import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os


import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import numpy as np
import re

def clean_tags(tags):
    if isinstance(tags, str):
        clean_s = re.sub(r'[^a-zA-Z0-9\s_]', '', tags)
        return [tag.strip() for tag in clean_s.split() if tag.strip()]    
    elif isinstance(tags, list):
        return tags
    else:
        print(f"Unexpected tags format: {tags}")
        return []
def eda(df, out_dir='result/eurlex', top = 10):
    os.makedirs(out_dir, exist_ok=True)
    df= df.copy()
    
    df['t_len'] = df['text'].apply(lambda x: len(str(x).split()))
    df['til_len'] = df['title'].apply(lambda x: len(str(x).split()))
    df['r_len'] = df['recitals'].apply(lambda x: len(str(x).split()))
    df['h_len'] = df['header'].apply(lambda x: len(str(x).split()))
    df["m_len"] = df['main_body'].apply(lambda x: len(str(x).split()))


    df['tags'] = df['eurovoc_concepts'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    all_labels = []
    df['tags'].apply(lambda x: all_labels.extend(clean_tags(x)))
    df['tags_count'] = df['tags'].apply(lambda x: len(clean_tags(x)))
    
    print("\n" + "="*20 + " DATASET STATS " + "="*20)
    print(df[['t_len','til_len','r_len','h_len','m_len','tags_count']].describe())

    label_counts = Counter(all_labels)
    unique_labels = len(label_counts)
    counts = np.array(list(label_counts.values()))

    frequent = np.sum(counts > 50)
    few_shot = np.sum((counts <= 50) & (counts >= 1))
    
    print(f"\nUnique labels found in data: {unique_labels}")
    print(f"--> Frequent labels (>50 samples): {frequent}")
    print(f"--> Few-shot labels (1-50 samples): {few_shot}")
    print(f"Top {top} labels:", label_counts.most_common(top))
    print(f"Bottom {top} labels:", label_counts.most_common()[:-top-1:-1])

    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(df['t_len'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Text Length Distribution (Words)')
    axes[0].axvline(512, color='red', linestyle='--', label='MiniLM Limit (approx)')
    axes[0].legend()

    sns.histplot(df['tags_count'], bins=range(min(df['tags_count']), max(df['tags_count']) + 2), 
                 kde=False, ax=axes[1], color='salmon')
    axes[1].set_title('Labels per Document Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/data_dist.png')
    plt.show()

    sorted_counts = sorted(counts, reverse=True)
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_counts, color='purple', linewidth=2)
    plt.fill_between(range(len(sorted_counts)), sorted_counts, color='purple', alpha=0.2)
    plt.yscale('log')
    plt.title('Label Frequency (Log Scale) - The Long Tail')
    plt.xlabel('Label Rank')
    plt.ylabel('Number of Samples')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f'{out_dir}/long_tail.png')
    plt.show()

    return label_counts, df

def check_unseen_labels(train : Counter, test : Counter, thresh = 50):
    all_test_labels = list(test.keys())
    
    analysis = []
    zero_shot = []
    fw = []
    for label in all_test_labels:
        train_freq = train.get(label, 0)
        test_freq = test.get(label, 0)
        
        if train_freq == 0:
            category = "Zero-shot"
            zero_shot.append({"Label": label, "Test_Freq": test_freq})
        elif train_freq < thresh:
            category = "Few-shot"
            fw.append({"Label": label, "Test_Freq": test_freq})

        else:
            category = "Normal"
            
        analysis.append(category)

    summary_counts = Counter(analysis)
    summary_df = pd.DataFrame([
        {"Category": f"Normal (>= {thresh})", "Count": summary_counts.get("Normal", 0)},
        {"Category": f"Few-shot (< {thresh})", "Count": summary_counts.get("Few-shot", 0)},
        {"Category": "Zero-shot (Unseen)", "Count": summary_counts.get("Zero-shot", 0)}
    ])

    zs_df = pd.DataFrame(zero_shot).sort_values(by="Test_Freq", ascending=False)
    fw_df = pd.DataFrame(fw).sort_values(by="Test_Freq", ascending=False)

    print("\n" + "="*20 + " LABEL DISTRIBUTION SUMMARY " + "="*20)
    print(summary_df.to_markdown(index=False)) 
    print("\n" + "="*20 + " ZERO-SHOT SAMPLES IN TEST " + "="*20)
    
    if not zs_df.empty:
        print(f"Total Zero-shot labels: {len(zs_df)}")
        print(zs_df.head(10).to_string(index=False))
    else:
        print("No Zero-shot labels found!")

    return summary_df, zs_df, fw_df



import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

def plot_tag(m, u_tags, target = "1402", top=30, mapper=None, out='result/eurlex/cov_matrix.png'):
    if target not in u_tags: return print(f"Tag {target} not found")
    
    v = np.sqrt(np.diag(m))
    v[v == 0] = 1e-9
    corr_m = m / np.outer(v, v)
    
    t2i = {t: i for i, t in enumerate(u_tags)}
    tid = t2i[target]
    
    row = corr_m[tid]
    idx = np.argsort(row)[::-1]
    rel_idx = idx[idx != tid][:top]
    
    sub_m = corr_m[np.ix_(rel_idx, rel_idx)].copy()
    t_sub = [u_tags[i] for i in rel_idx]
    
    for i, idx_val in enumerate(rel_idx):
        sub_m[i, i] = corr_m[idx_val, tid]
        
    plt.figure(figsize=(16, 13))
    sns.heatmap(sub_m, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1, 
                xticklabels=t_sub, yticklabels=t_sub, square=True)
    if mapper is not None:
        target = mapper[mapper["concept_id"] == target]["title"].iloc[0]
    plt.title(f'Target: "{target}" | Diagonal: Correlation with "{target}"')
    plt.savefig(out)
    plt.show()
    plt.close()

def plot_co(cov_m, u_tags, top_n=30, out='result/eurlex/co_matrix.png'):
    idx = np.argsort(np.diag(cov_m))[::-1][:top_n]
    m_sub = cov_m[np.ix_(idx, idx)]
    t_sub = [u_tags[i] for i in idx]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(m_sub, annot=False, cmap='RdBu_r', center=0, 
                xticklabels=t_sub, yticklabels=t_sub, square=True)
    plt.title(f'Top {top_n} Tag Covariance Matrix')
    
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.show()

def get_cov_matrix(df, mapper,plot = True):
    tag_lists = df['tags'].apply(clean_tags)
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(tag_lists)
    u_tags = mlb.classes_
    print(f"Unique tags: {u_tags}")
    n = X.shape[0]

    sum_x = np.array(X.sum(axis=0)).flatten()
    dot_product = (X.T @ X).toarray() 
    
    cov_m = (dot_product / (n - 1)) - np.outer(sum_x, sum_x) / (n * (n - 1))
    if plot:
        plot_tag(cov_m, u_tags, mapper = mapper)
    return cov_m, u_tags



import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain

def get_subsumption(df, threshold=0.6):
    tag_lists = df['tags'].apply(clean_tags)
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(tag_lists)
    u_tags = mlb.classes_
    
    counts = np.array(X.sum(axis=0)).flatten()
    co_matrix = (X.T @ X).toarray()
    
    prob_matrix = co_matrix / counts
    
    hierarchy = []
    for i, tag_b in enumerate(u_tags):
        for j, tag_a in enumerate(u_tags):
            if i != j and prob_matrix[j, i] >= threshold:
                if prob_matrix[i, j] < prob_matrix[j, i]:
                    hierarchy.append({'child': tag_b, 'parent': tag_a, 'conf': prob_matrix[j, i]})
                    
    return pd.DataFrame(hierarchy), co_matrix, u_tags

def get_communities(co_matrix, u_tags, min_weight=3):
    G = nx.Graph()
    for i in range(len(u_tags)):
        for j in range(i + 1, len(u_tags)):
            if co_matrix[i, j] >= min_weight:
                G.add_edge(u_tags[i], u_tags[j], weight=co_matrix[i, j])
    
    partition = community_louvain.best_partition(G, weight='weight')
    return partition

def analyze_hierarchy(df):
    sub_df, co_m, tags = get_subsumption(df)
    comm_map = get_communities(co_m, tags)
    
    res = []
    for tag in tags:
        res.append({
            'tag': tag,
            'community': comm_map.get(tag, -1),
            'is_root': tag in sub_df['parent'].values,
            'parent_candidate': sub_df[sub_df['child'] == tag]['parent'].tolist()[:3]
        })
        
    roots = sub_df.groupby('parent').size().sort_values(ascending=False)
    return pd.DataFrame(res), roots


from sklearn.cluster import AgglomerativeClustering

def get_rep_tags(m, u_tags, labels):
    unique_labels = np.unique(labels)
    rep_mapping = {}
    v = np.sqrt(np.diag(m))
    v[v == 0] = 1e-9
    corr_m = m / np.outer(v, v)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) == 0: continue
        sub_corr = corr_m[np.ix_(idx, idx)]
        centrality = sub_corr.sum(axis=1)
        best_idx = idx[np.argmax(centrality)]
        rep_mapping[label] = u_tags[best_idx]
    return rep_mapping

def analyze_clusters_hybrid(df, min_weight=1):
    sub_df, co_m, u_tags = get_subsumption(df)
    
    G = nx.Graph()
    G.add_nodes_from(u_tags)
    for i in range(len(u_tags)):
        for j in range(i + 1, len(u_tags)):
            if co_m[i, j] >= min_weight:
                G.add_edge(u_tags[i], u_tags[j], weight=co_m[i, j])
    
    partition = community_louvain.best_partition(G, weight='weight')
    
    v = np.sqrt(np.diag(co_m))
    v[v == 0] = 1e-9
    corr_m = co_m / np.outer(v, v)
    
    unique_comms = set(partition.values())
    louvain_rep_map = {}
    for c_id in unique_comms:
        idx = [i for i, t in enumerate(u_tags) if partition.get(t) == c_id]
        if not idx: continue
        sub_corr = corr_m[np.ix_(idx, idx)]
        louvain_rep_map[c_id] = u_tags[idx[np.argmax(sub_corr.sum(axis=1))]]

    cluster_counts = pd.Series(partition.values()).value_counts()
    k_opt = len(cluster_counts[cluster_counts > 1])
    if k_opt < 2: k_opt = 30
        
    dist_m = 1 - np.clip(corr_m, 0, 1)
    ac = AgglomerativeClustering(n_clusters=k_opt, metric='precomputed', linkage='average')
    ac_labels = ac.fit_predict(dist_m)
    
    ac_rep_map = get_rep_tags(co_m, u_tags, ac_labels)
    
    res_list = []
    for i, tag in enumerate(u_tags):
        l_id = partition.get(tag)
        a_id = ac_labels[i]
        res_list.append({
            'tag': tag,
            'ac_id': a_id,
            'ac_name': ac_rep_map.get(a_id),
            'louvain_id': l_id,
            'louvain_name': louvain_rep_map.get(l_id, "Isolated"),
            'is_root': tag in sub_df['parent'].values
        })
        
    res_df = pd.DataFrame(res_list)
    
    summary = res_df.groupby(['louvain_id', 'louvain_name']).size().reset_index(name='tag_count')
    summary = summary.sort_values('tag_count', ascending=False).reset_index(drop=True)
    
    return res_df, summary

import pandas as pd
import numpy as np

def run_comprehensive_audit(df, mapper):
    results = []
    l3_counts = []
    
    for _, row in df.iterrows():
        assigned = clean_tags(row['l1'])

        assigned_set = {assigned} if not isinstance(assigned, (list, set, np.ndarray)) else set(assigned)
        
        raw_tags = row['tags']
        tag_list = clean_tags(raw_tags) 
        tag_list = [t.strip() for t in tag_list if t.strip()]
        
        actual_l1_from_tags = {mapper.get(t) for t in tag_list if mapper.get(t) is not None}
        
        matches = [mapper.get(t) in assigned_set for t in tag_list if mapper.get(t) is not None]
        num_matches = sum(matches)
        total_valid_tags = len(matches)
        
        l3_counts.append(len(tag_list))
        results.append({
            'n_l3': len(tag_list),
            'n_l1_actual': len(actual_l1_from_tags),
            'is_fully_consistent': num_matches == total_valid_tags if total_valid_tags > 0 else True,
            'mismatch_count': total_valid_tags - num_matches
        })

    res_df = pd.DataFrame(results)
    l3_series = pd.Series(l3_counts)
    l1_actual_series = res_df['n_l1_actual']

    print("--- GLOBAL TAG (L3) STATISTICS ---")
    print(f"Mean L3 per Question: {l3_series.mean():.4f}")
    print(f"Std L3 per Question: {l3_series.std():.4f}")
    print(f"Min/Max L3 per Question: {l3_series.min()} / {l3_series.max()}")

    print("\n--- ACTUAL CLUSTER (L1) DISTRIBUTION (FROM TAGS) ---")
    print(f"Mean L1 per Question: {l1_actual_series.mean():.4f}")
    print(f"Std L1 per Question: {l1_actual_series.std():.4f}")
    print(f"Ratio L1 > 1 (Multi-disciplinary): {(l1_actual_series > 1).mean()*100:.2f}%")
    print(f"Ratio L1 == 0 (Unmapped): {(l1_actual_series == 0).mean()*100:.2f}%")
    print(f"Max L1 clusters in a single Question: {l1_actual_series.max()}")

    print("\n--- HIERARCHY CONSISTENCY ANALYSIS ---")
    print(f"Fully Consistent Questions: {res_df['is_fully_consistent'].mean()*100:.2f}%")
    
    total_mismatches = res_df['mismatch_count'].sum()
    total_tags_processed = l3_series.sum()
    print(f"Total Tags Mismatched with Assigned L1: {total_mismatches}")
    print(f"Tag-level Error Rate: {(total_mismatches / total_tags_processed)*100:.2f}%")

    return res_df

