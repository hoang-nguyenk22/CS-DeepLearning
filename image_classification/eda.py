"""
Exploratory Data Analysis (EDA) for the FGVC-Aircraft Dataset.

This script performs a comprehensive analysis of the FGVC-Aircraft dataset, 
including label distribution, image statistics, visual audits, and 
feature space visualization using UMAP.

The outputs are designed to match the analysis presented in the project report.
"""

import os
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import timm
from torchvision import transforms
from umap import UMAP
import squarify
import matplotlib.colors as mcolors

# ==========================================
# Configuration
# ==========================================
DATA_ROOT = './data/fgvc-aircraft-2013b/data'
OUTPUT_DIR = 'eda_reports'
SEED = 42
SUBSET_SIZE_UMAP = 3000  # Number of images to use for UMAP feature extraction

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def setup_environment():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

def load_dataset_metadata():
    """
    Load the FGVC-Aircraft hierarchy (Variant, Family, Manufacturer).
    Returns a unified DataFrame.
    """
    print("Reading dataset metadata...")
    
    def parse_meta_file(filename):
        metadata = {}
        file_path = os.path.join(DATA_ROOT, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            for line in f:
                # The format is 'IMAGE_ID LABEL'
                # Use split(' ', 1) because label can contain spaces
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]
        return metadata

    # We use the trainval split for the overview as it contains most labels
    try:
        variants = parse_meta_file('images_variant_trainval.txt')
        families = parse_meta_file('images_family_trainval.txt')
        manufacturers = parse_meta_file('images_manufacturer_trainval.txt')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    image_ids = list(variants.keys())
    df = pd.DataFrame({
        'id': image_ids,
        'variant': [variants[i] for i in image_ids],
        'family': [families[i] for i in image_ids],
        'manufacturer': [manufacturers[i] for i in image_ids]
    })
    
    # Map full image paths
    df['path'] = df['id'].apply(lambda x: os.path.join(DATA_ROOT, 'images', f"{x}.jpg"))
    
    print(f"Loaded {len(df)} images across {df['variant'].nunique()} variants.")
    return df

def plot_label_statistics(df):
    """
    Generate visualizations for label distribution:
    1. Treemap of all 100 variants.
    2. Bar chart of samples per manufacturer.
    """
    print("Generating label distribution plots...")
    
    # --- 1. Variant Treemap ---
    counts = df['variant'].value_counts()
    unique_counts = sorted(counts.unique()) # Usually 66 and 67
    n_colors = len(unique_counts)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use Viridis extremes for a premium look (Dark vs Light)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_colors))
    count_to_color = {cnt: color for cnt, color in zip(unique_counts, colors)}
    plot_colors = [count_to_color[cnt] for cnt in counts.values]
    
    squarify.plot(sizes=counts.values, label=counts.index, alpha=0.8, 
                  text_kwargs={'fontsize': 8, 'color': 'white', 'weight': 'bold'}, 
                  color=plot_colors, ax=ax)
    
    plt.title("Sample Distribution: 100 Aircraft Variants", fontsize=18, pad=20)
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_variant_treemap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # --- 2. Manufacturer Bar Chart ---
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    manufacturer_counts = df['manufacturer'].value_counts()
    
    sns.barplot(x=manufacturer_counts.values, y=manufacturer_counts.index, palette="magma", hue=manufacturer_counts.index, legend=False)
    plt.title("Sample Distribution by Manufacturer", fontsize=16)
    plt.xlabel("Number of Samples")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_manufacturer_dist.png'), dpi=150)
    plt.close()

def plot_image_dimension_analysis(df):
    """Analyze and plot height, width, and aspect ratio distributions."""
    print("Analyzing image dimensions (sampling 1000 images)...")
    widths, heights = [], []
    
    # Sampling for speed
    sample_df = df.sample(min(1000, len(df)))
    for p in tqdm(sample_df['path'], desc="Processing images"):
        with Image.open(p) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            
    widths, heights = np.array(widths), np.array(heights)
    ratios = widths / heights
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Width
    sns.histplot(widths, ax=axes[0], color='RoyalBlue', kde=True)
    axes[0].set_title("Width Distribution")
    
    # Height
    sns.histplot(heights, ax=axes[1], color='IndianRed', kde=True)
    axes[1].set_title("Height Distribution")
    
    # Aspect Ratio
    sns.histplot(ratios, ax=axes[2], color='SeaGreen', kde=True)
    axes[2].set_title("Aspect Ratio Distribution")
    
    plt.suptitle("Metadata Analysis: Image Dimensions", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_dimensions.png'), dpi=150)
    plt.close()

def generate_visual_audit(df, n_classes=5, n_samples=5):
    """Create a grid of sample images from diverse classes."""
    print("Generating visual audit grid...")
    selected_variants = random.sample(list(df['variant'].unique()), n_classes)
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(15, 3 * n_classes))
    
    for i, variant in enumerate(selected_variants):
        samples = df[df['variant'] == variant].sample(n_samples)
        for j, (_, row) in enumerate(samples.iterrows()):
            img = Image.open(row['path']).convert('RGB')
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(variant, fontsize=12, fontweight='bold')
                # To show y-label we need visibility
                axes[i, j].axis('on')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                axes[i, j].spines['bottom'].set_visible(False)
                axes[i, j].spines['left'].set_visible(False)
                axes[i, j].set_title(variant, loc='left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_visual_audit.png'), dpi=150)
    plt.close()

def run_feature_space_visualization(df):
    """
    Use a pre-trained ResNet50 to extract features and visualize 
    the feature space using UMAP.
    """
    print(f"Running UMAP on {SUBSET_SIZE_UMAP} samples...")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Feature Extractor
    # Using timm's resnet50 with no classification head (global pooling)
    model = timm.create_model('resnet50.tv_in1k', pretrained=True, num_classes=0)
    model = model.to(device).eval()
    
    # Standard ImageNet transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 2. Extract Features
    subset_df = df.sample(min(SUBSET_SIZE_UMAP, len(df)))
    features_list = []
    
    with torch.no_grad():
        for path in tqdm(subset_df['path'], desc="Extracting visual features"):
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                feat = model(tensor).cpu().numpy().flatten()
                features_list.append(feat)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
    features = np.array(features_list)
    
    # 3. Dimensionality Reduction
    print("Computing UMAP projection...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    embedding = reducer.fit_transform(features)
    
    subset_df['umap_1'] = embedding[:, 0]
    subset_df['umap_2'] = embedding[:, 1]
    
    # 4. Plot UMAP (Colored by Manufacturer)
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=subset_df, x='umap_1', y='umap_2', 
                    hue='manufacturer', palette='Spectral', alpha=0.6, s=25)
    
    plt.title("UMAP Projection of ResNet50 Latent Features", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Manufacturer")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_umap_features.png'), dpi=200, bbox_inches='tight')
    plt.close()

def main():
    """Main execution block."""
    print("FGVC-AIRCRAFT EXPLORATORY DATA ANALYSIS")
    
    setup_environment()
    
    df = load_dataset_metadata()
    if df is None:
        print("Initialization failed. Please check your data directory.")
        return
        
    plot_label_statistics(df)
    plot_image_dimension_analysis(df)
    generate_visual_audit(df)
    
    # UMAP is computationally expensive; skip if only metadata analysis is needed.
    try:
        run_feature_space_visualization(df)
    except Exception as e:
        print(f"UMAP visualization failed: {e}")
    
    print(f"EDA completed. Results saved in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
