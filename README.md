# Image Classification: FGVC-Aircraft Benchmark

This directory contains a strictly controlled benchmarking suite for CNN (ResNet-50) and ViT (ViT-B/16) architectures on the FGVC-Aircraft dataset.

## Setup
1.  **Environment**: Install dependencies: `torch`, `torchvision`, `timm`, `umap-learn`, `squarify`, `seaborn`, `opencv-python`.
2.  **Data Preparation**:
    - **Automatic**: Run any script (e.g., `python3 eda.py`). The `torchvision` library will automatically download and extract the FGVC-Aircraft dataset to the specified `--data_dir` (default: `./data`).
    - **Manual**: If automatic download fails, download the dataset from the [official VGG page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and extract it. Ensure the final structure is:
      ```
      data/fgvc-aircraft-2013b/A
          data/
              images/
              variants.txt
              ...
      ```

## Component Overview
- **`dataset.py`**: Data loaders with multi-level augmentation (Light vs. RandAugment).
- **`models.py`**: Architecture definitions and Layer-wise Learning Rate Decay (LLRD).
- **`train.py`**: Core training engine with tracking, early stopping, and efficiency metrics.
- **`eda.py`**: Dataset exploration (class distribution, dimensions, UMAP feature projection).
- **`error_analysis.py`**: Confusion matrix, misclassification grids, and per-class accuracy.
- **`gradcam.py`**: Interpretability using Grad-CAM (CNNs) and Attention Rollout (ViTs).
- **`plot_results.py`**: Generates comparative learning curves and final accuracy charts from logs.

## Execution
- **Run EDA**: `python3 eda.py`
- **Orchestrate Grid Search**: `bash orchestrate_experiments.sh` (Runs 30 experiments across strategies/seeds).
- **Architecture Showdown**: `bash fair_arch_compare.sh` (Controlled LR and long-budget comparison).
- **Analyze Errors**: `python3 error_analysis.py --model <model> --checkpoint <path>`

All outputs and logs are saved to `logs/`, `checkpoints/` respectively.
