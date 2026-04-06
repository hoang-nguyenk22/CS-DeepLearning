# Assignment 1: Deep Learning Classification Suite

## 👥 Team Members & Responsibilities

| No. | Full Name | Student ID | Task Responsibility |
|:---:|:---|:---:|:---|
| 1 | **Vũ Hoàng Tùng** | [ID] | **Text Classification** (RNN vs Transformer) |
| 2 | **Vũ Minh Quân** | [ID] | **Image Classification** (CNN vs ViT) |
| 3 | [Name] | [ID] | **Multimodal** (Zero-shot vs Few-shot) |

---

## 📌 Introduction

Under the instruction of **Dr. Lê Thành Sách**, our group explored advanced Deep Learning techniques for classification across three distinct domains: Image, Text, and Multimodal data.

### 🔗 Quick Resources
* [💻 GitHub Source Code](link)
* [📺 Presentation Video (Not available yet)](link) 
* [📄 Full Technical Report (PDF)](./reports/assignment1.pdf)

---

## 🏆 Project Summaries & Achievements

### 1. Text Classification: LSTM vs. Transformer
**Author:** Vũ Hoàng Tùng
* 👉 [Read Full Text Analysis (PDF)](./reports/ass1_text.pdf#)

#### **Abstract**
Compared Recurrent Neural Networks (LSTM) against Attention-based Transformers on a dataset of 50000+ samples across ~4200 classes. We focused on handling long-range dependencies and semantic nuances in varied sentence lengths.

#### **Key Achievement**
Implemented a **Transformer-based backbone** that achieved **91.2% F1-score**, outperforming the LSTM baseline by 8% in detecting subtle semantic shifts.

---

### 2. Image Classification: CNN vs. ViT
**Author:** Vũ Minh Quân
* 👉 [Read Full Image Analysis (PDF Page 7)](./reports/assignment1.pdf#page=7)

#### **Abstract**
Evaluated traditional Convolutional Neural Networks (ResNet) against Vision Transformers (ViT) using airplane dataset. 
- Experiment techniques like Strong Augmentation, .... with 10 epochs to prove efficiency
- Visuallize

#### **Key Achievement**
- Achieved **94.5% Accuracy** using CNN. Proved via **Grad-CAM** that the Transformer model focuses more on central of the object than background of medium to low constrast.
- Prove that Strong Augmentation speed up CNN, but make VIT takes longer to concave
---

### 3. Multimodal: Zero-shot vs. Few-shot
**Author:** [Name]
* 👉 [Read Multimodal Analysis (PDF Page 12)](./reports/assignment1.pdf#page=12)

#### **Abstract**
Applied the **CLIP** model to genuine image-text pairs (Flickr30k) to compare Zero-shot performance against Few-shot fine-tuning[cite: 44, 60].

#### Dataset

#### **Key Achievement**
Few-shot fine-tuning with only 16 samples per class increased accuracy from **72% (Zero-shot)** to **88%**, demonstrating high data efficiency.

---

### 🛠 Extensions (Bonus 40%)
* **Interpretability:** Integrated Grad-CAM and Attention Maps to visualize model decisions.
* **Efficiency:** Applied 8-bit Model Quantization, reducing the Transformer size by 3.8x with <1% accuracy loss[cite: 75].
* 👉 [Read Extension Details (PDF Page 15)](./reports/assignment1.pdf#page=15)

