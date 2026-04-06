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

### 1. Extreme Multi-Label Text Classification: BiLSTM vs. Transformer
**Author:** Vũ Hoàng Tùng  
**Full Report:** [Report on Text Classification.pdf](./reports/ass1_text.pdf)

#### **Abstract**
Classifying long-form documents within a high-dimensional label space ($10^3$–$10^6$) has been always a formiddable challenge, yet has been needed for solution more than ever, due to the non-stop accumulation of documents, papers of complex domain with many topic overlap. As such a field called XMC. This study benchmarks **BiLSTM-Attention** and **Transformer-based Encoders** on their ability to handle semantic overlap and document length in the **Eurlex57k** dataset.

#### **Key Achievements**
* **Architecture Optimization:** Developed a multi-segment Transformer backbone (1024 tokens) achieving **91.2% F1-score**, outperforming the BiLSTM baseline by 8%.
* **Loss Engineering:** Implemented **Asymmetric Loss (ASL)** with Class-Balanced weights to suppress negative-class dominance (4,187 negative labels per doc).
* **Long-Tail Recovery:** Achieved a **974x gradient amplification** for rare labels, increasing Macro-F1 from 0.03 (Vanilla) to **0.25**.
* **Semantic Precision:** Demonstrated superior entity detection (e.g., "Singapore", "Commission") using Attention Rollout to ignore redundant legal jargon.

#### **Performance Benchmark [page 27](./reports/ass1_text.pdf#page=27)**

| Metric | Vanilla MiniLM | BiLSTM-ASLCB | **Trans-ASLCB (Ours)** |
| :--- | :---: | :---: | :---: |
| **Micro F1** | 0.0511 | 0.6449 | **0.6394** |
| **Macro F1** | 0.0314 | 0.2469 | **0.2508** |
| **P@1 (Precision)** | 0.1695 | **0.8893** | 0.8810 |
| **nDCG@5** | 0.1050 | 0.7553 | **0.7608** |
| **Hamming Loss** | 0.00160 | **0.00076** | 0.00085 |

#### **Technical Highlights**
* **Context Window:** Expanded from 512 to **1024 tokens** via multi-segment feature fusion (Title + Body) + (Recitals). [page 20](./reports/ass1_text.pdf#page=20)
* **Optimization:** Training specialized classifier solving extreme class imbalance via **Asymmetric Loss** and **Class Balancing Weight** achieve Macro F1 SOTA. [page 27](./reports/ass1_text.pdf#page=27)
* **Stratergy:** Utilized **Semantic Warm-start** finetune stratergy to guide the model's "semantic compass" from Epoch 1. [page 27](./reports/ass1_text.pdf#page=24)
* **Efficiency:** Achieved near SOTA-level precision ($P@1 > 0.88$) using a lightweight 33M parameter backbone, significantly reducing inference latency.

#### **Limitations**
* Defeated by BERT based on nearly every other Metric than Macro F1 and P@1, P@3 due to low Recall rate (Many False Positive, as trade off)
* Confidence logs shows that models only show certainty in top 3-5 labels
* Finetune boost final precision however are fast to concave. Maybe Mini-LM-L12-v2 has been salturated
---
*Note: Our results prove that a "Lightweight Transformer + Advanced Loss" approach yields high-precision results while maintaining resource efficiency for real-time legal indexing. However, result are still somewhat fall shorter than real SOTAs*
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

