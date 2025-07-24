# UltraScanNet: A Mamba-Inspired Hybrid Backbone for Breast Ultrasound Classification

> This repository accompanies the scientific article:  
> **UltraScanNet: A Mamba-Inspired Hybrid Backbone for Breast Ultrasound Classification**  
> 🖊 Alexandra-Gabriela Laicu-Hausberger, Călin-Adrian Popa  
> 📍 Politehnica University of Timișoara, Romania  
---

## 🧠 About the Paper

Breast ultrasound imaging offers a safe, accessible, and non-invasive alternative for early breast cancer detection. However, its interpretation is hindered by low contrast, speckle noise, and inter-class variability.

**UltraScanNet** is a novel deep learning architecture specifically designed to address these challenges. Inspired by Mamba-style state-space modeling, the network blends:

- A **convolutional stem** with learnable 2D positional embeddings,
- A **hybrid Stage 1**, combining convolutional and MobileViT-style blocks with spatial gating,
- **Depth-adaptive Stage 2 and 3**, leveraging a mixture of:
  - 🌀 `UltraScanUnit` — a custom selective-scan SSM module with gated convolutions and low-rank residuals,
  - 🔄 `ConvAttnMixers` — lightweight convolutional attention blocks,
  - 🧠 `Multi-head self-attention` — for global context modeling.

The model achieves state-of-the-art performance on the BUSI dataset with **91.67% top-1 accuracy**, **0.9096 F1-score**, and competitive precision/recall, outperforming or matching models such as ViT-Small, MaxViT-Tiny, and ConvNeXt-Tiny.

---

## 📈 Key Results

| Model              | Top-1 Acc | Precision | Recall | F1-Score |
|-------------------|-----------|-----------|--------|----------|
| **UltraScanNet**   | **91.67%** | **0.9072** | **0.9174** | **0.9096** |
| ViT-Small          | 91.67%    | —         | —      | —        |
| MaxViT-Tiny        | 91.67%    | —         | —      | —        |
| Swin-Tiny          | 90.38%    | —         | —      | —        |
| ConvNeXt-Tiny      | 89.74%    | —         | —      | —        |
| ResNet-50          | 85.90%    | —         | —      | —        |

---

## 🚧 Repository Status

📢 **Code release coming soon**  
This repository is currently under active development. The code, dataset preprocessing scripts, and trained model weights will be made publicly available after the paper's peer review process.

Stay tuned for:
- ✅ PyTorch implementation of UltraScanNet  
- ✅ Dataset preparation instructions for BUSI and BUSBRA  
- ✅ Training & evaluation scripts  
- ✅ Inference demo 




