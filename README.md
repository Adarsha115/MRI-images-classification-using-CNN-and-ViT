# CNNs and the Transformer Architecture: From Spatial Feature Extraction to Vision Transformers

Project report and code for **Modeling Workshop (MA5770)**, Department of Mathematics, Indian Institute of Technology Madras — submitted by Group 1 (Aditya Arora, Sagnik Barman, Rumaiz Ibrahim K, Adarsha Mandal) under the guidance of Dr. Priyanka Shukla, April 2026.

This repository explores the transition from Convolutional Neural Networks (CNNs) to the Transformer architecture, culminating in a from-scratch implementation of a Vision Transformer (ViT) and a comparative study against a CNN on a real-world image classification task.

## Contents

| File | Description |
|---|---|
| `final_report_1.docx` | Full project report covering CNN fundamentals (convolution, pooling, ResNet), the Transformer architecture, and self-attention, along with the experimental comparison of CNN vs. ViT. |
| `Coding_vision_transformer_from_scratch.ipynb` | A minimal Vision Transformer built from scratch in PyTorch (patch embedding, transformer encoder block, classification head) and trained on MNIST. |
| `comparative_analysis_of_CNN_and_ViT.ipynb` | An end-to-end comparison of a custom CNN vs. a pretrained ViT (`vit_base_patch16_224` from `timm`) on a brain tumor MRI classification dataset, including training curves, confusion matrices, and a patch-swap sensitivity ("CAS") analysis. |

## 1. Coding_vision_transformer_from_scratch.ipynb

Implements a small ViT from first principles and trains it on MNIST digit classification.

**Architecture / hyperparameters:**
- Input: 28×28 grayscale MNIST images
- Patch size: 7×7 → 16 patches per image
- Embedding dimension: 16
- Attention heads: 1
- Transformer encoder blocks: 1
- Components: `PatchEmbedding` (Conv2d-based patchify + flatten), `TransformerArchitecture` (LayerNorm + Multi-Head Self-Attention + MLP block with residual connections), `VisionTransformer` (adds a learnable `[CLS]` token and positional embeddings, then a classification head)

**Pipeline:** load MNIST → patchify and embed → prepend CLS token + add positional embeddings → pass through transformer block → classify from CLS token → train for 5 epochs with Adam (lr=0.01) and cross-entropy loss → evaluate on validation set → visualize predictions on sample images.

**Requirements:** `torch`, `torchvision`, `matplotlib`, `numpy`

## 2. comparative_analysis_of_CNN_and_ViT.ipynb

Compares a custom CNN against a pretrained Vision Transformer on a 4-class brain tumor MRI dataset (uploaded as a zip archive, e.g. via Google Colab).

**Workflow:**
1. **Data preparation** — extract the MRI dataset, build train/validation/test splits with `torchvision.transforms`, and set a fixed seed for reproducibility.
2. **CNN baseline** (`BrainTumorCNN`) — a custom convolutional network (Conv2d + ReLU + MaxPool blocks followed by fully connected layers), trained for 5 epochs with Adam (lr=0.001) and cross-entropy loss.
3. **Vision Transformer** — `vit_base_patch16_224` loaded from `timm` (ImageNet-pretrained), fine-tuned for 5 epochs with Adam (lr=1e-4).
4. **Evaluation** — loss/accuracy curves, confusion matrices, and a final comparison table of test/validation accuracy for both models.
5. **Cross-Attribution Sensitivity (CAS) analysis** — a custom patch-swap technique that blends a "donor" image's patch into a "base" image to measure how much each model's prediction confidence shifts, run both as a single-image demo and aggregated across ~50 test images, for both the CNN and a ViT-attention-guided variant.

**Requirements:** `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`

## Setup

```bash
pip install torch torchvision timm numpy pandas matplotlib scikit-learn tqdm
```

`comparative_analysis_of_CNN_and_ViT.ipynb` expects the MRI dataset as a zip file (`archive.zip`) uploaded at runtime (originally via Google Colab's file upload widget) and extracts it to `/content/brain_tumor_data` with `Training/` and `Testing/` subfolders. `Coding_vision_transformer_from_scratch.ipynb` downloads MNIST automatically via `torchvision.datasets.MNIST`.

## Key Results

| Model | Test Accuracy | Best Validation Accuracy |
|---|---|---|
| CNN | 85.88% | 89.82% |
| Vision Transformer | 92.75% | 93.84% |

The pretrained ViT outperformed the custom CNN on the MRI classification task, consistent with the report's discussion of how self-attention captures long-range spatial dependencies that convolution's local receptive fields miss.

## Report Structure (`Report.pdf`)

1. Abstract
2. Introduction
3. Convolutional Neural Networks — motivation, layer types, filters/feature maps, stride/padding, pooling, fully connected layers, complete CNN architecture, ResNet
4. The Transformer Architecture — self-attention and sequence transduction
5. (Continued) Vision Transformer application and CNN vs. ViT comparative analysis
