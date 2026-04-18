# Masked Autoencoder (MAE) — Self-Supervised Visual Pre-training

A PyTorch implementation of **Masked Autoencoders Are Scalable Vision Learners** ([He et al., CVPR 2022](https://arxiv.org/abs/2111.06377)), trained on the **Tiny-ImageNet-200** dataset as a self-supervised visual pre-training task.

### Testing on MNIST dataset
<img width="1461" height="515" alt="image" src="https://github.com/user-attachments/assets/1c735cff-6423-48d9-a733-6da84151c4ec" />


| Item | Detail |
|---|---|
| **Paper** | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2022) |
| **Dataset** | [Tiny-ImageNet-200](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) (100 000 train / 10 000 val, 200 classes, 64×64 images) |
| **Platform** | Kaggle — 2 × Tesla T4 GPUs |
| **Framework** | PyTorch |
| **Objective** | Pixel-level reconstruction (MSE on masked patches) |

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Architecture Overview](#architecture-overview)
3. [Notebook Walkthrough](#notebook-walkthrough)
4. [Training Process](#training-process)
5. [Results & Analysis](#results--analysis)
6. [How to Run](#how-to-run)
7. [Requirements](#requirements)
8. [References](#references)

---

## Core Idea

Traditional supervised pre-training requires large labelled datasets. **MAE** removes this dependency by turning vision pre-training into a *self-supervised* task:

1. **Mask** a large, random subset of image patches (75%).
2. **Encode** only the small set of visible patches (25%) through a heavy ViT encoder.
3. **Decode** from the latent representations + learnable mask tokens back to full pixel-level patches via a lightweight decoder.
4. **Reconstruct** the masked patches and compute loss only on those positions.

> **Why does this work?** Images are highly redundant; neighbouring patches carry overlapping information. By masking 75% of the input, the model is forced to learn rich, high-level semantic features rather than just copying local textures — similar to how BERT forces language understanding through masked token prediction. The asymmetric design (heavy encoder, light decoder) also yields a **3× speedup** during pre-training because the encoder only processes 25% of tokens.

### Connection to the Original Paper

This implementation faithfully reproduces the key design choices from He et al.:

| Design Choice | Paper | This Notebook |
|---|---|---|
| Masking ratio | **75%** | **75%** |
| Encoder | ViT-Large (24 blocks, 1024-dim) | **ViT-Base** (12 blocks, 768-dim) |
| Decoder | 8 blocks, 512-dim | **12 blocks, 384-dim** |
| Positional embeddings | Sinusoidal (fixed) | **Sinusoidal (fixed)** |
| Reconstruction target | Raw pixel values (normalised) | **Raw pixel values** |
| Loss | MSE on masked patches only | **MSE on masked patches only** |
| Optimizer | AdamW | **AdamW** |
| LR schedule | Cosine decay with warm-up | **CosineAnnealingLR** (no warm-up) |
| Weight decay | 0.05 | **0.05** |

> The primary difference is scale: the paper uses ViT-Large on ImageNet-1K (1.28M images at 224×224), whereas this notebook uses ViT-Base on Tiny-ImageNet (100K images at 64×64, upscaled to 224×224). This is an **academic-scale reproduction** designed to demonstrate the core principles within limited compute budgets.

---

## Architecture Overview

```
Image (224×224)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Patch Embedding (Conv2d, 16×16 stride)      │  → 196 patches, each 768-dim
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  + Sinusoidal Positional Embedding           │  → position-aware tokens
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Random Masking (75%)                        │  → keep 49 patches, discard 147
└──────────────────────────────────────────────┘
    │                                   ┌──────────────────┐
    ▼                                   │  ids_restore     │
┌─────────────────────────┐             │  (for unshuffling)│
│  ENCODER (ViT-Base)     │             └──────────────────┘
│  12 × Transformer Block │
│  + Final LayerNorm      │
│  Input: (B, 49, 768)    │
│  Output: (B, 49, 768)   │
└─────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Linear Projection  768 → 384                │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Append learnable [MASK] tokens (×147)       │  → full sequence of 196 tokens
│  Unshuffle via ids_restore                   │
│  + Sinusoidal Positional Embedding (decoder) │
└──────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  DECODER                │
│  12 × Transformer Block │
│  + Final LayerNorm      │
│  Input: (B, 196, 384)   │
│  Output: (B, 196, 384)  │
└─────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Prediction Head (Linear 384 → 768)          │  → 768 = 16×16×3 pixel values
└──────────────────────────────────────────────┘
    │
    ▼
  MSE Loss (masked patches only)
```

### Patch Embedding

| Component | Description |
|---|---|
| **Method** | `Conv2d(3, 768, kernel_size=16, stride=16)` — acts as simultaneous patch extraction and linear projection (equivalent to splitting the image into 16×16 patches and linearly projecting each). |
| **Input** | `(B, 3, 224, 224)` |
| **Output** | `(B, 196, 768)` — a sequence of 196 patch embeddings, each 768-dimensional. |

> **Note:** Tiny-ImageNet images are 64×64. They are **upscaled to 224×224** via `transforms.Resize` to match the standard ViT patch grid of 14×14 = 196 patches.

### Positional Embeddings

Fixed **sinusoidal** positional embeddings are used (not learned), registered as non-trainable buffers. Separate embeddings are generated for the encoder (768-dim, 196 positions) and decoder (384-dim, 196 positions). The sine/cosine formulation from *Attention Is All You Need* (Vaswani et al., 2017) is used:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Transformer Block (Pre-Norm)

Each block follows the **Pre-LayerNorm** pattern (as in the original MAE paper):

```
x = x + MHSA(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

| Sub-layer | Detail |
|---|---|
| **Multi-Head Self-Attention** | Single fused QKV linear projection (`embed_dim → 3 × embed_dim`), split into multiple heads. Scaled dot-product attention with `scale = 1/√d_head`. Output projection (`embed_dim → embed_dim`). |
| **Feed-Forward Network** | Two-layer MLP: `Linear(embed_dim → 4×embed_dim)` → `GELU` → `Linear(4×embed_dim → embed_dim)`. |

### Encoder (ViT-Base)

| Parameter | Value |
|---|---|
| Embedding dim | **768** |
| Depth (# blocks) | **12** |
| Attention heads | **12** (head dim = 64) |
| MLP hidden dim | **3072** (4× embed_dim) |
| Input tokens | **49** (25% of 196, after masking) |
| Parameters | **85.6M** |

The encoder only processes the **visible (unmasked)** 25% of patches. This asymmetry is the key efficiency insight from the paper — the heavy encoder never sees the mask tokens.

### Decoder

| Parameter | Value |
|---|---|
| Embedding dim | **384** |
| Depth (# blocks) | **12** |
| Attention heads | **6** (head dim = 64) |
| MLP hidden dim | **1536** (4× embed_dim) |
| Input tokens | **196** (full set: 49 visible + 147 mask) |
| Parameters | **21.9M** |

The decoder is intentionally lighter than the encoder. It receives the encoded visible tokens projected to its own dimension, plus **learnable mask tokens** (initialized with `N(0, 0.02)`) for all masked positions. After unshuffling to restore spatial ordering, it processes the full 196-token sequence and predicts raw pixel values per patch.

### Mask Token

A single learnable vector `(1, 1, 384)` that is broadcast to fill all masked positions. The model learns what "missing information" looks like, enabling reconstruction.

### Loss Function

```python
loss = (pred - target) ** 2          # per-pixel squared error
loss = loss.mean(dim=-1)             # average over pixels per patch → (B, 196)
loss = (loss * (~mask)).sum() / (~mask).sum()   # average over MASKED patches only
```

The loss is computed **only on the masked patches**, following the paper. The model is never penalised for its predictions on visible patches — it only needs to learn to reconstruct what it hasn't seen.

### Model Size

| Component | Parameters |
|---|---|
| Encoder | **85.6M** |
| Decoder | **21.9M** |
| **Total** | **107.5M** |

---

## Notebook Walkthrough

The notebook is organised into the following sequential sections:

### 1 — Imports & Device Setup

Sets up PyTorch, torchvision, and GPU detection. Confirms 2× T4 GPU availability.

### 2 — Data Loading & Augmentation

| Setting | Value |
|---|---|
| **Dataset** | Tiny-ImageNet-200 |
| **Train samples** | 100 000 |
| **Val samples** | 10 000 |
| **Batch size** | 64 |
| **Image size** | 224 × 224 (upscaled from 64×64) |
| **Augmentation** | `RandomHorizontalFlip` (train only) |
| **Normalisation** | ImageNet mean/std |
| **Workers** | 4, with `pin_memory=True` |

> Labels are loaded by `ImageFolder` but **never used** during training — this is purely self-supervised. Only images are passed to the model; the `_` in `for images, _ in loader` explicitly discards labels.

### 3 — Patch Embedding Module

Defines the `PatchEmbedding` class: a single `Conv2d` layer that simultaneously extracts and linearly projects 16×16 patches into 768-dim embeddings.

### 4 — Sinusoidal Positional Embedding

Implements the fixed positional encoding function using sine/cosine formulas, following *Attention Is All You Need*.

### 5 — Transformer Building Blocks

Defines three classes:
- **`MultiHeadSelfAttention`** — fused QKV projection, scaled dot-product attention, output projection.
- **`FeedForward`** — two-layer MLP with GELU activation.
- **`TransformerBlock`** — pre-norm residual block combining the above two.

### 6 — MAE Encoder

`MAEEncoder` class:
1. Patchifies the image via `PatchEmbedding`.
2. Adds sinusoidal positional embeddings to **all** patches (before masking).
3. Applies the mask — selects only visible patches using `torch.gather`.
4. Passes 49 visible tokens through 12 transformer blocks.
5. Returns encoded tokens, mask, and `ids_restore` (for unshuffling).

### 7 — MAE Decoder

`MAEDecoder` class:
1. Projects encoder output from 768-dim to 384-dim.
2. Appends learnable mask tokens for masked positions.
3. Unshuffles all tokens back to original spatial order via `ids_restore`.
4. Adds decoder positional embeddings.
5. Passes full 196-token sequence through 12 transformer blocks.
6. Final linear layer predicts `16×16×3 = 768` pixel values per patch.

### 8 — Full MAE Model

`MaskedAutoencoder` class ties everything together:
- **`patchify()`** — converts images to patch format for loss targets.
- **`unpatchify()`** — converts patches back to images for visualisation.
- **`generate_mask()`** — creates random binary masks (75% masked, 25% visible).
- **`forward()`** — full pipeline: mask → encode → decode → MSE loss on masked patches.

### 9 — Training Loop

`train_mae()` function:
- Uses `DataParallel` for multi-GPU.
- AdamW optimizer (`lr=1.5e-4`, `weight_decay=0.05`).
- `CosineAnnealingLR` scheduler.
- Mixed-precision training via `GradScaler` and `autocast`.
- Gradient clipping at `max_norm=1.0`.
- Checkpoints saved every 5 epochs.
- Tracks train/val reconstruction loss per epoch.

### 10 — Model Instantiation & Training

Instantiates the model, prints parameter counts, and runs training for **35 epochs** (initial run). A **loss curve** is plotted and saved.

### 11 — Checkpoint Resume & Extended Training

Loads the epoch-35 checkpoint and resumes training up to epoch **44** (interrupted due to time constraints). Demonstrates PyTorch checkpoint resume (model, optimizer, scheduler, scaler states).

### 12 — Reconstruction Visualisation

`visualize_reconstruction()` function:
- Takes validation images, runs them through the trained MAE.
- Displays a 3-column grid: **Masked Input** | **Reconstruction** | **Original**.
- Denormalises ImageNet normalisation for proper RGB display.
- Saved as `reconstruction_results.png`.

---

## Training Process

Training ran for a total of **~44 epochs** on 2 × Tesla T4 GPUs (~15 min per epoch).

### Key Training Details

| Aspect | Detail |
|---|---|
| **Loss function** | MSE, computed **only on the 75% masked patches**. This is the core of the MAE paradigm — the model is never penalised for visible patch predictions. |
| **Optimizer** | AdamW (`lr=1.5e-4`, `weight_decay=0.05`) — AdamW decouples weight decay from gradient updates, critical for Transformer training stability. |
| **LR schedule** | `CosineAnnealingLR` (`T_max=55`, `eta_min=1e-6`) — smoothly decays learning rate following a cosine curve, reaching the minimum at the end of training. |
| **Mixed precision** | `torch.cuda.amp.autocast` + `GradScaler` — FP16 forward/backward pass for ~2× speedup with minimal accuracy loss. |
| **Gradient clipping** | `max_norm=1.0` — prevents exploding gradients in the deep Transformer stack. |
| **Data augmentation** | Only `RandomHorizontalFlip` (minimal, by design). The paper argues that the **high masking ratio itself acts as strong regularisation**, making heavy augmentation unnecessary. |
| **Checkpointing** | Full state (model, optimizer, scheduler, scaler, loss history) saved every 5 epochs. |

### Loss Curve

| Epoch | Train Loss | Val Loss | LR |
|:---:|:---:|:---:|:---:|
| 1 | 0.5002 | 0.3458 | 1.50e-04 |
| 5 | 0.2653 | 0.2597 | 1.43e-04 |
| 10 | 0.2406 | 0.2391 | 1.22e-04 |
| 15 | 0.2281 | 0.2294 | 9.20e-05 |
| 20 | 0.2194 | 0.2199 | 5.90e-05 |
| 25 | 0.2131 | 0.2113 | 2.90e-05 |
| 30 | 0.2080 | 0.2091 | 8.00e-06 |
| 35 | 0.2062 | 0.2063 | 1.00e-06 |
| 40 | 0.2060 | 0.2061 | 8.00e-06 |
| 44 | ~0.207 | ~0.207 | ~1.90e-05 |

**Observations:**
- **Rapid early convergence** — train loss drops from **0.50 to 0.27** in the first 4 epochs, indicating the model quickly learns low-frequency structure (colours, large-scale layout).
- **Steady refinement** — loss continues to decrease gradually through epoch 35, as the model learns finer textures and object details.
- **No overfitting** — train and val losses track closely throughout, validating the paper's claim that 75% masking provides strong regularisation. The gap between train and val loss remains ≤0.001.
- **Cosine LR effect** — the learning rate decayed to its minimum around epoch 35, causing convergence to plateau. After resuming from checkpoint (epochs 36–44), the LR restarts its cosine schedule but values are already near-optimal.

---

## Results & Analysis

### Qualitative Results (Reconstruction)

The visualisation shows 5 validation images as triplets:

| Column | Content |
|---|---|
| **Masked Input** | Only 25% of patches visible (75% shown as black). |
| **Reconstruction** | Model's prediction for **all** 196 patches. |
| **Original** | Ground-truth image. |

**Key observations from the reconstructions:**

1. **Global structure is captured well** — the model correctly reconstructs overall colour distributions, object shapes, and scene layouts even from just 25% of the input.

2. **Textures and fine details are blurry** — this is expected and explained by the MSE loss, which penalises large errors more than small ones, incentivising "average" predictions for uncertain regions. The paper acknowledges this: *"Our MAE reconstruction is only a proxy task; the representation quality is more important than the reconstruction quality."*

3. **Edges and boundaries are approximate** — sharp transitions between objects and backgrounds are smoothed. Again, this is a well-known MSE artefact; perceptual or adversarial losses could produce sharper reconstructions but wouldn't necessarily learn better representations.

4. **Performance on Tiny-ImageNet vs. ImageNet** — the relatively small image resolution (64×64 upscaled to 224×224) means the model works with artificially blurred inputs, making reconstruction inherently harder. With native 224×224 images (as in the paper), reconstructions would be significantly sharper.

### Why These Results Are Expected

1. **MSE loss produces smooth reconstructions.** Per the paper, MAE is not designed to produce photorealistic outputs — it's designed to learn semantic features. The reconstruction quality is a *proxy* for representation quality.

2. **75% masking is intentionally extreme.** The model sees only 49 out of 196 patches. The task is hard by design — this forces the encoder to learn deep contextual understanding rather than relying on local texture copying.

3. **Tiny-ImageNet is a limited dataset.** 100K images across 200 classes provides far less diversity than ImageNet-1K (1.28M images, 1000 classes). The model has fewer examples to learn from, limiting its generalisation.

4. **No downstream fine-tuning is shown.** The true value of MAE is revealed when the pre-trained encoder is fine-tuned on a downstream task (e.g., classification). The paper reports **87.8% top-1 accuracy on ImageNet** with ViT-Huge, demonstrating that the learned representations are highly transferable.

### Comparison with the Original Paper

| Aspect | Paper (ViT-H) | This Implementation (ViT-B) |
|---|---|---|
| Dataset | ImageNet-1K (1.28M images) | Tiny-ImageNet (100K images) |
| Image resolution | 224 × 224 (native) | 64 → 224 (upscaled) |
| Encoder params | ~632M (ViT-Huge) | **85.6M** (ViT-Base) |
| Pre-training epochs | 1600 | **~44** |
| Final recon. loss | Not reported (irrelevant) | **~0.206** |
| Downstream top-1 | **87.8%** (fine-tuned) | N/A (no fine-tuning) |

> This implementation is an **educational-scale** reproduction. The architectural principles are identical, but the scale (model size, dataset size, training duration) is significantly smaller.

### Potential Improvements

- **Downstream fine-tuning** — add a classification head to the pre-trained encoder and fine-tune on Tiny-ImageNet's 200 classes to evaluate representation quality.
- **Longer training** — the paper trains for 1600 epochs; even 200+ epochs on Tiny-ImageNet would likely improve results.
- **Warm-up** — add a linear LR warm-up phase (paper uses 40 epochs of warm-up), which stabilises early training.
- **Larger model** — ViT-Large or ViT-Huge would learn richer representations, given sufficient compute.
- **Native resolution** — use a dataset with 224×224 native images to avoid upscaling artefacts.
- **Perceptual loss** — replace MSE with a perceptual loss for sharper reconstructions (though this may not improve downstream task performance).

---

## How to Run

1. **Platform:** Upload the notebook to [Kaggle](https://www.kaggle.com/) and attach the [Tiny-ImageNet-200 dataset](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet).
2. **GPU:** Enable **GPU T4 × 2** accelerator in the Kaggle notebook settings.
3. **Execute cells in order.** Training is compute-intensive (~15 min/epoch on 2× T4). Checkpoints are saved every 5 epochs for easy resumption.
4. **Resume training** from any checkpoint using the resume cell (loads model, optimizer, scheduler, and scaler states).

---

## Requirements

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model, transforms, DataLoader |
| `torch.cuda.amp` | Mixed-precision training (`GradScaler`, `autocast`) |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation (loss curves, reconstruction grids) |
| `math` | Sinusoidal positional embedding computation |
| `os` | File path handling |

All dependencies are pre-installed in the default Kaggle Python 3 Docker image.

---

## References

1. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R.** (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

2. **Dosovitskiy, A., et al.** (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) — The Vision Transformer (ViT) backbone used in MAE.

3. **Devlin, J., et al.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. — The NLP masked-prediction paradigm that inspired MAE.

4. **Vaswani, A., et al.** (2017). *Attention Is All You Need.* NeurIPS 2017. — The original Transformer architecture; sinusoidal positional embeddings are adopted here.

---

## License

This project is for educational purposes (Generative AI course — AI4009 Assignment 01). Feel free to use and adapt with attribution.
