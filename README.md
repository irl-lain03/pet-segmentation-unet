# Pet Image Segmentation and Classification Using U-Net and Attention U-Net

An end-to-end deep learning pipeline implemented in PyTorch for the **CSE428: Image Processing** course. Developed on the Oxford-IIIT Pet Dataset (37 pet breeds) for simultaneous semantic segmentation (binary foreground mask extraction from trimap annotations) and multi-class breed classification using standard U-Net and Attention U-Net backbones with joint classification heads.

---

## Dataset Pipeline

* **Dataset:** Oxford-IIIT Pet Dataset (3,680 trainval, 3,669 test images across 37 cat and dog breeds).
* **Target Modalities:** Dual-target pipeline loading categorical breed labels alongside pixel-level segmentation trimaps.
* **Pre-processing:**
  * Spatial interpolation: Images resized to 128x128 via bilinear interpolation; trimaps resized via nearest-neighbor interpolation.
  * Trimap Binarization: Merged foreground (class 1) and boundary pixels (class 3) into a single binary foreground mask.
  * Split: Stratified 85/15 train/validation split on `trainval` (3,128 train, 552 validation, 3,669 test).
  * Feature Normalization: Per-channel Standardization across RGB channels.

---

## Model Architectures

### 1. U-Net Baseline
* Encoder-decoder architecture with contracting downsampling blocks and symmetrical expanding upsampling transposed convolutions.
* Direct skip connections between encoder and decoder feature maps to retain fine-grained spatial and boundary details.
* Dual output formulation: A 1x1 convolution for single-channel segmentation logits, combined with a pooled global classification head for 37-class breed prediction.

### 2. Attention U-Net Variant
* Incorporates additive Attention Gates (AGs) along skip connections prior to concatenation with decoder features.
* Suppresses irrelevant background activations and highlights salient anatomical boundaries of the target subject.

---

## Training and Optimization

* **Framework:** PyTorch (CUDA acceleration enabled).
* **Loss Functions:** Combined Binary Cross-Entropy with Logits / Dice Loss for segmentation, paired with Cross-Entropy Loss for multi-class classification.
* **Checkpointing:** Epoch-level model state serialization tracking best validation segmentation metric.

---

## Setup and Reproduction

```bash
git clone [https://github.com/irl-lain03/](https://github.com/irl-lain03/)<repository-name>.git
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
