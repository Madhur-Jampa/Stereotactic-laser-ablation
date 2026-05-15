# 🧠 Stereotactic Laser Ablation — AI-Assisted Surgical Probe Path Planner

> An end-to-end AI pipeline that detects brain tumors in T1-weighted MRI scans, segments them using a fine-tuned U-Net, and computes a safe cubic Bézier probe trajectory from the skull boundary to the tumor center — while avoiding Broca's region.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bWGwXagtuBbhIy1hXT-aYmsmC4qwhun3?usp=sharing)

---

## 📌 Background

Stereotactic laser ablation is a minimally invasive neurosurgical procedure where a laser probe is inserted through a small port in the skull to heat and destroy (necrotise) tumors located in physically inoperable regions of the brain. Despite its potential, the procedure is not widely adopted because surgeons currently lack reliable, automated tools to plan the probe's path and trajectory.

This project addresses that gap with a three-stage AI pipeline.

---

## 🔁 Pipeline Overview

```
Input: T1-weighted MRI Image (uploaded by user)
        │
        ▼
┌───────────────────────────────────────┐
│  Stage 1 — MedGemma (Detection)       │
│  google/medgemma-1.5-4b-it (4-bit)    │──── "No" ──▶ Output: No tumor detected
│  Prompt: "Is there a tumor?"          │
└───────────────────────────────────────┘
        │ "Yes"
        ▼
┌───────────────────────────────────────┐
│  Stage 2 — U-Net (Segmentation)       │
│  Custom 3-level encoder-decoder       │
│  Input: 128×128 grayscale MRI         │──▶ Binary tumor mask + smooth contour
│  Loss: BCE (pos_weight=2.5) + Dice    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Stage 3 — Probe Path Planner         │
│  Cubic Bézier curve                   │
│  Entry: bottom-left skull boundary    │──▶ Annotated MRI with probe trajectory
│  Target: tumor centroid               │    (avoids Broca's region)
└───────────────────────────────────────┘
        │
        ▼
Output: final_ai_navigation.png
        predicted_mask.png
        tumor_data.json
```

---

## 🧩 Stage Details

### Stage 1 — Tumor Detection with MedGemma

- **Model:** `google/medgemma-1.5-4b-it` loaded with **4-bit quantization** via `BitsAndBytesConfig`
- The model is prompted with the MRI image and asked: *"Is there a tumor in this T1-weighted brain MRI?"*
- If a tumor is detected, MedGemma is prompted a second time to return a **tight bounding box** around the tumor in JSON format (percentage coordinates), along with a size estimate (`small`, `medium`, or `large`).
- The bounding box and image are saved to Google Drive (`/drive/MyDrive/tumor_data.json`) for use in downstream stages.
- **Accuracy: 93.33%** (28/30 manually tested images correctly classified)

### Stage 2 — Tumor Segmentation with U-Net

**Architecture:**

| Layer | Details |
|---|---|
| Encoder Block 1 | Conv2d(1→64) × 2, BatchNorm, ReLU → MaxPool2d |
| Encoder Block 2 | Conv2d(64→128) × 2, BatchNorm, ReLU → MaxPool2d |
| Bottleneck | Conv2d(128→256) × 2, BatchNorm, ReLU |
| Decoder Block 1 | Upsample + Skip(e2) → Conv2d(384→128) × 2 |
| Decoder Block 2 | Upsample + Skip(e1) → Conv2d(192→64) × 2 |
| Output | Conv2d(64→1), 1×1 kernel |

**Training Setup:**

| Parameter | Value |
|---|---|
| Input size | 128 × 128 (grayscale) |
| Epochs | 20 |
| Batch size | 16 |
| Optimizer | Adam (lr = 1e-3) |
| Loss function | BCE (pos_weight=2.5) + Dice Loss |
| Train/Val split | 80% / 20% |
| Mixed precision | Enabled (AMP, CUDA) |
| Augmentation | Random horizontal flip, Gaussian blur (3×3) |

**Post-processing:**
- Sigmoid activation → threshold at (mean + 0.45 × std)
- Morphological closing (5×5 kernel, 2 iterations)
- Gaussian blur (11×11) for smooth edges
- Largest connected component retained

**Accuracy:**
- Training dataset: **100%**
- Unseen Kaggle dataset: **76.7%** (23/30 manually tested images correctly segmented)

### Stage 3 — Probe Path Planning

- The tumor centroid is computed from the predicted segmentation mask (`np.mean` of non-zero pixel coordinates).
- A **cubic Bézier curve** is generated from a fixed entry point on the skull boundary (bottom-left: 12% W, 82% H) to the tumor centroid, using two intermediate control points to arc around obstacles.
- **Broca's region** is modelled as a proportional polygon (relative to image dimensions) in the left frontal region of the brain. It is rendered as a purple semi-transparent overlay and the probe path is designed to avoid it.
- The final output image includes: tumor contour (red/blue), Broca's region overlay (purple), probe path (orange), entry point marker (cyan), and text labels.
- **Path success rate: 100%**

---

## 📂 Datasets

| Dataset | Source | Purpose |
|---|---|---|
| Brain Tumor Dataset | [Figshare](https://ndownloader.figshare.com/articles/1512427/versions/5) | Training (`.mat` files, MATLAB format) |
| LGG MRI Segmentation | [Kaggle — mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | Training (`.tif` images + masks) |
| Brain MRI for Tumor Detection | [Kaggle — navoneel](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) | Testing & validation only |

- **Total training images: 3,929** (Figshare `.mat` files + LGG `.tif` files, combined and preprocessed)
- All images are resized to **128×128** and converted to grayscale PNG before training.

---

## 📊 Accuracy Summary

| Component | Accuracy |
|---|---|
| Tumor Detection (MedGemma) | 93.33% (28/30) |
| Tumor Segmentation — Training Set | 100% |
| Tumor Segmentation — Kaggle (unseen) | 76.7% (23/30) |
| Probe Path Planning | 100% |

---

## 🛠️ Requirements

All cells are designed to run on **Google Colab** with a **GPU runtime (T4 recommended)**.

**Python packages installed in the notebook:**

```
torch torchvision torchaudio   # CUDA 12.1
bitsandbytes>=0.46.1
transformers>=4.37.2
accelerate
h5py
scipy
opencv-python (cv2)
numpy
matplotlib
Pillow
huggingface_hub
```

**External requirements:**
- A [Hugging Face account](https://huggingface.co/) with access to `google/medgemma-1.5-4b-it`
- A Hugging Face API token stored as a Colab secret under the key `HF_TOKEN`
- A Kaggle API token (for downloading the LGG dataset)
- Google Drive (for saving intermediate outputs between cells)

---

## 🚀 How to Run

> **Runtime:** Go to `Runtime → Change runtime type → T4 GPU` before running.

Run the cells in the following order:

| Cell | Purpose |
|---|---|
| Cell 1 | Install PyTorch (CUDA 12.1) and core dependencies |
| Cell 2 | Mount Drive, upload MRI image, run MedGemma detection + bounding box |
| Cell 3 | Download Figshare brain tumor dataset |
| Cell 4 | Preview Figshare `.mat` samples |
| Cell 5 | Download LGG dataset from Kaggle |
| Cell 6 | Extract LGG dataset |
| Cell 7 | Check directory structure |
| Cell 8 | Combine Figshare + LGG into unified dataset (`/content/final_dataset/`) |
| Cell 9 | Train the U-Net model (saves best weights to `/content/best_model.pth`) |
| Cell 10 | Run U-Net inference — segment tumor, draw boundary |
| Cell 11 | Full navigation system — segmentation + Broca overlay + Bézier probe path |

> **Note:** Cells 3–9 (dataset download and training) only need to be run **once**. After training, go directly to Cell 10/11 for inference.

---

## 📁 Output Files

| File | Description |
|---|---|
| `/drive/MyDrive/current_mri.png` | Uploaded MRI saved to Drive |
| `/drive/MyDrive/tumor_data.json` | MedGemma bounding box + metadata |
| `/content/best_model.pth` | Trained U-Net weights |
| `/content/final_boundary.png` | MRI with tumor boundary overlay |
| `/content/final_mask.png` | Binary segmentation mask |
| `/content/final_mask.npy` | Mask as NumPy array |
| `/content/final_ai_navigation.png` | Final annotated image with probe path |
| `/content/predicted_mask.png` | Predicted mask from navigation cell |

---

## ⚠️ Limitations

- Trained on **3,929 images** — a larger, more diverse dataset would substantially improve out-of-distribution generalization.
- Currently constrained by **8 GB RAM** and **Google Colab GPU memory limits**.
- The Broca's region is defined as a **fixed proportional polygon** and is not patient-specific or registration-based.
- The probe entry point is currently **hardcoded** to the bottom-left skull boundary (12% W, 82% H) and is not dynamically computed.
- Tested only on **2D axial T1-weighted MRI slices** — no 3D volumetric planning yet.
- Not validated against clinical neurosurgical standards.

---

## 🔭 Future Improvements

- Train on a significantly larger corpus of T1-weighted MRIs to improve cross-dataset generalization.
- Replace the fixed Broca's region polygon with **atlas-based or registration-based** anatomical localization.
- Dynamically compute the optimal **skull entry point** rather than using a hardcoded location.
- Extend to **3D MRI volumes** for full volumetric path planning.
- Add additional critical region avoidance (motor cortex, visual cortex, ventricular system).
- Validate probe path planning against clinical neurosurgical workflows.

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. See [`LICENSE`](./LICENSE) for details.

---

## 🙏 Acknowledgements

- [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contributors.
- [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) — Mateusz Buda et al.
- [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) — Navoneel Chakrabarty.
- Google DeepMind for the [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) model.
- Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation.*

---

> **⚕️ Disclaimer:** This is a research prototype and is **not intended for clinical use**. All results must be reviewed and validated by a qualified neurosurgeon before any clinical application.

