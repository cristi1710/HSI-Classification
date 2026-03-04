# HSI Classification

Macro F1: **0.8417**

This project classifies hyperspectral image patches using a custom hybrid 1D/2D/3D CNN. Each sample is a 19×19×48 patch (48 spectral bands, 380–2500nm, ~1m/px). The model assigns a land cover class (1–7) based on the spectral and spatial signature of the central pixel.

---

## Quickstart (copy–paste)

```bash
# 1) Clone and enter the repository
git clone https://github.com/cristi1710/HSI-Classification.git
cd HSI-Classification

# 2) Create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm
# For CPU-only: pip install torch torchvision

# 4) Download the dataset via Kaggle CLI
kaggle competitions download -c hsi-classification
unzip hsi-classification.zip -d /kaggle/input/hsi-classification

# 5) Run the training and inference pipeline
python hsi_classification.py
```

A `submission.csv` is saved after each seed completes. Full training runs up to 8.5 hours across 3 seeds × 3 folds on a GPU.

---

## Description

The pipeline loads all labeled `.npy` patches, normalizes them using per-band global statistics, and trains an ensemble of 9 models (3-fold CV × 3 seeds). At inference time, Test Time Augmentation and iterative prior calibration are applied to produce the final class predictions.

Main stages:

1. **Load & normalize** — read `.npy` patches from `train/` and `test/`, apply per-band z-score normalization
2. **Train** — 3-fold stratified CV per seed, with MixUp, geometric augmentations, WeightedRandomSampler, and AMP
3. **Predict** — TTA over 4 geometric transforms, averaged across all folds and seeds
4. **Calibrate** — iterative prior adjustment to align predicted class distribution with expected test distribution
5. **Export** — save `submission.csv` with `filename, label` columns

---

## Input Data

The script expects data at `/kaggle/input/hsi-classification/` (configurable via `BASE_PATH` at the top of the script):

| File / Folder | Description |
|---|---|
| `train/*.npy` | Labeled patches, shape `(19, 19, 48)` or `(48, 19, 19)` |
| `test/*.npy` | Unlabeled patches for inference |
| `labels.csv` | Columns: `filename, label` — label ∈ {1, 2, 3, 4, 5, 6, 7} |
| `sample_submission.csv` | Defines test filenames and submission format |

---

## Output Data

| File | Description |
|---|---|
| `submission.csv` | Final predictions — columns: `filename, label` |

---

## Results

| Metric | Value |
|---|---|
| Leaderboard score (macro F1) | 0.8417 |
| Validation F1 (avg across folds) | ~0.54 |

The gap between validation F1 (~0.54) and leaderboard F1 (0.84) is explained by the calibration step: classes 4 and 5 are absent from training, which suppresses macro F1 at validation time. The iterative prior calibration recovers the correct distribution on the test set.

---

## Project Structure

```
HSI-Classification/
├── hsi_classification.py   # full training and inference pipeline
├── README.md
└── .gitignore              # excludes raw data (train/, test/, *.npy)
```

> Raw data files are not included (5.15 GB). Download via Kaggle CLI as shown above.

---

## Notes

- **Architecture:** three parallel branches — 1D spectral CNN on the central pixel, 3D spatial-spectral CNN on the full patch, 2D CNN after spectral dimension collapse; outputs fused in a shared classification head
- **Class imbalance:** classes 4 and 5 have zero training samples; handled via near-zero loss weights and iterative calibration at inference
- **Hardware:** GPU with CUDA strongly recommended; training time ~8–9h on a Kaggle P100
- **Memory:** the RAM loader pre-allocates a fixed buffer (configurable via `max_memory_gb`); reduce if running on a machine with less RAM

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `ModuleNotFoundError: tqdm` | `pip install tqdm` |
| CUDA out of memory | Reduce `BATCH_SIZE` in the config section at the top of the script |
| Kaggle CLI not found | `pip install kaggle` then set up `~/.kaggle/kaggle.json` |
| Wrong data path | Edit `BASE_PATH` at the top of `hsi_classification.py` |
