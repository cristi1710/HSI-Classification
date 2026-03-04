HSI Classification — Kaggle Competition
Score: 0.8417 macro F1
Overview
Hyperspectral image classification on patch-level data extracted from a full hyperspectral scene. Each sample is a 19×19×48 patch (48 spectral bands, 380–2500nm range, ~1m spatial resolution). The task is to classify each patch by the land cover type of its central pixel into one of 7 classes.
The dataset contains ~6822 labeled training patches and ~37087 test patches. Classes are heavily imbalanced — class 6 dominates with ~13500 samples, while classes 4 and 5 are completely absent from the training set.
Approach
Architecture — Hybrid 1D/2D/3D CNN
A custom multi-branch network that processes each patch simultaneously from three perspectives:

Spectral branch (1D CNN): extracts deep spectral signatures from the central pixel across all 48 bands
Spatial-spectral branch (3D CNN): captures joint spatial and spectral patterns across the full 19×19 patch volume
Global spatial branch (2D CNN): refines spatial features after collapsing the spectral dimension

The outputs of the spectral and spatial-spectral branches are concatenated and passed through a shared classification head.
Training Strategy

3-fold stratified cross-validation repeated across 3 random seeds (9 total models per submission)
WeightedRandomSampler to counteract class imbalance during training
MixUp augmentation applied with 50% probability per batch
Geometric augmentations: random horizontal/vertical flips and 90° rotations
AdamW optimizer with OneCycleLR scheduler and AMP (mixed precision) for speed
CrossEntropyLoss with label smoothing (0.1) and near-zero weights for absent classes (4 and 5)

Inference

Test Time Augmentation (TTA): averages softmax probabilities over 4 geometric transformations per sample
Iterative prior calibration: adjusts class priors over 50 iterations to align the predicted distribution with the expected distribution of the test set

Results
MetricValueLeaderboard score (macro F1)0.8417Final rank7th placeValidation F1 (avg across folds)~0.54
The gap between validation F1 (~0.54) and leaderboard F1 (0.84) is explained by the calibration step: at validation time classes 4 and 5 are present but the model cannot predict them, suppressing macro F1. On the test set, the iterative prior calibration recovers the correct distribution.
Tech Stack
Python, PyTorch, NumPy, pandas, scikit-learn, tqdm
Dataset
Available on Kaggle: hsi-classification
kaggle competitions download -c hsi-classification
Repository Structure
hsi_classification.py   # full training and inference pipeline
README.md
.gitignore              # excludes raw data (train/, test/, *.npy)

Raw data files are not included due to size (5.15 GB). Download via Kaggle CLI above.
