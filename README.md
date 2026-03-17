# 🫁 Lung CT Nodule Detection & Malignancy Prediction
### A Complete 3D Deep Learning System | Final-Year AIML Project

---

## Architecture Overview

```
CT Scan (.mhd)
     │
     ▼
┌─────────────────────────────┐
│  PREPROCESSING              │
│  • Resample → 1mm isotropic │
│  • Clip & normalise HU      │
│  • Patch / crop extraction  │
└────────────┬────────────────┘
             │  64³ patches
             ▼
┌──────────────────────────────┐
│  STAGE 1: 3D U-Net DETECTOR  │
│  • 4-level encoder/decoder   │
│  • Skip connections          │
│  • Gaussian sphere labels    │
│  • Focal + Dice loss         │
│  → Probability map           │
│  → Candidate extraction (NMS)│
└────────────┬─────────────────┘
             │  32³ nodule crops
             ▼
┌────────────────────────────────┐
│  STAGE 2: ResNet-10 CLASSIFIER │
│  • SE attention blocks         │
│  • Mixup augmentation          │
│  • Label smoothing BCE         │
│  → Malignancy probability      │
└────────────┬───────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  STAGE 3: GRAD-CAM 3D        │
│  • Gradient-based attention  │
│  • Axial/coronal/sagittal    │
│  • Score-CAM (gradient-free) │
└──────────────────────────────┘
```

---

## VRAM Budget (4 GB GPU)

| Component           | Patch Size | Batch | VRAM Usage |
|---------------------|------------|-------|------------|
| UNet3D (detector)   | 64³        | 2     | ~2.8 GB    |
| ResNet-10 (classif) | 32³        | 8     | ~0.6 GB    |
| Grad-CAM            | 32³        | 1     | ~0.4 GB    |

All models use **FP16 (AMP)** + **gradient checkpointing** (UNet3D) to stay under 4 GB.

---

## Dataset: LUNA16

LUNA16 is a subset of **LIDC-IDRI** (the largest public lung nodule database).

| Property           | Value                       |
|--------------------|-----------------------------|
| CT scans           | 888 scans                   |
| Annotated nodules  | 1,186 nodules               |
| Nodule size        | 3–30 mm                     |
| Annotation source  | 4 radiologists per scan     |
| Format             | .mhd / .raw                 |
| Download           | luna16.grand-challenge.org  |

### Required Folder Structure
```
data/LUNA16/
    subset0/   *.mhd  *.raw
    subset1/   *.mhd  *.raw
    ...
    subset9/   *.mhd  *.raw
    annotations.csv
    candidates.csv
    seg-lungs-LUNA16/   (optional)
```

---

## File Structure

```
lung_nodule_ai/
├── main.py                      # Entry point
├── config.py                    # All hyperparameters
├── inference.py                 # End-to-end pipeline
├── requirements.txt
│
├── data/
│   ├── preprocessing.py         # LUNA16 load + resample + patch extraction
│   └── dataset.py               # PyTorch Dataset classes + augmentation
│
├── models/
│   ├── unet3d.py                # 3D U-Net + FocalDiceLoss
│   └── resnet3d.py              # 3D ResNet-10 + LabelSmoothingBCE
│
├── training/
│   ├── train_detector.py        # Detector training loop
│   └── train_classifier.py      # Classifier training loop + Mixup
│
├── evaluation/
│   └── metrics.py               # FROC, AUC, FROC plots, classification report
│
├── explainability/
│   └── gradcam3d.py             # Grad-CAM + Score-CAM + visualisation
│
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard logs
└── results/                     # FROC curves, ROC plots, Grad-CAM images
```

---

## Part 1 — First-Time Setup & Training (subset0)

### Step 1 — Create folder structure

```bash
mkdir -p lung_nodule_ai/data/LUNA16
mkdir -p lung_nodule_ai/data/processed/detector_patches
mkdir -p lung_nodule_ai/data/processed/classifier_crops
mkdir -p lung_nodule_ai/checkpoints
mkdir -p lung_nodule_ai/logs
mkdir -p lung_nodule_ai/results
```

### Step 2 — Place LUNA16 data

Download from [luna16.grand-challenge.org](https://luna16.grand-challenge.org) and place the extracted subset folder and CSV files as:

```
lung_nodule_ai/
└── data/
    └── LUNA16/
        ├── subset0/          ← extracted folder (89 .mhd + .raw files)
        ├── annotations.csv
        └── candidates.csv
```

### Step 3 — Set up Python environment

```bash
cd lung_nodule_ai/

# Pin Python 3.11 for this folder (requires pyenv)
pyenv local 3.11.9
python --version   # Python 3.11.9

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
which python   # should show venv path
```

### Step 4 — Install PyTorch (CUDA first)

```bash
# CUDA 12.6 wheels — compatible with CUDA 13.1 driver
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Verify GPU is detected
python -c "import torch; print(torch.__version__); \
    print('GPU:', torch.cuda.is_available()); \
    print(torch.cuda.get_device_name(0))"
# Expected:
# 2.9.0+cu126
# GPU: True
# NVIDIA GeForce RTX XXXX
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt

# Verify key imports
python -c "import SimpleITK, monai, sklearn, matplotlib; print('All imports OK')"
```

### Step 6 — Run unit tests

```bash
python main.py test
# Expected output:
# ✓ UNet3D forward pass
# ✓ ResNet3D forward pass + features
# ✓ FocalDiceLoss
# ✓ Grad-CAM 3D
# ✓ Classification metrics
# ✓ Preprocessing utilities
# Tests passed: 6/6
```

> ⚠️ Fix any failing tests before proceeding. Do not skip this step.

### Step 7 — Run synthetic demo (optional but recommended)

```bash
python main.py demo
# Creates fake CT data, runs the full pipeline end-to-end
# Saves sample outputs to results/demo/
# Takes ~2 minutes — useful to preview outputs before real training
```

### Step 8 — Preprocess subset0

This is the longest step. It reads all 89 CT scans, resamples them, and extracts patches.

```bash
python main.py preprocess
# - Loads each .mhd from subset0/
# - Resamples to 1mm isotropic spacing
# - Normalises HU values to [0, 1]
# - Extracts 64³ detector patches (positive + negative)
# - Extracts 32³ classifier crops (nodule only)
# - Saves everything as .npz files
#
# Expected totals at completion:
#   Detector patches — total: ~4000, pos: ~800, neg: ~3200
#   Classifier crops — total: ~150
#
# Time: 30–60 minutes
# Skipped scans print a warning — that is OK
```

Verify after completion:

```bash
ls data/processed/detector_patches/ | wc -l   # ~4000
ls data/processed/classifier_crops/  | wc -l   # ~150
```

### Step 9 — Record a processing snapshot

```bash
# Save list of processed UIDs for reference
ls data/LUNA16/subset0/*.mhd | xargs -I{} basename {} .mhd \
    > data/processed/processed_uids_subset0.txt

echo "subset0 processed on $(date)" >> data/processed/processing_log.txt
ls data/processed/detector_patches/ | wc -l >> data/processed/processing_log.txt

cat data/processed/processing_log.txt
# subset0 processed on Mon Mar 17 2026
# 4012
```

### Step 10 — Train the detector (Stage 1)

Open a second terminal and start TensorBoard to monitor live:

```bash
source venv/bin/activate
tensorboard --logdir logs/
# Open http://localhost:6006 in your browser
```

Back in the main terminal:

```bash
python main.py train-det
# - Loads all .npz patches from detector_patches/
# - Trains 3D U-Net with FocalDiceLoss
# - FP16 (AMP) + gradient checkpointing active (4 GB safe)
# - Saves best model whenever val loss improves
# - Early stopping after 15 epochs with no improvement
#
# Watch in TensorBoard:
#   loss:        ~0.9 → ~0.3
#   dice:        ~0.1 → ~0.5+
#   sensitivity: increases over time
#
# Time: 4–8 hours for 60 epochs on subset0
# Best checkpoint → checkpoints/detector_best.pth
```

### Step 11 — Train the classifier (Stage 2)

```bash
python main.py train-cls
# - Loads all .npz crops from classifier_crops/
# - Trains 3D ResNet-10 with LabelSmoothingBCE
# - Mixup augmentation on 50% of batches
# - Saves best model whenever val AUC improves
#
# Watch in TensorBoard:
#   loss: ~0.7 → ~0.3
#   AUC:  ~0.5 → ~0.75+
#
# Time: 2–4 hours for 80 epochs on subset0
# Best checkpoint → checkpoints/classifier_best.pth
```

### Step 12 — Evaluate

```bash
python main.py evaluate
# Outputs written to results/:
#   roc_curve.png
#   eval_metrics.json
#   detector_training_curves.png
#   classifier_training_curves.png
#
# Terminal summary (expected for subset0 only):
#   AUC-ROC:     ~0.78
#   Sensitivity: ~0.75
#   Specificity: ~0.72
#   CPM:         ~0.65–0.70
# (performance improves as more subsets are added)
```

### Step 13 — Save baseline checkpoint

Before adding more data, back up the subset0-trained weights:

```bash
cp checkpoints/detector_best.pth    checkpoints/detector_subset0_baseline.pth
cp checkpoints/classifier_best.pth  checkpoints/classifier_subset0_baseline.pth

echo "Baseline saved. Metrics:" >> data/processed/processing_log.txt
cat results/eval_metrics.json | python -c \
    "import json,sys; m=json.load(sys.stdin); print(f'AUC: {m[\"auc\"]:.4f}')" \
    >> data/processed/processing_log.txt
```

---

## Part 2 — Adding subset1 and Retraining

### Step 1 — Unzip subset1

```bash
cd lung_nodule_ai/data/LUNA16/
unzip subset1.zip
ls subset1/   # verify ~89 .mhd and .raw files
rm subset1.zip
```

### Step 2 — Back up existing patches

This prevents subset0 patches from being generated twice.

```bash
mkdir -p data/processed/backup_subset0/detector_patches
mkdir -p data/processed/backup_subset0/classifier_crops

mv data/processed/detector_patches/*.npz \
    data/processed/backup_subset0/detector_patches/
mv data/processed/classifier_crops/*.npz \
    data/processed/backup_subset0/classifier_crops/

# Verify
ls data/processed/backup_subset0/detector_patches/ | wc -l  # matches previous count
ls data/processed/detector_patches/ | wc -l                  # 0
```

### Step 3 — Hide subset0 from the preprocessor

```bash
mv data/LUNA16/subset0 data/LUNA16/subset0_DONE
# Preprocessor only reads folders named subsetN — subset0_DONE is ignored
```

### Step 4 — Preprocess subset1 only

```bash
python main.py preprocess
# Processes only subset1
# New patches → data/processed/detector_patches/
# New crops   → data/processed/classifier_crops/
# Time: 30–60 minutes
```

### Step 5 — Restore and merge

```bash
# Restore subset0 folder name
mv data/LUNA16/subset0_DONE data/LUNA16/subset0

# Merge subset1 patches with backed-up subset0 patches
mv data/processed/backup_subset0/detector_patches/*.npz \
    data/processed/detector_patches/
mv data/processed/backup_subset0/classifier_crops/*.npz \
    data/processed/classifier_crops/

rm -rf data/processed/backup_subset0/

# Verify combined counts (~double the subset0 numbers)
ls data/processed/detector_patches/ | wc -l
ls data/processed/classifier_crops/  | wc -l
```

### Step 6 — Record the new state

```bash
ls data/LUNA16/subset1/*.mhd | xargs -I{} basename {} .mhd \
    > data/processed/processed_uids_subset1.txt

echo "subset1 added on $(date)" >> data/processed/processing_log.txt
ls data/processed/detector_patches/ | wc -l >> data/processed/processing_log.txt

cat data/processed/processing_log.txt
# subset0 processed on Mon Mar 17 2026
# 4012
# subset1 added on Tue Mar 18 2026
# 8100
```

### Step 7 — Resume detector training

```bash
python main.py train-det \
    --resume checkpoints/detector_best.pth \
    --epochs 30
# Loads existing weights + optimiser state + LR schedule
# 30 additional epochs on the combined subset0 + subset1 patches
# Overwrites detector_best.pth if val loss improves
# Time: 2–4 hours
```

### Step 8 — Resume classifier training

```bash
python main.py train-cls \
    --resume checkpoints/classifier_best.pth \
    --epochs 40
# Resumes from existing checkpoint
# Trains on crops from both subsets
# AUC expected to improve from ~0.78 to ~0.82+
# Time: 1–2 hours
```

### Step 9 — Re-evaluate and compare

```bash
python main.py evaluate
cat data/processed/processing_log.txt  # compare new results to baseline
```

---

## Part 3 — Adding More Subsets (subset2, 3, 4 …)

Repeat Part 2 exactly for each new subset. The steps are always identical:

```
1.  unzip subsetN.zip  inside  data/LUNA16/
2.  back up existing patches  →  backup_subsets_so_far/
3.  rename all previous subset folders  →  subsetN_DONE
4.  python main.py preprocess          (processes only subsetN)
5.  rename subsetN_DONE  →  subsetN
6.  merge backup patches back in
7.  python main.py train-det  --resume checkpoints/detector_best.pth   --epochs N
8.  python main.py train-cls  --resume checkpoints/classifier_best.pth  --epochs N
9.  python main.py evaluate
10. update processing_log.txt
```

Reduce additional epochs each round — the model is already partially trained:

| Round | Subsets Added  | Extra Detector Epochs | Extra Classifier Epochs |
|-------|----------------|-----------------------|-------------------------|
| 1     | subset0        | 60 (full)             | 80 (full)               |
| 2     | + subset1      | 30                    | 40                      |
| 3     | + subset2      | 25                    | 35                      |
| 4     | + subset3      | 20                    | 30                      |
| 5     | + subset4–6    | 15                    | 25                      |

---

## Quick Reference

```
FIRST TIME:
  mkdir folders → unzip subset0 → pip install →
  main.py test → main.py preprocess →
  main.py train-det → main.py train-cls → main.py evaluate

ADD NEW SUBSET:
  unzip subsetN → backup old patches → rename old subsets to _DONE →
  main.py preprocess → rename back → merge patches →
  train-det --resume → train-cls --resume → evaluate

MONITORING:
  tensorboard --logdir logs/    # http://localhost:6006
  python main.py infer --scan /path/to/scan.mhd

NEVER:
  ✗ Run preprocess with old and new subsets both visible (causes double processing)
  ✗ Train without --resume after the first round (discards all previous learning)
  ✗ Delete checkpoints/detector_best.pth or classifier_best.pth between rounds
```

---

## Model Details

### Stage 1 — 3D U-Net Detector

```
Input:  (2, 1, 64, 64, 64)  [batch=2, channel=1, D×H×W]
Encoder: Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → Conv(64→128) → MaxPool → Conv(128→256)
Bottleneck: Conv(256→512)
Decoder: Up+Skip → Conv(512+256→256) → Up+Skip → ... → Conv(64+32→32)
Head: Conv(32→16) → Conv(16→1) → sigmoid
Output: (2, 1, 64, 64, 64)  probability map

Loss:      0.5 × Focal(γ=2) + 0.5 × Dice
Optimiser: AdamW  lr=3e-4  wd=1e-5
Scheduler: Linear warmup (5 ep) → Cosine annealing
AMP:       FP16
Gradient checkpointing: YES (saves ~35% VRAM)
```

### Stage 2 — 3D ResNet-10 Classifier

```
Input:  (8, 1, 32, 32, 32)
Stem:   Conv5(1→32, stride=2) → 16³
Layer1: SEBasicBlock(32→64,  stride=1) → 16³
Layer2: SEBasicBlock(64→128, stride=2) → 8³
Layer3: SEBasicBlock(128→256,stride=2) → 4³
Head:   GAP → Dropout(0.4) → FC(256→1)
Output: (8, 1) logit

Loss:      LabelSmoothing-BCE(ε=0.05) + pos_weight=3.0
Optimiser: AdamW  lr=1e-4  wd=1e-4
Mixup:     α=0.2 (50% of batches)
TTA:       8-fold at inference
```

---

## Evaluation Metrics

### Detection (FROC)
- **CPM (Competition Performance Metric)**: Mean sensitivity at FP/scan = [0.125, 0.25, 0.5, 1, 2, 4, 8]
- State-of-the-art CPM on LUNA16: ~0.90 (with full training)
- **Sensitivity @ 1 FP/scan**: clinical deployment target

### Classification
- **AUC-ROC**: primary metric
- **Sensitivity / Specificity** at Youden-J threshold
- **PPV / NPV**: clinically meaningful
- **Average Precision (AP)**: handles class imbalance well


