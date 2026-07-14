"""
config.py — Central configuration for Lung CT Nodule Detection & Malignancy Prediction
Dataset: LUNA16 (subset of LIDC-IDRI)
VRAM budget: 4 GB — all choices tuned accordingly
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data" / "LUNA16"
SUBSET_DIRS     = [DATA_DIR / f"subset{i}" for i in range(10)]
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
CANDIDATES_CSV  = DATA_DIR / "candidates.csv"
SEG_MASK_DIR    = DATA_DIR / "seg-lungs-LUNA16"

PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
DETECTOR_PATCHES_DIR  = PROCESSED_DIR / "detector_patches"
CLASSIFIER_CROPS_DIR  = PROCESSED_DIR / "classifier_crops"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR        = ROOT_DIR / "logs"
RESULTS_DIR     = ROOT_DIR / "results"

for d in [PROCESSED_DIR, DETECTOR_PATCHES_DIR, CLASSIFIER_CROPS_DIR,
          CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CT PREPROCESSING
# ─────────────────────────────────────────────
TARGET_SPACING  = (1.0, 1.0, 1.0)   # mm — isotropic resampling
HU_MIN          = -1000.0            # air
HU_MAX          =  400.0             # soft tissue
CLIP_RANGE      = (-1000, 400)

# ─────────────────────────────────────────────
# STAGE 1 — 3D U-Net DETECTOR
# ─────────────────────────────────────────────
DETECTOR_PATCH_SIZE  = (64, 64, 64)   # voxels — fits in 4GB with batch=2, fp16
DETECTOR_STRIDE      = (32, 32, 32)   # 50% overlap during sliding-window inference
DETECTOR_BATCH_SIZE  = 2
DETECTOR_EPOCHS      = 80
DETECTOR_LR          = 2e-4
DETECTOR_WEIGHT_DECAY= 1e-5
DETECTOR_CHANNELS    = (32, 64, 128, 192)  # encoder channels — memory-friendly

# Focal + Dice loss weights
DETECTOR_FOCAL_GAMMA = 2.0
DETECTOR_BCE_WEIGHT  = 0.3
DETECTOR_DICE_WEIGHT = 0.7

# Positive sample: voxels within nodule sphere
# Gaussian label smoothing on nodule sphere
GAUSSIAN_SIGMA_RATIO = 0.3   # sigma = radius * 0.3

# ─────────────────────────────────────────────
# STAGE 2 — 3D ResNet-10 CLASSIFIER
# ─────────────────────────────────────────────
CLASSIFIER_CROP_SIZE  = (32, 32, 32)   # centred on detected nodule
CLASSIFIER_BATCH_SIZE = 4
CLASSIFIER_EPOCHS     = 100
CLASSIFIER_LR         = 2e-5
CLASSIFIER_WEIGHT_DECAY = 1e-4
NUM_CLASSES           = 1              # binary: benign / malignant

# Malignancy threshold from LIDC annotations (1-5 scale, ≥3 = malignant)
MALIGNANCY_THRESHOLD  = 3

# Class-weighted loss (malignant << benign in LUNA16)
POS_WEIGHT            = 1

# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────
USE_AMP          = True          # Automatic Mixed Precision (FP16) — crucial for 4GB
GRAD_CLIP        = 0.5
SCHEDULER        = "cosine"      # cosine annealing
WARMUP_EPOCHS    = 5
NUM_WORKERS      = 12
PIN_MEMORY       = True
SEED             = 42

# ─────────────────────────────────────────────
# NODULE CANDIDATE POST-PROCESSING
# ─────────────────────────────────────────────
DETECTION_THRESHOLD   = 0.3      # sigmoid output threshold
NMS_IOU_THRESHOLD     = 0.1      # 3D IoU for non-maximum suppression
MIN_NODULE_DIAM_MM    = 3.0      # LUNA16 minimum
MAX_NODULE_DIAM_MM    = 30.0

# ─────────────────────────────────────────────
# EVALUATION — FROC
# ─────────────────────────────────────────────
FROC_FP_RATES    = [0.125, 0.25, 0.5, 1, 2, 4, 8]   # standard CPM FP/scan
SENSITIVITY_TARGETS = [0.7, 0.8, 0.9]

# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
AUG_FLIP_PROB    = 0.5
AUG_ROTATE_MAX   = 15            # degrees
AUG_SCALE_RANGE  = (0.85, 1.15)
AUG_NOISE_STD    = 0.01
AUG_BRIGHTNESS   = 0.1
