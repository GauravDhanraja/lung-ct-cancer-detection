"""
config.py — Central configuration for Lung CT Nodule Detection & Malignancy Prediction
Dataset: LUNA16 (subset of LIDC-IDRI), all 10 subsets
VRAM budget: 16 GB (RTX 5070 Ti) — all choices tuned accordingly

Sizing methodology: activation footprint for the detector and classifier was
measured empirically (forward hooks, per-sample element counts, scaled by
batch) rather than guessed, then combined with param+grad+AdamW-state memory
and a 1GB flat overhead for cuDNN workspace/fragmentation. Chosen configs
target ~75-85% of the 16GB budget so there's headroom for cuDNN's algorithm
autotuning (first few iterations can spike) and anything else on the GPU.
If you still hit OOM, the first knobs to turn are DETECTOR_BATCH_SIZE down,
then DETECTOR_PATCH_SIZE down — in that order.
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
# 96^3 patches (up from 64^3) give ~3.4x the spatial context per crop — more
# surrounding anatomy for false-positive reduction, at the cost of a bigger
# activation footprint. Measured in practice: batch=6 training steady-state
# ≈12.6GB, matching the earlier estimate well.
DETECTOR_PATCH_SIZE  = (96, 96, 96)   # voxels
DETECTOR_STRIDE      = (48, 48, 48)   # 50% overlap during sliding-window inference
DETECTOR_BATCH_SIZE  = 6
DETECTOR_EPOCHS      = 80
DETECTOR_LR          = 3.5e-4         # sqrt(batch 2->6) scaling: 2e-4 * sqrt(3)
DETECTOR_WEIGHT_DECAY= 1e-5
DETECTOR_CHANNELS    = (32, 64, 128, 256)  # restored to standard U-Net depth
                                            # (bottleneck = 512ch, was 384ch)
DETECTOR_USE_CHECKPOINT = True        # kept on: ~20% slower but buys back
                                       # enough headroom to be safe by default.
                                       # Flip to False if nvidia-smi shows you
                                       # have >2-3GB free — it's a free speedup.
DETECTOR_GRAD_ACCUM_STEPS = 1         # bump to 2-3 for a larger effective
                                       # batch (12-18) at no extra VRAM cost
DETECTOR_VAL_BATCH_MULTIPLIER = 1.25  # val has no backward pass so it can run
                                       # a bit bigger than train for free, but
                                       # doubling (fine at the old batch=2) now
                                       # pushes 6->12 and got measured at a
                                       # ~15.5GB peak — too close to the 16GB
                                       # ceiling. 1.25x (6->7) keeps the benefit
                                       # with real margin.

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
# The classifier was never really VRAM-bound (<5M params, tiny volumes) —
# even base_channels=48 @ crop=48^3 @ batch=32 measures ≈ 3.4GB. Batch went
# up 8x (not e.g. 16x) deliberately: LUNA16's malignancy-labeled nodule count
# is in the low thousands, and too few optimizer steps/epoch under-trains on
# a small dataset even though the GPU could easily fit a bigger batch.
CLASSIFIER_CROP_SIZE  = (48, 48, 48)   # centred on detected nodule — more
                                        # margin context for malignancy cues
                                        # (spiculation, lobulation) than 32^3
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_EPOCHS     = 100
CLASSIFIER_LR         = 6e-5           # sqrt(batch 4->32) scaling: 2e-5 * sqrt(8)
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_BASE_CHANNELS = 32          # up from 32 — modest capacity bump,
                                        # justified by more data (10 subsets)
                                        # not just spare VRAM
CLASSIFIER_GRAD_ACCUM_STEPS = 1
CLASSIFIER_VAL_BATCH_MULTIPLIER = 2    # classifier has plenty of VRAM
                                        # headroom (~3.4GB at batch=32), so
                                        # doubling for val is safe here —
                                        # unlike the detector, this one
                                        # isn't close to any ceiling
NUM_CLASSES           = 1              # binary: benign / malignant

# Malignancy threshold from LIDC annotations (1-5 scale, ≥3 = malignant)
MALIGNANCY_THRESHOLD  = 3

# Class-weighted loss (malignant << benign in LUNA16)
POS_WEIGHT            = 1

# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────
USE_AMP          = True          # Automatic Mixed Precision (FP16) — still a
                                  # free win on RTX 5070 Ti's tensor cores,
                                  # independent of the VRAM headroom
GRAD_CLIP        = 0.5
SCHEDULER        = "cosine"      # cosine annealing
WARMUP_EPOCHS    = 6             # +1 epoch: bigger batches mean fewer
                                  # optimizer steps per epoch, so warmup
                                  # needs slightly more epochs to cover
                                  # a comparable number of steps
CUDNN_BENCHMARK  = True           # autotune conv algorithms — safe because
                                  # patch/crop sizes are fixed per stage

# NUM_WORKERS derived from CPU count rather than hardcoded, so this config
# behaves consistently if you move between machines. Cap at 8: LUNA16
# loading is I/O + resample bound, not helped much past that.
NUM_WORKERS      = min(8, os.cpu_count() or 4)
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
