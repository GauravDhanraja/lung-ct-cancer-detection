"""
model_factory.py
─────────────────
Single source of truth for constructing UNet3D/ResNet3D instances and for
loading their checkpoints.

Why this exists: previously, every call site that needed a model (train
scripts, main.py's evaluate stage, presumably inference.py too) built one
inline with its own copy of "which config values matter". They drifted —
train_classifier.py read CLASSIFIER_BASE_CHANNELS from config, but
main.py's evaluate stage didn't, so a config change (32 -> 48 channels)
broke loading a real checkpoint with a wall of size-mismatch errors that
took a while to trace back to one missing kwarg.

Two layers of defense against that happening again:
  1. build_detector()/build_classifier() are the ONLY place default
     architecture args are read from config. Every other file should call
     these instead of constructing UNet3D/ResNet3D directly.
  2. load_detector_checkpoint()/load_classifier_checkpoint() go further:
     checkpoints saved by the current train_detector.py/train_classifier.py
     record their own architecture (see the "arch" key), so loading a
     checkpoint reconstructs the model that checkpoint actually needs,
     regardless of what config.py says *now*. If config.py changes after
     you've trained a model, old checkpoints still load correctly instead
     of silently trying (and failing) to match the new config.
     Checkpoints saved before this change won't have "arch" and fall back
     to current config, same as before.
"""

import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from models.unet3d import UNet3D
from models.resnet3d import ResNet3D


# ═══════════════════════════════════════════════════════
# BUILD — construct a fresh model from (explicit args, else config)
# ═══════════════════════════════════════════════════════

def build_detector(device="cpu", channels=None, use_checkpoint: Optional[bool] = None) -> UNet3D:
    channels = channels if channels is not None else cfg.DETECTOR_CHANNELS
    if use_checkpoint is None:
        use_checkpoint = cfg.DETECTOR_USE_CHECKPOINT
    return UNet3D(channels=channels, use_checkpoint=use_checkpoint).to(device)


def build_classifier(device="cpu", base_channels=None, dropout: float = 0.4) -> ResNet3D:
    base_channels = base_channels if base_channels is not None else cfg.CLASSIFIER_BASE_CHANNELS
    return ResNet3D(base_channels=base_channels, use_se=True, dropout=dropout).to(device)


def detector_arch() -> dict:
    """Architecture metadata to embed in a detector checkpoint at save time."""
    return {"channels": tuple(cfg.DETECTOR_CHANNELS)}


def classifier_arch() -> dict:
    """Architecture metadata to embed in a classifier checkpoint at save time."""
    return {"base_channels": cfg.CLASSIFIER_BASE_CHANNELS}


# ═══════════════════════════════════════════════════════
# LOAD — reconstruct the model a checkpoint actually needs, then load it
# ═══════════════════════════════════════════════════════

def load_detector_checkpoint(path, device="cpu", use_checkpoint: Optional[bool] = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", {})
    model = build_detector(device=device, channels=arch.get("channels"),
                            use_checkpoint=use_checkpoint)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def load_classifier_checkpoint(path, device="cpu", dropout: float = 0.4):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", {})
    model = build_classifier(device=device, base_channels=arch.get("base_channels"),
                              dropout=dropout)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


# ═══════════════════════════════════════════════════════
# RESUME-TIME SANITY CHECK — fail with a clear message, not a shape-mismatch dump
# ═══════════════════════════════════════════════════════

def warn_on_arch_mismatch(ckpt_arch: dict, current_arch: dict, label: str):
    """
    Compares a checkpoint's recorded architecture against what config.py
    currently says, and prints an actionable warning if they differ. Meant
    to be called BEFORE load_state_dict, so if something's wrong you get
    one clear line instead of a 40-line shape-mismatch traceback.
    """
    if not ckpt_arch:
        return  # older checkpoint saved before "arch" existed — nothing to compare
    mismatches = [f"{k}: checkpoint has {v!r}, config.py currently has {current_arch.get(k)!r}"
                  for k, v in ckpt_arch.items() if current_arch.get(k) != v]
    if mismatches:
        print(f"  ⚠ {label} architecture mismatch — config.py has changed since this "
              f"checkpoint was saved:")
        for m in mismatches:
            print(f"      {m}")
        print(f"      Reconstructing the model to match the CHECKPOINT (not current "
              f"config.py) so this resumes correctly.")
