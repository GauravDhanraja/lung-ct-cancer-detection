"""
models/resnet3d.py
──────────────────
Lightweight 3D ResNet-10 for malignancy classification.

Architecture:
  Stem      : 7×7×7 conv, 32 channels, stride 2
  Stage 1-4 : BasicBlock pairs  (32→64→128→256)
  Global Avg Pool → FC → sigmoid

Why ResNet-10?
  • Fits easily in 4GB VRAM with batch 8 on 32³ crops
  • Fast to train (< 5M params)
  • Proven effective for 3D medical image classification
  • Easy to extract features from for Grad-CAM

References:
  He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
  Tran et al. "Learning Spatiotemporal Features with 3D CNNs" (ICCV 2015)
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ═══════════════════════════════════════════════════════
# BASIC RESIDUAL BLOCK
# ═══════════════════════════════════════════════════════

class BasicBlock3D(nn.Module):
    """
    Two 3×3×3 conv layers with a skip connection.
    Downsampling (stride=2) applied at first conv + projection shortcut.
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                                padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1,
                                padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Projection shortcut when dims change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# ═══════════════════════════════════════════════════════
# SQUEEZE-AND-EXCITATION (channel attention)
# ═══════════════════════════════════════════════════════

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise recalibration.
    Adds ~2% more params but often +1-2% AUC for nodule classification.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SEBasicBlock3D(BasicBlock3D):
    """BasicBlock3D with channel SE attention after second conv."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 se_reduction: int = 8):
        super().__init__(in_ch, out_ch, stride)
        self.se = SEBlock3D(out_ch, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)              # channel attention
        out += self.shortcut(x)
        return self.relu(out)


# ═══════════════════════════════════════════════════════
# 3D ResNet-10
# ═══════════════════════════════════════════════════════

class ResNet3D(nn.Module):
    """
    Lightweight 3D ResNet for malignancy classification.

    Input  : (B, 1, 32, 32, 32) normalised CT crop
    Output : (B, 1) raw logit  (apply sigmoid for probability)

    Architecture (ResNet-10):
      Stem    →  16³   (32 ch, stride-2 conv)
      Stage 1 →  16³   (64 ch,  1 block)
      Stage 2 →   8³   (128 ch, 1 block, stride-2)
      Stage 3 →   4³   (256 ch, 1 block, stride-2)
      GAP  → FC → 1 logit
    """

    def __init__(self,
                 in_channels:    int   = 1,
                 base_channels:  int   = 32,
                 num_classes:    int   = cfg.NUM_CLASSES,
                 use_se:         bool  = True,
                 dropout:        float = 0.2):
        super().__init__()
        bc  = base_channels
        Block = SEBasicBlock3D if use_se else BasicBlock3D

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, bc, kernel_size=5, stride=2,
                      padding=2, bias=False),   # 32³ → 16³
            nn.BatchNorm3d(bc),
            nn.ReLU(inplace=True)
        )

        # ── Residual stages ──
        self.layer1 = Block(bc,     bc*2,  stride=1)   # 16³  → 64ch
        self.layer2 = Block(bc*2,   bc*4,  stride=2)   # 8³   → 128ch
        self.layer3 = Block(bc*4,   bc*8,  stride=2)   # 4³   → 256ch

        # ── Head ──
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(bc * 8, num_classes)

        # ── Feature dim (for Grad-CAM) ──
        self.feature_dim = bc * 8

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the final feature map (before GAP) — used by Grad-CAM.
        Returns: (B, C, d, h, w)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.forward_features(x)
        pooled = self.gap(feats).flatten(1)     # (B, C)
        pooled = self.dropout(pooled)
        return self.fc(pooled)                  # (B, 1) logit


# ═══════════════════════════════════════════════════════
# CLASSIFIER LOSS
# ═══════════════════════════════════════════════════════

class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy with positive class weight.
    Addresses class imbalance (malignant << benign in LUNA16).
    """

    def __init__(self, pos_weight: float = cfg.POS_WEIGHT):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets,
                                                   pos_weight=pw)


class LabelSmoothingBCE(nn.Module):
    """
    BCEWithLogits + label smoothing — reduces overconfidence.
    Especially useful when malignancy labels are uncertain.
    """

    def __init__(self, smoothing: float = 0.05,
                 pos_weight: float = cfg.POS_WEIGHT):
        super().__init__()
        self.smoothing  = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets,
                                                   pos_weight=pw)


# ═══════════════════════════════════════════════════════
# MODEL INFO
# ═══════════════════════════════════════════════════════

def count_params(model: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("3D ResNet-10 Classifier — Sanity Check")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ResNet3D(use_se=True, dropout=0.4).to(device)
    loss_fn = LabelSmoothingBCE()

    total, trainable = count_params(model)
    print(f"Parameters: {total/1e6:.3f}M total, {trainable/1e6:.3f}M trainable")

    # Forward pass
    batch  = torch.randn(8, 1, 32, 32, 32).to(device)
    labels = torch.tensor([0,1,0,1,0,1,0,1], dtype=torch.long).to(device)

    from torch.cuda.amp import autocast
    with autocast(enabled=device=="cuda"):
        logits = model(batch)
        loss   = loss_fn(logits, labels)
        probs  = torch.sigmoid(logits)

    print(f"Input:  {batch.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Output probs:  {probs.squeeze().tolist()}")
    print(f"Loss:   {loss.item():.4f}")

    # Feature map (for Grad-CAM)
    feats = model.forward_features(batch)
    print(f"Feature map (for Grad-CAM): {feats.shape}")

    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {mem:.3f} GB")

    print("✓ ResNet3D functional.")
