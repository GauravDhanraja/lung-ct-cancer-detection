"""
training/train_detector.py
──────────────────────────
Training loop for the 3D U-Net nodule detector.

Features:
  • Automatic Mixed Precision (FP16)  → fits 4GB VRAM
  • Cosine LR scheduler with warmup
  • Gradient clipping
  • TensorBoard logging
  • Best-model checkpointing by val loss
  • Early stopping
"""

import sys, time, json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from data.dataset import get_detector_loaders, SyntheticNoduleDataset
from models.unet3d import UNet3D, FocalDiceLoss, count_params
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════
# WARMUP + COSINE LR SCHEDULER
# ═══════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch: int):
        import math
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / \
                       max(self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for g, base_lr in zip(self.opt.param_groups, self.base_lrs):
            g['lr'] = max(self.min_lr, base_lr * scale)


# ═══════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════

def batch_metrics(logits: torch.Tensor,
                  labels: torch.Tensor,
                  threshold: float = 0.5) -> Dict:
    """Compute voxel-wise TP, FP, FN for segmentation metrics."""
    probs = torch.sigmoid(logits).detach().cpu().float()
    lbl   = labels.detach().cpu().float()
    pred  = (probs > threshold).float()

    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    tn = ((1 - pred) * (1 - lbl)).sum().item()

    dice   = (2*tp + 1e-5) / (2*tp + fp + fn + 1e-5)
    iou    = (tp + 1e-5) / (tp + fp + fn + 1e-5)
    sens   = (tp + 1e-7) / (tp + fn + 1e-7)
    spec   = (tn + 1e-7) / (tn + fp + 1e-7)
    return {"dice": dice, "iou": iou, "sensitivity": sens, "specificity": spec}


# ═══════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════════════════════

def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                loss_fn: nn.Module,
                scaler: GradScaler,
                device: str) -> Dict:
    model.train()
    total_loss = 0.0
    all_metrics = {"dice": 0., "iou": 0., "sensitivity": 0., "specificity": 0.}
    n_batches  = 0

    bar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for volumes, labels, _ in bar:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device == "cuda")):
            logits = model(volumes)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        m = batch_metrics(logits, labels)
        for k in all_metrics:
            all_metrics[k] += m[k]
        n_batches += 1

        bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.3f}")

    avg = {k: v / max(n_batches, 1) for k, v in all_metrics.items()}
    avg["loss"] = total_loss / max(n_batches, 1)
    return avg


# ═══════════════════════════════════════════════════════
# VALIDATE ONE EPOCH
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def val_epoch(model: nn.Module,
              loader: DataLoader,
              loss_fn: nn.Module,
              device: str) -> Dict:
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": 0., "iou": 0., "sensitivity": 0., "specificity": 0.}
    n_batches  = 0

    for volumes, labels, _ in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=(device == "cuda")):
            logits = model(volumes)
            loss   = loss_fn(logits, labels)

        total_loss += loss.item()
        m = batch_metrics(logits, labels)
        for k in all_metrics:
            all_metrics[k] += m[k]
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in all_metrics.items()}
    avg["loss"] = total_loss / max(n_batches, 1)
    return avg


# ═══════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════

def train_detector(
        use_synthetic:   bool = False,   # True = quick test without LUNA16
        resume_from:     Optional[str] = None,
        epochs:          int  = cfg.DETECTOR_EPOCHS,
        lr:              float = cfg.DETECTOR_LR,
        batch_size:      int  = cfg.DETECTOR_BATCH_SIZE,
        use_checkpoint:  bool = True,    # gradient checkpointing
        device:          str  = "auto"
):
    # ── Setup ──
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  3D U-Net Detector Training")
    print(f"  Device: {device}  |  AMP: {cfg.USE_AMP}  |  Epochs: {epochs}")
    print(f"{'='*60}\n")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # ── Data ──
    if use_synthetic:
        print("⚠  Using SYNTHETIC data (for testing — no LUNA16 required)")
        from torch.utils.data import random_split
        full_ds = SyntheticNoduleDataset(n_samples=400, mode="detector")
        n_val   = 80
        train_ds, val_ds = random_split(full_ds, [320, n_val])
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size*2,
                                   shuffle=False, num_workers=0)
    else:
        train_loader, val_loader = get_detector_loaders(batch_size=batch_size)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ──
    model = UNet3D(use_checkpoint=use_checkpoint).to(device)
    total, trainable = count_params(model)
    print(f"Parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable\n")

    # ── Optimiser & Scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=cfg.DETECTOR_WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, cfg.WARMUP_EPOCHS, epochs)
    loss_fn   = FocalDiceLoss()
    scaler    = GradScaler("cuda", enabled=(cfg.USE_AMP and device == "cuda"))

    # ── Optional resume ──
    start_epoch   = 0
    best_val_dice = 0.0          # monitor dice, not loss
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val dice: {best_val_dice:.4f}")

    # ── Logging ──
    writer   = SummaryWriter(cfg.LOGS_DIR / "detector")
    history  = {"train": [], "val": []}
    patience = 25                # was 15 — give model more time
    patience_counter = 0

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_metrics = train_epoch(model, train_loader, optimizer,
                                     loss_fn, scaler, device)
        val_metrics   = val_epoch(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0

        # TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f"detector/train_{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"detector/val_{k}", v, epoch)
        writer.add_scalar("detector/lr", current_lr, epoch)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Ep {epoch+1:03d}/{epochs}"
              f"  LR={current_lr:.2e}"
              f"  Train: loss={train_metrics['loss']:.4f}  dice={train_metrics['dice']:.3f}"
              f"  Val:   loss={val_metrics['loss']:.4f}  dice={val_metrics['dice']:.3f}"
              f"  ({elapsed:.0f}s)")

        # ── Checkpoint — save when val DICE improves ──
        is_best = val_metrics["dice"] > best_val_dice
        if is_best:
            best_val_dice    = val_metrics["dice"]
            patience_counter = 0
            ckpt = {
                "epoch"         : epoch,
                "model"         : model.state_dict(),
                "optimizer"     : optimizer.state_dict(),
                "val_metrics"   : val_metrics,
                "train_metrics" : train_metrics,
                "best_val_dice" : best_val_dice
            }
            torch.save(ckpt, cfg.CHECKPOINTS_DIR / "detector_best.pth")
            print(f"  ★  Saved best detector (val_dice={best_val_dice:.4f})")
        else:
            patience_counter += 1

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch"        : epoch,
                "model"        : model.state_dict(),
                "best_val_dice": best_val_dice
            }, cfg.CHECKPOINTS_DIR / f"detector_ep{epoch+1}.pth")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    writer.close()

    # Save training history
    with open(cfg.RESULTS_DIR / "detector_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Detector training complete. Best val dice: {best_val_dice:.4f}")
    print(f"  Checkpoint saved: {cfg.CHECKPOINTS_DIR}/detector_best.pth")
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use synthetic data (no LUNA16 needed)")
    parser.add_argument("--epochs",     type=int,   default=cfg.DETECTOR_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=cfg.DETECTOR_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.DETECTOR_LR)
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--device",     type=str,   default="auto")
    args = parser.parse_args()

    train_detector(
        use_synthetic  = args.synthetic,
        resume_from    = args.resume,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
        device         = args.device
    )
