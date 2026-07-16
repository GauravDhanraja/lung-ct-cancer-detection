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

import sys, os, time, json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))         # this dir, for checkpoint_utils
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root, for config/models/data
import config as cfg
from data.dataset import get_detector_loaders, SyntheticNoduleDataset
from models.unet3d import UNet3D, FocalDiceLoss, count_params
from torch.utils.data import DataLoader
from checkpoint_utils import save_checkpoint_async, capture_rng_state, restore_rng_state


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
                device: str,
                accum_steps: int = 1) -> Dict:
    model.train()
    total_loss = 0.0
    all_metrics = {"dice": 0., "iou": 0., "sensitivity": 0., "specificity": 0.}
    n_batches  = 0

    bar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    optimizer.zero_grad(set_to_none=True)
    n_steps = len(loader)
    for i, (volumes, labels, _) in enumerate(bar):
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=(device == "cuda")):
            logits = model(volumes)
            loss   = loss_fn(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        is_last = (i + 1) == n_steps
        if (i + 1) % accum_steps == 0 or is_last:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        step_loss = loss.item() * accum_steps   # undo the accum scaling for logging
        total_loss += step_loss
        m = batch_metrics(logits, labels)
        for k in all_metrics:
            all_metrics[k] += m[k]
        n_batches += 1

        bar.set_postfix(loss=f"{step_loss:.4f}", dice=f"{m['dice']:.3f}")

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
        use_checkpoint:  bool = cfg.DETECTOR_USE_CHECKPOINT,  # gradient checkpointing
        accum_steps:     int  = cfg.DETECTOR_GRAD_ACCUM_STEPS,
        device:          str  = "auto"
):
    # ── Setup ──
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK  # fixed patch
        # size per stage, so letting cuDNN autotune conv algorithms is safe
        # and typically gives a meaningful speedup on Ampere/Blackwell cards

    print(f"\n{'='*60}")
    print(f"  3D U-Net Detector Training")
    print(f"  Device: {device}  |  AMP: {cfg.USE_AMP}  |  Epochs: {epochs}")
    print(f"  Batch: {batch_size}  (x{accum_steps} accum = "
          f"{batch_size*accum_steps} effective)  |  Checkpointing: {use_checkpoint}")
    print(f"{'='*60}\n")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # ── Data ──
    if use_synthetic:
        print("⚠  Using SYNTHETIC data (for testing — no LUNA16 required)")
        from torch.utils.data import random_split
        full_ds = SyntheticNoduleDataset(n_samples=400, mode="detector",
                                          patch_size=cfg.DETECTOR_PATCH_SIZE)
        n_val   = 80
        train_ds, val_ds = random_split(full_ds, [320, n_val])
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
        val_batch = int(batch_size * cfg.DETECTOR_VAL_BATCH_MULTIPLIER)
        val_loader   = DataLoader(val_ds,   batch_size=val_batch,
                                   shuffle=False, num_workers=0)
    else:
        train_loader, val_loader = get_detector_loaders(batch_size=batch_size)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Windows worker-respawn diagnostic ──
    # If num_workers > 0 and persistent_workers is False (the PyTorch
    # default), every DataLoader worker process is torn down at the end of
    # each epoch and respawned at the start of the next. On Windows this
    # uses the (slow) 'spawn' start method — each worker re-imports your
    # training script and rebuilds the dataset object — which can easily
    # cost several seconds to tens of seconds per epoch, right at the
    # boundary where it's easy to mistake it for "the save is slow".
    for name, loader in [("train_loader", train_loader), ("val_loader", val_loader)]:
        nw = getattr(loader, "num_workers", 0)
        persistent = getattr(loader, "persistent_workers", False)
        if nw > 0 and not persistent:
            print(f"  ⚠ {name} has num_workers={nw} but persistent_workers=False — "
                  f"on Windows this respawns all {nw} worker processes every "
                  f"epoch. Add persistent_workers=True to its DataLoader in "
                  f"data/dataset.py (requires num_workers > 0, which you have).")

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
    start_epoch      = 0
    best_val_dice    = 0.0          # monitor dice, not loss
    patience_counter = 0
    history          = {"train": [], "val": []}
    history_path     = cfg.RESULTS_DIR / "detector_history.json"

    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_dice    = ckpt.get("best_val_dice", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        restore_rng_state(ckpt.get("rng_state"))
        print(f"Resumed from epoch {start_epoch}, best val dice: {best_val_dice:.4f}, "
              f"patience_counter: {patience_counter}")

        # Reload prior history so detector_history.json is a continuous
        # record across resumes instead of being overwritten from epoch 0.
        if history_path.exists():
            with open(history_path) as f:
                prior = json.load(f)
            # Trim to exactly the epochs this checkpoint actually covers —
            # guards against resuming from an older/non-latest checkpoint
            # while a longer history.json from a later run still exists.
            history["train"] = prior.get("train", [])[:start_epoch]
            history["val"]   = prior.get("val", [])[:start_epoch]

    # ── Logging ──
    writer   = SummaryWriter(cfg.LOGS_DIR / "detector")
    patience = 25                # was 15 — give model more time
    save_thread = None           # tracks the in-flight background checkpoint write

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        try:
            train_metrics = train_epoch(model, train_loader, optimizer,
                                         loss_fn, scaler, device, accum_steps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM at epoch {epoch+1}. Try, in order: (1) set "
                    f"use_checkpoint=True if it's currently False, (2) lower "
                    f"DETECTOR_BATCH_SIZE (currently {batch_size}) and raise "
                    f"DETECTOR_GRAD_ACCUM_STEPS to compensate, (3) lower "
                    f"DETECTOR_PATCH_SIZE in config.py."
                ) from e
            raise
        t_train = time.time()

        val_metrics   = val_epoch(model, val_loader, loss_fn, device)
        t_val = time.time()

        train_s, val_s = t_train - t0, t_val - t_train

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
              f"  (train={train_s:.0f}s val={val_s:.0f}s)")

        # ── Checkpoint — save when val DICE improves (async, non-blocking) ──
        is_best = val_metrics["dice"] > best_val_dice
        if is_best:
            best_val_dice    = val_metrics["dice"]
            patience_counter = 0
        else:
            patience_counter += 1

        # Common resumable state, shared by the "best" and "last" checkpoints
        resumable_ckpt = {
            "epoch"           : epoch,
            "model"           : model.state_dict(),
            "optimizer"       : optimizer.state_dict(),
            "scaler"          : scaler.state_dict(),
            "val_metrics"     : val_metrics,
            "train_metrics"   : train_metrics,
            "best_val_dice"   : best_val_dice,
            "patience_counter": patience_counter,
            "rng_state"       : capture_rng_state(),
        }

        # detector_last.pth — overwritten EVERY epoch. This is what --resume
        # should point at: it always reflects the most recent completed
        # epoch, regardless of whether that epoch was "best". Without this,
        # resuming after a non-improving stretch would silently roll you
        # back to the last epoch that happened to improve val dice, which
        # can be many epochs behind where you actually stopped.
        t_save0 = time.time()
        save_thread = save_checkpoint_async(
            resumable_ckpt, cfg.CHECKPOINTS_DIR / "detector_last.pth", save_thread)
        save_msg = f"main-thread cost: {time.time()-t_save0:.1f}s"

        # detector_best.pth — only overwritten on improvement. This is the
        # one to load for inference/deployment, not for resuming training.
        if is_best:
            save_thread = save_checkpoint_async(
                resumable_ckpt, cfg.CHECKPOINTS_DIR / "detector_best.pth", save_thread)
            print(f"  ★  Saving best detector (val_dice={best_val_dice:.4f}) "
                  f"in background — {save_msg}")

        # Periodic named snapshot every 10 epochs — a rollback point that
        # detector_last.pth being overwritten every epoch can't provide.
        if (epoch + 1) % 10 == 0:
            save_thread = save_checkpoint_async(
                resumable_ckpt, cfg.CHECKPOINTS_DIR / f"detector_ep{epoch+1}.pth", save_thread)

        # Write history every epoch (not just at the end) so a hard kill —
        # exactly the "shut the machine down" scenario — doesn't lose it.
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    # Make sure the last checkpoint write has actually landed on disk before
    # we report "done" or return control to the caller.
    if save_thread is not None:
        save_thread.join()

    writer.close()

    print(f"\n✓ Detector training complete. Best val dice: {best_val_dice:.4f}")
    print(f"  Best checkpoint (for inference): {cfg.CHECKPOINTS_DIR}/detector_best.pth")
    print(f"  Latest checkpoint (for resume):  {cfg.CHECKPOINTS_DIR}/detector_last.pth")
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use synthetic data (no LUNA16 needed)")
    parser.add_argument("--epochs",     type=int,   default=cfg.DETECTOR_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=cfg.DETECTOR_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.DETECTOR_LR)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to a checkpoint, or 'auto' to resume from "
                             "checkpoints/detector_last.pth if it exists")
    parser.add_argument("--accum-steps", type=int,  default=cfg.DETECTOR_GRAD_ACCUM_STEPS,
                        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable gradient checkpointing (faster, uses more VRAM)")
    parser.add_argument("--device",     type=str,   default="auto")
    args = parser.parse_args()

    resume_path = args.resume
    if resume_path == "auto":
        auto_path = cfg.CHECKPOINTS_DIR / "detector_last.pth"
        if auto_path.exists():
            resume_path = str(auto_path)
            print(f"--resume auto -> found {auto_path}")
        else:
            print(f"--resume auto -> no checkpoint at {auto_path}, starting fresh")
            resume_path = None

    train_detector(
        use_synthetic  = args.synthetic,
        resume_from    = resume_path,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
        use_checkpoint = cfg.DETECTOR_USE_CHECKPOINT and not args.no_checkpoint,
        accum_steps    = args.accum_steps,
        device         = args.device
    )
