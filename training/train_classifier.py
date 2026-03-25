"""
training/train_classifier.py
────────────────────────────
Training loop for the 3D ResNet-10 malignancy classifier.

Key additions over the detector trainer:
  • AUC-ROC monitoring (primary metric for clinical relevance)
  • Mixup augmentation for 3D volumes (reduces overfitting on small datasets)
  • Test-Time Augmentation (TTA) for robust probability estimation
"""

import sys, time, json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from data.dataset import get_classifier_loaders, SyntheticNoduleDataset
from models.resnet3d import ResNet3D, LabelSmoothingBCE, count_params


# ═══════════════════════════════════════════════════════
# MIXUP AUGMENTATION (3D)
# ═══════════════════════════════════════════════════════

def mixup_batch(volumes: torch.Tensor,
                labels:  torch.Tensor,
                alpha:   float = 0.2):
    """
    Mixup: interpolate pairs of samples to create virtual training examples.
    λ ~ Beta(alpha, alpha);  x_mixed = λ·x_a + (1-λ)·x_b
    """
    if alpha <= 0:
        return volumes, labels.float(), labels.float(), 1.0

    lam  = np.random.beta(alpha, alpha)
    B    = volumes.size(0)
    perm = torch.randperm(B, device=volumes.device)

    vol_a, vol_b = volumes, volumes[perm]
    lbl_a, lbl_b = labels.float(), labels[perm].float()

    mixed_vol = lam * vol_a + (1 - lam) * vol_b
    return mixed_vol, lbl_a, lbl_b, lam


def mixup_loss(criterion, logits, lbl_a, lbl_b, lam):
    return lam * criterion(logits, lbl_a) + (1 - lam) * criterion(logits, lbl_b)


# ═══════════════════════════════════════════════════════
# WARMUP COSINE SCHEDULER (reused)
# ═══════════════════════════════════════════════════════

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        import math
        if epoch < self.warmup:
            scale = (epoch + 1) / self.warmup
        else:
            p = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * p))
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g['lr'] = max(self.min_lr, base * scale)


# ═══════════════════════════════════════════════════════
# TEST-TIME AUGMENTATION
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module,
                volume: torch.Tensor,
                n_augments: int = 8,
                device: str = "cuda") -> float:
    """
    Average predictions over multiple augmented views of one 3D volume.
    volume: (1, D, H, W) tensor.
    Returns mean sigmoid probability.
    """
    model.eval()
    probs = []
    vol_np = volume.squeeze(0).cpu().numpy()   # (D, H, W)

    for _ in range(n_augments):
        aug = vol_np.copy()
        # Random flips
        for ax in range(3):
            if np.random.rand() > 0.5:
                aug = np.flip(aug, axis=ax)
        # Random 90° rotations
        k = np.random.randint(4)
        aug = np.rot90(aug, k, axes=(1, 2))

        aug_t = torch.from_numpy(aug.copy()).unsqueeze(0).unsqueeze(0).to(device)
        with autocast("cuda", enabled=(device == "cuda")):
            logit = model(aug_t)
        probs.append(torch.sigmoid(logit).item())

    return float(np.mean(probs))


# ═══════════════════════════════════════════════════════
# TRAIN / VAL EPOCHS
# ═══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, loss_fn, scaler,
                device, use_mixup=True) -> Dict:
    model.train()
    total_loss = 0.
    all_logits, all_labels = [], []
    n_batches = 0

    bar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for volumes, labels, _ in bar:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        # Mixup
        if use_mixup and np.random.rand() > 0.5:
            mixed_vol, lbl_a, lbl_b, lam = mixup_batch(volumes, labels)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device == "cuda")):
                logits = model(mixed_vol)
                loss   = mixup_loss(loss_fn, logits, lbl_a, lbl_b, lam)
        else:
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
        all_logits.append(logits.detach().cpu().float().sigmoid().squeeze())
        all_labels.append(labels.detach().cpu().float())
        n_batches += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auc = roc_auc_score(all_labels, all_logits)
        ap  = average_precision_score(all_labels, all_logits)
    except Exception:
        auc = ap = 0.5

    return {
        "loss": total_loss / max(n_batches, 1),
        "auc" : auc,
        "ap"  : ap
    }


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device) -> Dict:
    model.eval()
    total_loss = 0.
    all_probs, all_labels = [], []
    n_batches = 0

    for volumes, labels, _ in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=(device == "cuda")):
            logits = model(volumes)
            loss   = loss_fn(logits, labels)

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().float().squeeze())
        all_labels.append(labels.cpu().float())
        n_batches += 1

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs)
        ap  = average_precision_score(all_labels, all_probs)
        preds = (all_probs > 0.5).astype(int)
        cm    = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv  = tp / max(tp + fp, 1)
        npv  = tn / max(tn + fn, 1)
    except Exception as e:
        auc = ap = sens = spec = ppv = npv = 0.5

    return {
        "loss"       : total_loss / max(n_batches, 1),
        "auc"        : auc,
        "ap"         : ap,
        "sensitivity": sens,
        "specificity": spec,
        "ppv"        : ppv,
        "npv"        : npv,
        "probs"      : all_probs.tolist(),
        "labels"     : all_labels.tolist()
    }


# ═══════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════

def train_classifier(
        use_synthetic:  bool  = False,
        resume_from:    Optional[str] = None,
        epochs:         int   = cfg.CLASSIFIER_EPOCHS,
        lr:             float = cfg.CLASSIFIER_LR,
        batch_size:     int   = cfg.CLASSIFIER_BATCH_SIZE,
        use_mixup:      bool  = True,
        device:         str   = "auto"
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  3D ResNet-10 Classifier Training")
    print(f"  Device: {device}  |  Mixup: {use_mixup}  |  Epochs: {epochs}")
    print(f"{'='*60}\n")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # ── Data ──
    if use_synthetic:
        print("⚠  Using SYNTHETIC data")
        from torch.utils.data import random_split
        full_ds = SyntheticNoduleDataset(n_samples=300,
                                          patch_size=(32,32,32),
                                          mode="classifier")
        train_ds, val_ds = random_split(full_ds, [240, 60])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=0)
    else:
        train_loader, val_loader = get_classifier_loaders(batch_size=batch_size)

    # ── Model ──
    model   = ResNet3D(use_se=True, dropout=0.4).to(device)
    loss_fn = LabelSmoothingBCE(smoothing=0.05)
    total, trainable = count_params(model)
    print(f"Parameters: {total/1e6:.3f}M\n")

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=cfg.CLASSIFIER_WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, cfg.WARMUP_EPOCHS, epochs)
    scaler    = GradScaler("cuda", enabled=(cfg.USE_AMP and device == "cuda"))

    # Resume
    start_epoch  = 0
    best_val_auc = 0.0
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC: {best_val_auc:.4f}")

    writer  = SummaryWriter(cfg.LOGS_DIR / "classifier")
    history = {"train": [], "val": []}
    patience, patience_counter = 20, 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        scheduler.step(epoch)
        lr_now = optimizer.param_groups[0]['lr']

        train_m = train_epoch(model, train_loader, optimizer, loss_fn,
                               scaler, device, use_mixup)
        val_m   = val_epoch(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0

        for k, v in train_m.items():
            if not isinstance(v, list):
                writer.add_scalar(f"classifier/train_{k}", v, epoch)
        for k, v in val_m.items():
            if not isinstance(v, list):
                writer.add_scalar(f"classifier/val_{k}", v, epoch)
        writer.add_scalar("classifier/lr", lr_now, epoch)

        # Don't save raw prob/label arrays in history
        train_log = {k: v for k, v in train_m.items() if not isinstance(v, list)}
        val_log   = {k: v for k, v in val_m.items()   if not isinstance(v, list)}
        history["train"].append(train_log)
        history["val"].append(val_log)

        print(f"Ep {epoch+1:03d}/{epochs}"
              f"  LR={lr_now:.2e}"
              f"  Train: loss={train_m['loss']:.4f}  AUC={train_m['auc']:.3f}"
              f"  Val:   loss={val_m['loss']:.4f}  AUC={val_m['auc']:.3f}"
              f"  Sens={val_m['sensitivity']:.3f}  Spec={val_m['specificity']:.3f}"
              f"  ({elapsed:.0f}s)")

        is_best = val_m["auc"] > best_val_auc
        if is_best:
            best_val_auc     = val_m["auc"]
            patience_counter = 0
            torch.save({
                "epoch"       : epoch,
                "model"       : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_metrics" : val_log,
                "best_val_auc": best_val_auc
            }, cfg.CHECKPOINTS_DIR / "classifier_best.pth")
            print(f"  ★  Saved best classifier (AUC={best_val_auc:.4f})")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       cfg.CHECKPOINTS_DIR / f"classifier_ep{epoch+1}.pth")

        if patience_counter >= patience:
            print(f"\nEarly stopping (no AUC improvement for {patience} epochs)")
            break

    writer.close()
    with open(cfg.RESULTS_DIR / "classifier_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Classifier training complete. Best val AUC: {best_val_auc:.4f}")
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic",   action="store_true")
    parser.add_argument("--epochs",      type=int,   default=cfg.CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=cfg.CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=cfg.CLASSIFIER_LR)
    parser.add_argument("--no-mixup",    action="store_true")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--device",      type=str,   default="auto")
    args = parser.parse_args()

    train_classifier(
        use_synthetic = args.synthetic,
        resume_from   = args.resume,
        epochs        = args.epochs,
        lr            = args.lr,
        batch_size    = args.batch_size,
        use_mixup     = not args.no_mixup,
        device        = args.device
    )
