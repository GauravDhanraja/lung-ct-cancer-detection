"""
data/dataset.py
───────────────
PyTorch Dataset classes for:
  - LunaDetectorDataset  : 64³ patches → Gaussian sphere segmentation labels
  - LunaClassifierDataset: 32³ crops   → binary malignancy labels

Both support 3D augmentation and train/val splitting.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ═══════════════════════════════════════════════════════
# 3D AUGMENTATION UTILS
# ═══════════════════════════════════════════════════════

class Augment3D:
    """
    Composable 3D augmentation for CT patches.
    All ops work on (D, H, W) numpy float32 arrays.
    """

    def __init__(self,
                 flip_prob:     float = cfg.AUG_FLIP_PROB,
                 rotate_max:    float = cfg.AUG_ROTATE_MAX,
                 scale_range:   Tuple = cfg.AUG_SCALE_RANGE,
                 noise_std:     float = cfg.AUG_NOISE_STD,
                 brightness:    float = cfg.AUG_BRIGHTNESS,
                 training:      bool  = True):
        self.flip_prob  = flip_prob
        self.rotate_max = rotate_max
        self.scale_range= scale_range
        self.noise_std  = noise_std
        self.brightness = brightness
        self.training   = training

    def __call__(self, volume: np.ndarray,
                 label: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.training:
            return volume, label

        # ── Random flip (axial + coronal + sagittal) ──
        for axis in range(3):
            if np.random.rand() < self.flip_prob:
                volume = np.flip(volume, axis=axis).copy()
                if label is not None:
                    label = np.flip(label, axis=axis).copy()

        # ── Random rotation (in-plane, around z-axis) ──
        if self.rotate_max > 0 and np.random.rand() > 0.5:
            angle = np.random.uniform(-self.rotate_max, self.rotate_max)
            volume = nd_rotate(volume, angle, axes=(1, 2),
                               reshape=False, order=1, mode='nearest')
            if label is not None:
                label = nd_rotate(label, angle, axes=(1, 2),
                                  reshape=False, order=1, mode='nearest')

        # ── Random scaling ──
        if np.random.rand() > 0.5:
            scale = np.random.uniform(*self.scale_range)
            D, H, W = volume.shape
            scaled = nd_zoom(volume, scale, order=1)
            # Centre-crop or pad back to original size
            volume = self._resize_to(scaled, (D, H, W))
            if label is not None:
                scaled_lbl = nd_zoom(label, scale, order=1)
                label = self._resize_to(scaled_lbl, (D, H, W))

        # ── Gaussian noise ──
        if self.noise_std > 0:
            volume = volume + np.random.randn(*volume.shape).astype(np.float32) \
                              * self.noise_std

        # ── Brightness jitter ──
        if self.brightness > 0:
            volume = volume + np.random.uniform(-self.brightness, self.brightness)

        volume = np.clip(volume, 0.0, 1.0)
        return volume, label

    @staticmethod
    def _resize_to(vol: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """Centre-crop or zero-pad vol to target_shape.
        Handles zero-size dimensions safely by returning zeros."""
        result = np.zeros(target_shape, dtype=vol.dtype)
        # If any dimension is 0, just return zeros — nothing to copy
        if any(s == 0 for s in vol.shape):
            return result
        slices_src = []
        slices_dst = []
        for s_vol, s_tgt in zip(vol.shape, target_shape):
            if s_vol >= s_tgt:
                start = (s_vol - s_tgt) // 2
                slices_src.append(slice(start, start + s_tgt))
                slices_dst.append(slice(0, s_tgt))
            else:
                start = (s_tgt - s_vol) // 2
                slices_src.append(slice(0, s_vol))
                slices_dst.append(slice(start, start + s_vol))
        result[tuple(slices_dst)] = vol[tuple(slices_src)]
        return result


# ═══════════════════════════════════════════════════════
# DETECTOR DATASET
# ═══════════════════════════════════════════════════════

class LunaDetectorDataset(Dataset):
    """
    Loads pre-processed 64³ detector patches from disk.
    Each item:
        volume : Tensor (1, 64, 64, 64) float32
        label  : Tensor (1, 64, 64, 64) float32  (Gaussian sphere or zeros)
        is_nodule: int  (1 / 0)
    """

    def __init__(self,
                 patch_dir: Path = cfg.DETECTOR_PATCHES_DIR,
                 training:  bool = True,
                 val_split: float = 0.15,
                 seed:      int   = cfg.SEED):
        self.training   = training
        self.augment    = Augment3D(training=training)
        self.patch_paths = sorted(patch_dir.glob("*.npz"))

        if not self.patch_paths:
            raise FileNotFoundError(
                f"No .npz patches found in {patch_dir}. "
                "Run data/preprocessing.py first."
            )

        # Reproducible train/val split by file index
        rng = np.random.RandomState(seed)
        idx = np.arange(len(self.patch_paths))
        rng.shuffle(idx)
        n_val = int(len(idx) * val_split)
        val_idx   = set(idx[:n_val].tolist())
        train_idx = set(idx[n_val:].tolist())

        if training:
            self.patch_paths = [self.patch_paths[i] for i in sorted(train_idx)]
        else:
            self.patch_paths = [self.patch_paths[i] for i in sorted(val_idx)]

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx: int):
        data      = np.load(self.patch_paths[idx])
        volume    = data["volume"].astype(np.float32)   # (1, D, H, W)
        label     = data["label"].astype(np.float32)    # (1, D, H, W)
        is_nodule = int(data["is_nodule"])

        # ── Guard: force correct shape (64,64,64) ──
        # Some edge-case patches saved with wrong dims (e.g. [1,0,64,64])
        # due to CT volumes smaller than patch size. Resize to correct shape.
        ps = cfg.DETECTOR_PATCH_SIZE  # (64, 64, 64)
        if volume.shape[1:] != ps:
            volume = Augment3D._resize_to(volume[0], ps)[np.newaxis]
            label  = Augment3D._resize_to(label[0],  ps)[np.newaxis]

        # Augment (squeeze channel for aug, then restore)
        vol_aug, lbl_aug = self.augment(volume[0], label[0])
        volume = vol_aug[np.newaxis]
        label  = lbl_aug[np.newaxis]

        return (torch.from_numpy(volume.copy()),
                torch.from_numpy(label.copy()),
                torch.tensor(is_nodule, dtype=torch.long))

    def get_sampler(self) -> WeightedRandomSampler:
        """Balanced sampler: equal probability for nodule / non-nodule patches."""
        labels = []
        for p in self.patch_paths:
            d = np.load(p)
            labels.append(int(d["is_nodule"]))
        labels = np.array(labels)
        n_pos  = labels.sum()
        n_neg  = len(labels) - n_pos
        w_pos  = 1.0 / max(n_pos, 1)
        w_neg  = 1.0 / max(n_neg, 1)
        weights = np.where(labels == 1, w_pos, w_neg)
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(labels),
            replacement=True
        )


# ═══════════════════════════════════════════════════════
# CLASSIFIER DATASET
# ═══════════════════════════════════════════════════════

class LunaClassifierDataset(Dataset):
    """
    Loads pre-processed 32³ nodule crops from disk.
    Each item:
        volume : Tensor (1, 32, 32, 32) float32
        label  : Tensor scalar int64  (0=benign, 1=malignant)
        meta   : dict with diameter_mm, uid
    """

    def __init__(self,
                 crop_dir:  Path  = cfg.CLASSIFIER_CROPS_DIR,
                 training:  bool  = True,
                 val_split: float = 0.2,
                 seed:      int   = cfg.SEED):
        self.training  = training
        self.augment   = Augment3D(training=training)
        self.crop_paths = sorted(crop_dir.glob("*.npz"))

        if not self.crop_paths:
            raise FileNotFoundError(
                f"No .npz crops found in {crop_dir}. "
                "Run data/preprocessing.py first."
            )

        rng = np.random.RandomState(seed)
        idx = np.arange(len(self.crop_paths))
        rng.shuffle(idx)
        n_val = int(len(idx) * val_split)
        val_idx   = set(idx[:n_val].tolist())
        train_idx = set(idx[n_val:].tolist())

        if training:
            self.crop_paths = [self.crop_paths[i] for i in sorted(train_idx)]
        else:
            self.crop_paths = [self.crop_paths[i] for i in sorted(val_idx)]

    def __len__(self):
        return len(self.crop_paths)

    def __getitem__(self, idx: int):
        data   = np.load(self.crop_paths[idx])
        volume = data["volume"].astype(np.float32)   # (1, 32, 32, 32)
        label  = int(data["label"])
        d_mm   = float(data["diameter_mm"])
        uid    = str(data["uid"])

        # ── Guard: force correct shape (32,32,32) ──
        cs = cfg.CLASSIFIER_CROP_SIZE  # (32, 32, 32)
        if volume.shape[1:] != cs:
            volume = Augment3D._resize_to(volume[0], cs)[np.newaxis]

        vol_aug, _ = self.augment(volume[0], None)
        volume = vol_aug[np.newaxis]

        return (torch.from_numpy(volume.copy()),
                torch.tensor(label, dtype=torch.long),
                {"diameter_mm": d_mm, "uid": uid})

    def get_sampler(self) -> WeightedRandomSampler:
        """Balanced sampler for class imbalance."""
        labels = []
        for p in self.crop_paths:
            d = np.load(p)
            labels.append(int(d["label"]))
        labels = np.array(labels)
        n_pos  = labels.sum()
        n_neg  = len(labels) - n_pos
        w_pos  = 1.0 / max(n_pos, 1)
        w_neg  = 1.0 / max(n_neg, 1)
        weights = np.where(labels == 1, w_pos, w_neg)
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(labels),
            replacement=True
        )


# ═══════════════════════════════════════════════════════
# DATALOADER FACTORY
# ═══════════════════════════════════════════════════════

def get_detector_loaders(
        batch_size: int  = cfg.DETECTOR_BATCH_SIZE,
        num_workers: int = cfg.NUM_WORKERS,
        use_sampler: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_ds = LunaDetectorDataset(training=True)
    val_ds   = LunaDetectorDataset(training=False)

    sampler = train_ds.get_sampler() if use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = num_workers,
        pin_memory  = cfg.PIN_MEMORY,
        drop_last   = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = cfg.PIN_MEMORY
    )
    print(f"Detector  — train: {len(train_ds)}, val: {len(val_ds)}")
    return train_loader, val_loader


def get_classifier_loaders(
        batch_size: int  = cfg.CLASSIFIER_BATCH_SIZE,
        num_workers: int = cfg.NUM_WORKERS,
        use_sampler: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_ds = LunaClassifierDataset(training=True)
    val_ds   = LunaClassifierDataset(training=False)

    sampler = train_ds.get_sampler() if use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = num_workers,
        pin_memory  = cfg.PIN_MEMORY,
        drop_last   = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = cfg.PIN_MEMORY
    )
    print(f"Classifier— train: {len(train_ds)}, val: {len(val_ds)}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════
# SYNTHETIC DATASET FOR UNIT TESTS (no LUNA16 required)
# ═══════════════════════════════════════════════════════

class SyntheticNoduleDataset(Dataset):
    """
    Synthetic dataset that simulates nodule patches for testing.
    Does NOT need LUNA16. Use for rapid architecture debugging.
    """

    def __init__(self, n_samples: int = 200,
                 patch_size: Tuple  = (64, 64, 64),
                 mode: str = "detector"):
        self.n_samples  = n_samples
        self.patch_size = patch_size
        self.mode       = mode   # "detector" or "classifier"

        # Pre-generate to ensure reproducibility
        self.rng = np.random.RandomState(42)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        ps  = self.patch_size
        vol = self.rng.randn(*ps).astype(np.float32) * 0.05 + 0.1  # lung-like

        is_nodule = idx % 2  # alternating
        if is_nodule:
            # Insert a synthetic spherical nodule
            r = self.rng.uniform(3, 10)
            cz = self.rng.randint(int(r)+2, ps[0]-int(r)-2)
            cy = self.rng.randint(int(r)+2, ps[1]-int(r)-2)
            cx = self.rng.randint(int(r)+2, ps[2]-int(r)-2)
            zz, yy, xx = np.ogrid[:ps[0], :ps[1], :ps[2]]
            sphere = ((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2) < r**2
            vol[sphere] = self.rng.uniform(0.5, 0.8)

        vol = np.clip(vol, 0, 1)[np.newaxis]  # (1, D, H, W)

        if self.mode == "detector":
            label = np.zeros((1, *ps), dtype=np.float32)
            if is_nodule:
                # Put a simple sphere label
                from data.preprocessing import make_gaussian_sphere
                blob = make_gaussian_sphere(ps,
                           np.array([ps[0]//2]*3, dtype=float), r)
                label[0] = blob
            return (torch.from_numpy(vol),
                    torch.from_numpy(label),
                    torch.tensor(is_nodule, dtype=torch.long))
        else:
            label = is_nodule
            return (torch.from_numpy(vol),
                    torch.tensor(label, dtype=torch.long),
                    {"diameter_mm": float(is_nodule * 10), "uid": f"syn_{idx}"})


if __name__ == "__main__":
    print("Testing SyntheticNoduleDataset (no LUNA16 needed)...")
    ds = SyntheticNoduleDataset(n_samples=32, mode="detector")
    vol, lbl, is_nod = ds[0]
    print(f"  vol: {vol.shape}, label: {lbl.shape}, is_nodule: {is_nod}")

    ds_cls = SyntheticNoduleDataset(n_samples=32,
                                     patch_size=(32, 32, 32),
                                     mode="classifier")
    vol, lbl, meta = ds_cls[1]
    print(f"  vol: {vol.shape}, label: {lbl}, meta: {meta}")
    print("✓ Dataset classes working.")
