"""
data/preprocessing.py
─────────────────────
LUNA16 CT preprocessing pipeline.

Steps:
  1. Load .mhd/.raw CT volumes with SimpleITK
  2. Resample to 1×1×1 mm isotropic spacing
  3. Clip & normalise HU values → [0, 1]
  4. Extract lung mask (thresholding + morphology)
  5. Save patches + labels for detector / classifier

LUNA16 download: https://luna16.grand-challenge.org/
  Place subsets in:  data/LUNA16/subset{0-9}/
  CSV files in:      data/LUNA16/annotations.csv
                     data/LUNA16/candidates.csv
"""

import os
import sys
import csv
import pickle
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label as scipy_label, binary_fill_holes
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════

def load_mhd(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a .mhd file.
    Returns:
        volume  : (Z, Y, X) float32 array in HU
        origin  : (x, y, z) world origin in mm
        spacing : (x, y, z) voxel spacing in mm
    """
    itk_img = sitk.ReadImage(str(path))
    volume  = sitk.GetArrayFromImage(itk_img)          # (Z, Y, X)
    origin  = np.array(list(itk_img.GetOrigin()))      # (x, y, z)
    spacing = np.array(list(itk_img.GetSpacing()))     # (x, y, z)
    return volume.astype(np.float32), origin, spacing


def world_to_voxel(world_coord: np.ndarray,
                   origin: np.ndarray,
                   spacing: np.ndarray) -> np.ndarray:
    """Convert (x,y,z) mm → (z,y,x) voxel index."""
    stretched = np.abs(world_coord - origin)
    voxel_xyz = stretched / spacing
    return voxel_xyz[::-1]     # (z, y, x)


def voxel_to_world(voxel_coord: np.ndarray,
                   origin: np.ndarray,
                   spacing: np.ndarray) -> np.ndarray:
    """Convert (z,y,x) voxel → (x,y,z) mm."""
    voxel_xyz = voxel_coord[::-1]  # (x, y, z)
    return voxel_xyz * spacing + origin


# ═══════════════════════════════════════════════════════
# RESAMPLING
# ═══════════════════════════════════════════════════════

def resample_volume(volume: np.ndarray,
                    spacing: np.ndarray,
                    new_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample CT volume to isotropic spacing using B-spline interpolation.
    volume  : (Z, Y, X)
    spacing : (x, y, z) order (SimpleITK convention)
    Returns resampled volume and resize_factor for coordinate transform.
    """
    # spacing is (x, y, z) but volume is (Z, Y, X)
    spacing_zyx = spacing[::-1]        # (z, y, x)
    new_spacing_zyx = np.array(new_spacing)[::-1]

    resize_factor = spacing_zyx / new_spacing_zyx
    new_shape = np.round(volume.shape * resize_factor).astype(int)

    itk_vol = sitk.GetImageFromArray(volume)
    itk_vol.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(s) for s in new_spacing])
    resampler.SetSize([int(new_shape[2]), int(new_shape[1]), int(new_shape[0])])
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputOrigin(itk_vol.GetOrigin())
    resampler.SetOutputDirection(itk_vol.GetDirection())

    resampled_itk = resampler.Execute(itk_vol)
    resampled = sitk.GetArrayFromImage(resampled_itk).astype(np.float32)

    # Actual resize factor after rounding
    actual_factor = np.array(resampled.shape) / np.array(volume.shape)
    return resampled, actual_factor          # actual_factor is (z, y, x)


# ═══════════════════════════════════════════════════════
# HU NORMALISATION
# ═══════════════════════════════════════════════════════

def normalise_hu(volume: np.ndarray,
                 hu_min: float = cfg.HU_MIN,
                 hu_max: float = cfg.HU_MAX) -> np.ndarray:
    """Clip HU range and normalise to [0, 1]."""
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


# ═══════════════════════════════════════════════════════
# LUNG SEGMENTATION (no deep learning — pure morphology)
# ═══════════════════════════════════════════════════════

def segment_lung_mask(volume_hu: np.ndarray) -> np.ndarray:
    """
    Simple threshold + connected-component lung mask.
    Works well enough for patch extraction; not used for clinical segmentation.
    Returns binary mask (Z, Y, X).
    """
    # Threshold at -400 HU (air + lung parenchyma)
    binary = volume_hu < -400

    # Label connected components
    labels, n_labels = scipy_label(binary)
    if n_labels == 0:
        return np.ones_like(binary, dtype=bool)

    # The two largest components after removing background (label=0) are lungs
    counts = [(labels == i).sum() for i in range(1, n_labels + 1)]
    sorted_labels = np.argsort(counts)[::-1] + 1   # descending by size

    mask = np.zeros_like(binary, dtype=bool)
    for lbl in sorted_labels[:2]:
        mask |= (labels == lbl)

    # Fill holes slice-by-slice
    for z in range(mask.shape[0]):
        mask[z] = binary_fill_holes(mask[z])

    return mask


# ═══════════════════════════════════════════════════════
# PATCH / CROP EXTRACTION
# ═══════════════════════════════════════════════════════

def extract_patch(volume: np.ndarray,
                  centre_zyx: np.ndarray,
                  patch_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Extract a 3-D patch centred at centre_zyx.
    Pads with zeros (air, after normalisation ~0) if near boundary.
    """
    pz, py, px = patch_size
    cz, cy, cx = np.round(centre_zyx).astype(int)
    Z, Y, X = volume.shape

    # Half-sizes
    hz, hy, hx = pz // 2, py // 2, px // 2

    pad_vol = np.pad(volume,
                     ((hz, hz), (hy, hy), (hx, hx)),
                     mode='constant', constant_values=0.0)

    # After padding, centre shifts by (hz, hy, hx)
    cz_p, cy_p, cx_p = cz + hz, cy + hy, cx + hx

    patch = pad_vol[cz_p - hz : cz_p + hz + (pz % 2),
                    cy_p - hy : cy_p + hy + (py % 2),
                    cx_p - hx : cx_p + hx + (px % 2)]
    return patch


def make_gaussian_sphere(shape: Tuple[int, int, int],
                          centre_zyx: np.ndarray,
                          radius_vox: float,
                          sigma_ratio: float = cfg.GAUSSIAN_SIGMA_RATIO
                          ) -> np.ndarray:
    """
    Create a soft Gaussian blob label for a single nodule.
    Values range [0, 1], peak = 1 at centre, falls off with Gaussian.
    """
    sigma = max(radius_vox * sigma_ratio, 1.0)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = centre_zyx
    dist_sq = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
    blob = np.exp(-dist_sq / (2 * sigma**2))
    # Hard threshold: outside 1.5× radius → 0
    blob[dist_sq > (1.5 * radius_vox)**2] = 0.0
    return blob.astype(np.float32)


# ═══════════════════════════════════════════════════════
# LUNA16 SPECIFIC PROCESSING
# ═══════════════════════════════════════════════════════

def find_mhd_file(seriesuid: str) -> Optional[Path]:
    """Search all LUNA16 subsets for a given seriesuid."""
    for subset_dir in cfg.SUBSET_DIRS:
        path = subset_dir / f"{seriesuid}.mhd"
        if path.exists():
            return path
    return None


def load_annotations() -> pd.DataFrame:
    """
    Load LUNA16 annotations.csv.
    Columns: seriesuid, coordX, coordY, coordZ, diameter_mm
    """
    return pd.read_csv(cfg.ANNOTATIONS_CSV)


def load_candidates() -> pd.DataFrame:
    """
    Load LUNA16 candidates.csv.
    Columns: seriesuid, coordX, coordY, coordZ, class (1=nodule, 0=non-nodule)
    """
    return pd.read_csv(cfg.CANDIDATES_CSV)


# ═══════════════════════════════════════════════════════
# MAIN PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════

class LUNA16Preprocessor:
    """
    Full preprocessing pipeline for LUNA16.

    Generates:
      - Detector patches (64³) with Gaussian sphere labels
      - Classifier crops (32³) with binary malignancy labels
    """

    def __init__(self):
        self.annotations = load_annotations()
        self.candidates  = load_candidates()
        # Group by scan
        self.ann_by_uid  = self.annotations.groupby("seriesuid")
        self.cand_by_uid = self.candidates.groupby("seriesuid")

    # ────────────────────────────────
    # SCAN-LEVEL PROCESSING
    # ────────────────────────────────

    def process_scan(self, seriesuid: str) -> Optional[Dict]:
        """Process one CT scan end-to-end. Returns metadata dict."""
        mhd_path = find_mhd_file(seriesuid)
        if mhd_path is None:
            return None

        # 1. Load
        volume_hu, origin, spacing = load_mhd(mhd_path)

        # 2. Resample to 1 mm isotropic
        volume_1mm, resize_factor = resample_volume(volume_hu, spacing,
                                                    cfg.TARGET_SPACING)
        # Update effective spacing
        new_spacing_xyz = np.array(cfg.TARGET_SPACING)

        # 3. Normalise HU
        volume_norm = normalise_hu(volume_1mm)

        # 4. Nodule annotations in voxel space (resampled)
        nodules_vox = []
        if seriesuid in self.ann_by_uid.groups:
            for _, row in self.ann_by_uid.get_group(seriesuid).iterrows():
                world_xyz = np.array([row.coordX, row.coordY, row.coordZ])
                vox_zyx_orig = world_to_voxel(world_xyz, origin, spacing[::-1])
                # Scale by resize_factor (z,y,x)
                vox_zyx_new = vox_zyx_orig * resize_factor
                radius_mm   = row.diameter_mm / 2.0
                radius_vox  = radius_mm / 1.0  # isotropic 1mm
                nodules_vox.append({
                    "zyx"       : vox_zyx_new,
                    "radius_vox": radius_vox,
                    "diameter_mm": row.diameter_mm
                })

        return {
            "seriesuid"   : seriesuid,
            "volume"      : volume_norm,
            "origin"      : origin,
            "spacing"     : new_spacing_xyz,
            "resize_factor": resize_factor,
            "nodules"     : nodules_vox
        }

    # ────────────────────────────────
    # DETECTOR PATCH GENERATION
    # ────────────────────────────────

    def generate_detector_patches(self, scan_info: Dict):
        vol   = scan_info["volume"]
        nods  = scan_info["nodules"]
        uid   = scan_info["seriesuid"]
        origin        = scan_info["origin"]
        resize_factor = scan_info["resize_factor"]
        ps    = cfg.DETECTOR_PATCH_SIZE

        # ── Positive patches ──
        for nod in nods:
            czyx        = nod["zyx"]
            radius_vox  = nod["radius_vox"]

            patch_vol = extract_patch(vol, czyx, ps)
            centre = np.array([ps[0]//2, ps[1]//2, ps[2]//2], dtype=float)
            label = make_gaussian_sphere(ps, centre, radius_vox)

            yield {
                "volume": patch_vol[np.newaxis],
                "label": label[np.newaxis],
                "uid": uid,
                "is_nodule": 1,
                "centre_zyx": czyx,
                "radius_vox": radius_vox
            }

        # ── Hard negatives ──
        n_neg  = max(len(nods) * 3, 6)

        Z, Y, X = vol.shape
        hz, hy, hx = ps[0]//2, ps[1]//2, ps[2]//2
        hard_neg_coords = []

        if uid in self.cand_by_uid.groups:
            cand_df = self.cand_by_uid.get_group(uid)
            fp_df = cand_df[cand_df["class"] == 0]

            for _, row in fp_df.iterrows():
                world_xyz = np.array([row.coordX, row.coordY, row.coordZ])

                vox_zyx_orig = world_to_voxel(
                    world_xyz, origin, np.array(cfg.TARGET_SPACING)[::-1]
                )
                vox_zyx = vox_zyx_orig * resize_factor

                cz, cy, cx = vox_zyx
                if (hz <= cz < Z - hz and
                    hy <= cy < Y - hy and
                    hx <= cx < X - hx):

                    too_close = any(
                        np.linalg.norm(vox_zyx - nod["zyx"]) < nod["radius_vox"] * 2
                        for nod in nods
                    )

                    if not too_close:
                        hard_neg_coords.append(vox_zyx)

        np.random.shuffle(hard_neg_coords)

        count = 0
        for czyx in hard_neg_coords:
            if count >= n_neg:
                break

            patch_vol = extract_patch(vol, czyx, ps)
            label = np.zeros((1, *ps), dtype=np.float32)

            yield {
                "volume": patch_vol[np.newaxis],
                "label": label,
                "uid": uid,
                "is_nodule": 0,
                "centre_zyx": czyx,
                "radius_vox": 0.0
            }
            count += 1

        # ── Random negatives ──
        added = 0
        attempts = 0

        while added < (n_neg - count) and attempts < n_neg * 20:
            attempts += 1

            cz = np.random.randint(hz, Z - hz)
            cy = np.random.randint(hy, Y - hy)
            cx = np.random.randint(hx, X - hx)
            czyx = np.array([cz, cy, cx], dtype=float)

            too_close = any(
                np.linalg.norm(czyx - nod["zyx"]) < nod["radius_vox"] * 2
                for nod in nods
            )

            if too_close:
                continue

            patch_vol = extract_patch(vol, czyx, ps)
            label = np.zeros((1, *ps), dtype=np.float32)

            yield {
                "volume": patch_vol[np.newaxis],
                "label": label,
                "uid": uid,
                "is_nodule": 0,
                "centre_zyx": czyx,
                "radius_vox": 0.0
            }

            added += 1


# ────────────────────────────────
# CLASSIFIER CROPS (STREAMING)
# ────────────────────────────────

    def generate_classifier_crops(self, scan_info: Dict):
        vol   = scan_info["volume"]
        nods  = scan_info["nodules"]
        uid   = scan_info["seriesuid"]
        cs    = cfg.CLASSIFIER_CROP_SIZE

        for nod in nods:
            czyx  = nod["zyx"]
            d_mm  = nod["diameter_mm"]

            crop = extract_patch(vol, czyx, cs)
            label = int(d_mm >= 10.0)

            yield {
                "volume": crop[np.newaxis],
                "label": label,
                "uid": uid,
                "diameter_mm": d_mm
            }


# ────────────────────────────────
# RUN FUNCTION (STREAMING SAVE)
# ────────────────────────────────

    def run(self, max_scans: Optional[int] = None):
        import gc

        all_uids = self.candidates["seriesuid"].unique()
        if max_scans:
            all_uids = all_uids[:max_scans]

        det_idx = 0
        cls_idx = 0

        total_det = 0
        total_pos = 0
        total_cls = 0
        total_mal = 0

        print(f"[Preprocessor] Processing {len(all_uids)} scans...\n")

        for uid in tqdm(all_uids, desc="Scans"):
            try:
                info = self.process_scan(uid)
                if info is None:
                    continue

                # ── Detector patches (streaming) ──
                for p in self.generate_detector_patches(info):
                    np.savez_compressed(
                        cfg.DETECTOR_PATCHES_DIR / f"patch_{det_idx:06d}.npz",
                        volume=p["volume"],
                        label=p["label"],
                        is_nodule=np.array(p["is_nodule"]),
                        uid=np.array(p["uid"]),
                        centre_zyx=p["centre_zyx"],
                        radius_vox=np.array(p["radius_vox"])
                    )
                    total_pos += p["is_nodule"]
                    det_idx += 1
                    total_det += 1

                # ── Classifier crops (streaming) ──
                for c in self.generate_classifier_crops(info):
                    np.savez_compressed(
                        cfg.CLASSIFIER_CROPS_DIR / f"crop_{cls_idx:06d}.npz",
                        volume=c["volume"],
                        label=np.array(c["label"]),
                        uid=np.array(c["uid"]),
                        diameter_mm=np.array(c["diameter_mm"])
                    )
                    total_mal += c["label"]
                    cls_idx += 1
                    total_cls += 1

                # ── Free memory ──
                del info
                gc.collect()

            except Exception as e:
                print(f"⚠ Skipping {uid}: {e}")

        total_neg = total_det - total_pos
        total_ben = total_cls - total_mal

        print(f"\n✓ Detector patches — total: {total_det}, pos: {total_pos}, neg: {total_neg}")
        print(f"✓ Classifier crops — total: {total_cls}, malignant: {total_mal}, benign: {total_ben}")


# ═══════════════════════════════════════════════════════
# QUICK DEMO / UNIT TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("LUNA16 Preprocessor")
    print("=" * 50)

    # Smoke test with a synthetic volume
    dummy_vol = np.random.randn(128, 256, 256).astype(np.float32) * 200 - 500
    origin    = np.array([0., 0., 0.])
    spacing   = np.array([0.7, 0.7, 2.5])  # typical chest CT

    print("Testing resample_volume...")
    vol_1mm, factor = resample_volume(dummy_vol, spacing, (1.0, 1.0, 1.0))
    print(f"  Original shape: {dummy_vol.shape}, "
          f"Resampled shape: {vol_1mm.shape}, factor: {factor.round(2)}")

    print("Testing normalise_hu...")
    vol_norm = normalise_hu(vol_1mm)
    print(f"  Min: {vol_norm.min():.3f}, Max: {vol_norm.max():.3f}")

    print("Testing extract_patch...")
    patch = extract_patch(vol_norm, np.array([64., 128., 128.]), (64, 64, 64))
    print(f"  Patch shape: {patch.shape}")

    print("Testing gaussian sphere...")
    blob = make_gaussian_sphere((64, 64, 64), np.array([32., 32., 32.]), 8.0)
    print(f"  Blob max: {blob.max():.3f}, non-zero voxels: {(blob > 0.01).sum()}")

    print("\n✓ All preprocessing utilities functional.")
    print("Run LUNA16Preprocessor().run() once data is downloaded.")
