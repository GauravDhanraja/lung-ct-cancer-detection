"""
evaluation/metrics.py
──────────────────────
Clinical evaluation metrics for nodule detection and classification.

Implements:
  1. FROC (Free-Response ROC) curve  — primary LUNA16 metric
  2. CPM  (Competition Performance Metric)  — average sensitivity at 7 FP rates
  3. ROC / AUC for the classifier
  4. Nodule-level TP/FP/FN matching (IoU + distance-based)
  5. Visualisation helpers for publication-quality plots
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label as scipy_label, center_of_mass
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ═══════════════════════════════════════════════════════
# NODULE CANDIDATE POST-PROCESSING
# ═══════════════════════════════════════════════════════

def extract_candidates_from_probmap(
        prob_map:   np.ndarray,
        threshold:  float = cfg.DETECTION_THRESHOLD,
        min_radius: float = cfg.MIN_NODULE_DIAM_MM / 2,
        max_radius: float = cfg.MAX_NODULE_DIAM_MM / 2,
) -> List[Dict]:
    """
    Extract candidate nodule centres from a 3D probability map.

    Steps:
      1. Threshold the map
      2. Connected-component labelling
      3. Filter by estimated radius (volume-based)
      4. Return list of {zyx, prob, radius_mm}

    prob_map: (Z, Y, X) float32 in [0, 1]
    """
    binary = (prob_map > threshold).astype(np.uint8)
    labeled, n_comp = scipy_label(binary)

    candidates = []
    for comp_id in range(1, n_comp + 1):
        mask      = (labeled == comp_id)
        n_voxels  = mask.sum()
        if n_voxels < 5:              # ignore tiny specks
            continue

        # Estimate radius from volume (assuming sphere)
        radius_vox = (3 * n_voxels / (4 * np.pi)) ** (1/3)
        if radius_vox < min_radius or radius_vox > max_radius:
            continue

        # Centre of mass
        czyx = np.array(center_of_mass(mask))

        # Peak probability within the component
        peak_prob = prob_map[mask].max()

        candidates.append({
            "zyx"      : czyx,
            "radius_mm": float(radius_vox),
            "prob"     : float(peak_prob),
            "n_voxels" : int(n_voxels)
        })

    # Sort by probability descending
    candidates.sort(key=lambda c: c["prob"], reverse=True)
    return candidates


def nms_3d(candidates: List[Dict],
           iou_threshold: float = cfg.NMS_IOU_THRESHOLD) -> List[Dict]:
    """
    3D Sphere-IoU based Non-Maximum Suppression.
    Removes overlapping candidates, keeping highest-probability ones.
    """
    if not candidates:
        return []

    kept = []
    for c in candidates:
        suppress = False
        for k in kept:
            # Distance between centres
            dist = np.linalg.norm(c["zyx"] - k["zyx"])
            # Approximate IoU via overlap of radii
            r1, r2 = c["radius_mm"], k["radius_mm"]
            if dist < (r1 + r2) * 0.5:
                suppress = True
                break
        if not suppress:
            kept.append(c)
    return kept


# ═══════════════════════════════════════════════════════
# NODULE MATCHING (LUNA16 protocol)
# ═══════════════════════════════════════════════════════

def match_candidates_to_gt(
        candidates: List[Dict],
        gt_nodules: List[Dict],
        match_threshold_mm: float = 5.0
) -> Tuple[List[bool], int]:
    """
    Match each candidate to a ground-truth nodule.

    A candidate is a True Positive if:
      - Its centre falls within `match_threshold_mm` of a GT centre
      - Each GT nodule can only be matched once

    Returns:
      is_tp      : list of booleans (len = len(candidates))
      n_fn       : number of unmatched GT nodules (False Negatives)
    """
    gt_matched = [False] * len(gt_nodules)
    is_tp      = []

    for cand in candidates:
        tp = False
        for j, gt in enumerate(gt_nodules):
            if gt_matched[j]:
                continue
            dist = np.linalg.norm(cand["zyx"] - gt["zyx"])
            if dist <= match_threshold_mm:
                gt_matched[j] = True
                tp = True
                break
        is_tp.append(tp)

    n_fn = sum(1 for m in gt_matched if not m)
    return is_tp, n_fn


# ═══════════════════════════════════════════════════════
# FROC COMPUTATION
# ═══════════════════════════════════════════════════════

def compute_froc(
        all_candidates: List[List[Dict]],   # per-scan candidate lists
        all_gt:         List[List[Dict]],   # per-scan GT nodule lists
        n_scans:        int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the FROC curve across all scans.

    Parameters
    ----------
    all_candidates : list of per-scan candidate lists (post-NMS)
    all_gt         : list of per-scan GT nodule lists
    n_scans        : total number of CT scans evaluated

    Returns
    -------
    fp_per_scan    : array of FP/scan values (x-axis)
    sensitivity    : array of sensitivity values (y-axis)
    cpm            : Competition Performance Metric (area under FROC at 7 FP rates)
    """
    # Flatten all candidates with their scan index
    all_flat = []
    total_gt = 0
    for scan_idx, (cands, gts) in enumerate(zip(all_candidates, all_gt)):
        total_gt += len(gts)
        for c in cands:
            all_flat.append((c["prob"], scan_idx, c))

    if not all_flat or total_gt == 0:
        return np.array([0., 8.]), np.array([0., 0.]), 0.0

    # Sort by descending probability
    all_flat.sort(key=lambda x: x[0], reverse=True)

    # Walk down the sorted list, accumulating TP/FP
    gt_matched = [[False]*len(gts) for gts in all_gt]
    tp_count   = 0
    fp_count   = 0
    sensitivities = []
    fp_per_scan   = []

    for prob, scan_idx, cand in all_flat:
        matched = False
        for j, gt in enumerate(all_gt[scan_idx]):
            if gt_matched[scan_idx][j]:
                continue
            dist = np.linalg.norm(cand["zyx"] - gt["zyx"])
            if dist <= 5.0:   # 5mm matching radius
                gt_matched[scan_idx][j] = True
                tp_count  += 1
                matched    = True
                break
        if not matched:
            fp_count += 1

        sensitivities.append(tp_count / total_gt)
        fp_per_scan.append(fp_count / n_scans)

    fp_per_scan  = np.array(fp_per_scan)
    sensitivities = np.array(sensitivities)

    # CPM: average sensitivity at 7 standard FP/scan levels
    cpm_points = []
    for fp_target in cfg.FROC_FP_RATES:
        idx = np.searchsorted(fp_per_scan, fp_target)
        idx = min(idx, len(sensitivities) - 1)
        cpm_points.append(sensitivities[idx])
    cpm = float(np.mean(cpm_points))

    return fp_per_scan, sensitivities, cpm


# ═══════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════

def compute_classification_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5
) -> Dict:
    """
    Full classification report for malignancy prediction.

    Returns dict with: AUC, AP, sensitivity, specificity, PPV, NPV,
                        accuracy, F1, Youden-J threshold
    """
    from sklearn.metrics import (accuracy_score, f1_score,
                                  balanced_accuracy_score)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
    avg_prec = float(np.trapezoid(rec[::-1], prec[::-1]))  # AP

    # Youden's J index (maximise sens + spec)
    j_scores     = tpr - fpr
    best_idx     = np.argmax(j_scores)
    best_thresh  = float(thresholds[best_idx])

    # Binary metrics at given threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn  = int(((y_pred == 0) & (y_true == 0)).sum())
    fp  = int(((y_pred == 1) & (y_true == 0)).sum())
    fn  = int(((y_pred == 0) & (y_true == 1)).sum())
    tp  = int(((y_pred == 1) & (y_true == 1)).sum())

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)    # precision
    npv  = tn / max(tn + fn, 1)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred)

    return {
        "auc"          : float(roc_auc),
        "average_prec" : float(avg_prec),
        "accuracy"     : float(acc),
        "balanced_acc" : float(bal),
        "sensitivity"  : float(sens),
        "specificity"  : float(spec),
        "ppv"          : float(ppv),
        "npv"          : float(npv),
        "f1"           : float(f1),
        "youden_threshold": float(best_thresh),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "fpr"          : fpr.tolist(),
        "tpr"          : tpr.tolist()
    }


# ═══════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════

def plot_froc(fp_per_scan: np.ndarray,
              sensitivity: np.ndarray,
              cpm:         float,
              save_path:   Optional[Path] = None,
              label:       str = "Model") -> plt.Figure:
    """
    Publication-quality FROC curve plot (LUNA16 style).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fp_per_scan, sensitivity, 'b-', lw=2, label=f"{label} (CPM={cpm:.3f})")
    ax.fill_between(fp_per_scan, sensitivity, alpha=0.1, color='blue')

    # Mark the 7 standard FROC evaluation points
    for fp_target in cfg.FROC_FP_RATES:
        idx = np.searchsorted(fp_per_scan, fp_target)
        idx = min(idx, len(sensitivity) - 1)
        ax.scatter(fp_per_scan[idx], sensitivity[idx],
                   c='red', s=80, zorder=5)
        ax.annotate(f"{sensitivity[idx]:.2f}",
                    (fp_per_scan[idx], sensitivity[idx]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positives per Scan", fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.set_title("FROC Curve — Lung Nodule Detection", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.8, color='gray', ls='--', alpha=0.5, label='80% sensitivity')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FROC plot saved: {save_path}")
    return fig


def plot_roc(fpr: np.ndarray,
             tpr: np.ndarray,
             roc_auc: float,
             save_path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel("1 − Specificity (FPR)", fontsize=13)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=13)
    ax.set_title("ROC Curve — Malignancy Classification", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_training_history(history: Dict,
                           stage: str = "detector",
                           save_path: Optional[Path] = None) -> plt.Figure:
    """Plot training/validation curves for loss and primary metric."""
    train_m = history["train"]
    val_m   = history["val"]
    epochs  = range(1, len(train_m) + 1)

    metric_key = "dice" if stage == "detector" else "auc"
    metric_name = "Dice" if stage == "detector" else "AUC"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, [m["loss"] for m in train_m], 'b-', label="Train")
    ax1.plot(epochs, [m["loss"] for m in val_m],   'r-', label="Val")
    ax1.set_title(f"{stage.capitalize()} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metric
    ax2.plot(epochs, [m[metric_key] for m in train_m], 'b-', label="Train")
    ax2.plot(epochs, [m[metric_key] for m in val_m],   'r-', label="Val")
    ax2.set_title(f"{stage.capitalize()} — {metric_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing evaluation metrics with synthetic data...")

    # Synthetic FROC data
    n_scans = 50
    n_gt_per_scan = 3
    all_gt = []
    all_cands = []

    rng = np.random.RandomState(0)
    for i in range(n_scans):
        gts = [{"zyx": rng.rand(3)*100} for _ in range(n_gt_per_scan)]
        all_gt.append(gts)

        cands = []
        for gt in gts:
            if rng.rand() > 0.2:    # 80% detection rate
                noise = rng.randn(3) * 3  # ~3mm error
                cands.append({
                    "zyx"      : gt["zyx"] + noise,
                    "radius_mm": 5.0,
                    "prob"     : float(rng.rand() * 0.3 + 0.7)
                })
        # Add FP candidates
        for _ in range(rng.randint(0, 4)):
            cands.append({
                "zyx"      : rng.rand(3)*100,
                "radius_mm": 4.0,
                "prob"     : float(rng.rand() * 0.4)
            })
        all_cands.append(cands)

    fp_scan, sens, cpm = compute_froc(all_cands, all_gt, n_scans)
    print(f"CPM: {cpm:.3f}")

    # Synthetic classification metrics
    y_true = rng.randint(0, 2, 200)
    y_prob = np.clip(y_true * 0.6 + rng.rand(200) * 0.4, 0, 1)
    cm = compute_classification_metrics(y_true, y_prob)
    print(f"AUC: {cm['auc']:.3f}  Sens: {cm['sensitivity']:.3f}  Spec: {cm['specificity']:.3f}")

    # Plot
    fig = plot_froc(fp_scan, sens, cpm,
                     save_path=Path("results/froc_test.png"))
    print("✓ Metrics module functional.")
