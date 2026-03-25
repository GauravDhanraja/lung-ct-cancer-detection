"""
inference.py
─────────────
End-to-end inference pipeline.

Given a CT scan (.mhd or .nii.gz), produces:
  1. List of detected nodule candidates with locations and sizes
  2. Malignancy probability for each candidate
  3. Grad-CAM heatmaps for explainability
  4. JSON report

Usage:
    python inference.py --scan /path/to/scan.mhd
    python inference.py --scan /path/to/scan.mhd --visualise
"""

import sys, json, time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from data.preprocessing import load_mhd, resample_volume, normalise_hu
from models.unet3d import UNet3D
from models.resnet3d import ResNet3D
from evaluation.metrics import (extract_candidates_from_probmap, nms_3d,
                                  compute_classification_metrics)
from explainability.gradcam3d import GradCAM3D, visualise_gradcam


# ═══════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════

def load_detector(ckpt_path: Optional[str] = None,
                  device: str = "cuda") -> UNet3D:
    model = UNet3D(use_checkpoint=False).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Detector loaded from {ckpt_path}")
    else:
        print("⚠  Detector: using random weights (no checkpoint)")
    model.eval()
    return model


def load_classifier(ckpt_path: Optional[str] = None,
                    device: str = "cuda") -> ResNet3D:
    model = ResNet3D(use_se=True, dropout=0.0).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Classifier loaded from {ckpt_path}")
    else:
        print("⚠  Classifier: using random weights (no checkpoint)")
    model.eval()
    return model


# ═══════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════

class NoduleDetectionPipeline:
    """
    Two-stage pipeline:
      Stage 1: UNet3D → dense nodule probability map → candidate extraction
      Stage 2: ResNet3D → malignancy probability per candidate
      Stage 3: Grad-CAM → visual explanations
    """

    def __init__(self,
                 detector_ckpt:   Optional[str] = None,
                 classifier_ckpt: Optional[str] = None,
                 device:          str = "auto"):

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Pipeline running on: {device}")

        self.detector   = load_detector(detector_ckpt, device)
        self.classifier = load_classifier(classifier_ckpt, device)

        # Grad-CAM hooks
        self.cam = GradCAM3D(self.classifier, self.classifier.layer3)

    def _preprocess_scan(self, scan_path: str):
        """Load, resample, normalise. Returns (volume_norm, origin, new_spacing)."""
        path = Path(scan_path)
        if not path.exists():
            raise FileNotFoundError(f"Scan not found: {scan_path}")

        volume_hu, origin, spacing = load_mhd(scan_path)
        volume_1mm, resize_factor  = resample_volume(volume_hu, spacing,
                                                      cfg.TARGET_SPACING)
        volume_norm = normalise_hu(volume_1mm)
        return volume_norm, origin, np.array(cfg.TARGET_SPACING), resize_factor

    def _detect_candidates(self, volume_norm: np.ndarray) -> List[Dict]:
        """Stage 1: UNet3D sliding-window → candidates."""
        vol_t = torch.from_numpy(volume_norm[np.newaxis]).to(self.device)

        with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
            prob_map = self.detector.predict_volume(
                vol_t,
                patch_size = cfg.DETECTOR_PATCH_SIZE,
                stride     = cfg.DETECTOR_STRIDE,
                device     = self.device
            )

        prob_np    = prob_map.squeeze().cpu().numpy()
        candidates = extract_candidates_from_probmap(prob_np,
                                                      cfg.DETECTION_THRESHOLD)
        candidates = nms_3d(candidates, cfg.NMS_IOU_THRESHOLD)
        return candidates

    def _classify_candidate(self, volume_norm: np.ndarray,
                              candidate: Dict) -> Dict:
        """Stage 2 + 3: Extract crop → ResNet → Grad-CAM."""
        from data.preprocessing import extract_patch
        czyx    = candidate["zyx"]
        crop    = extract_patch(volume_norm, czyx, cfg.CLASSIFIER_CROP_SIZE)
        crop_t  = torch.from_numpy(crop[np.newaxis, np.newaxis]).to(self.device)

        # Classify
        with torch.no_grad(), autocast(enabled=(self.device == "cuda")):
            logit = self.classifier(crop_t)
            prob  = float(torch.sigmoid(logit).item())

        # Grad-CAM
        try:
            heatmap, _ = self.cam(crop_t.float().requires_grad_(False))
        except Exception:
            heatmap = None

        return {
            **candidate,
            "malignancy_prob": prob,
            "malignant"      : prob >= 0.5,
            "crop"           : crop,
            "heatmap"        : heatmap
        }

    def __call__(self, scan_path: str,
                  save_dir: Optional[str] = None) -> Dict:
        """
        Run full pipeline on a single CT scan.

        Returns report dict:
        {
          "scan_path": ...,
          "n_candidates": ...,
          "candidates": [
              {"zyx": ..., "radius_mm": ..., "detection_prob": ...,
               "malignancy_prob": ..., "malignant": bool}, ...
          ],
          "summary": { "n_malignant": ..., "highest_risk": ... }
        }
        """
        t0 = time.time()
        print(f"\n{'─'*55}")
        print(f"  Processing: {Path(scan_path).name}")

        # 1. Preprocess
        volume_norm, origin, spacing, resize_factor = self._preprocess_scan(scan_path)
        print(f"  Volume shape: {volume_norm.shape}  (after 1mm resampling)")

        # 2. Detect
        print("  Stage 1: Nodule detection...")
        candidates = self._detect_candidates(volume_norm)
        print(f"  → {len(candidates)} candidate(s) found after NMS")

        # 3. Classify each candidate
        print("  Stage 2: Malignancy classification + Grad-CAM...")
        results = []
        for i, cand in enumerate(candidates):
            classified = self._classify_candidate(volume_norm, cand)
            results.append(classified)
            status = "🔴 MALIGNANT" if classified["malignant"] else "🟢 Benign"
            print(f"    [{i+1}] {status}  p={classified['malignancy_prob']:.3f}"
                  f"  @ vox {classified['zyx'].round(1)}")

        # 4. Save outputs
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self._save_report(results, scan_path, save_path)
            if results:
                self._save_visualisations(results, volume_norm, save_path)

        elapsed = time.time() - t0

        report = {
            "scan_path"   : str(scan_path),
            "n_candidates": len(results),
            "elapsed_sec" : round(elapsed, 2),
            "candidates"  : [
                {
                    "voxel_zyx"       : r["zyx"].tolist(),
                    "radius_mm"       : round(r["radius_mm"], 2),
                    "detection_prob"  : round(r["prob"], 4),
                    "malignancy_prob" : round(r["malignancy_prob"], 4),
                    "malignant"       : bool(r["malignant"])
                }
                for r in results
            ],
            "summary": {
                "n_malignant" : sum(1 for r in results if r["malignant"]),
                "highest_risk": max((r["malignancy_prob"] for r in results),
                                     default=0.0)
            }
        }

        print(f"\n  ✓ Done in {elapsed:.1f}s  |  "
              f"Malignant candidates: {report['summary']['n_malignant']}")
        return report

    def _save_report(self, results, scan_path, save_dir):
        report_data = {
            "scan": str(scan_path),
            "candidates": [
                {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in r.items()
                 if k not in ("crop", "heatmap")}
                for r in results
            ]
        }
        p = save_dir / "report.json"
        with open(p, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"  Report saved: {p}")

    def _save_visualisations(self, results, volume_norm, save_dir):
        for i, r in enumerate(results):
            if r.get("heatmap") is None:
                continue
            fig = visualise_gradcam(
                r["crop"], r["heatmap"], r["malignancy_prob"],
                uid=f"Candidate {i+1}",
                save_path=save_dir / f"gradcam_candidate_{i+1}.png"
            )
            plt.close(fig)

        # Detection overview: mid-axial slice + all candidate locations
        Z, Y, X  = volume_norm.shape
        mid_z    = Z // 2
        fig, ax  = plt.subplots(figsize=(8, 8))
        ax.imshow(volume_norm[mid_z], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Mid-axial Slice (z={mid_z}) — Detected Candidates",
                      fontsize=12, fontweight='bold')
        for r in results:
            cz, cy, cx = r["zyx"]
            if abs(cz - mid_z) < r["radius_mm"] * 2:
                col = 'red' if r["malignant"] else 'lime'
                circ = plt.Circle((cx, cy), r["radius_mm"],
                                   color=col, fill=False, lw=2)
                ax.add_patch(circ)
                ax.text(cx + r["radius_mm"] + 2, cy,
                         f"p={r['malignancy_prob']:.2f}",
                         color=col, fontsize=8)
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(save_dir / "detection_overview.png",
                     dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"  Visualisations saved to {save_dir}")

    def cleanup(self):
        """Release Grad-CAM hooks."""
        self.cam.remove_hooks()


# ═══════════════════════════════════════════════════════
# SYNTHETIC DEMO (no real CT needed)
# ═══════════════════════════════════════════════════════

def run_synthetic_demo(save_dir: str = "results/demo"):
    """
    Demonstrate the pipeline on a synthetic CT volume with planted nodules.
    Useful for CI / testing without LUNA16 data.
    """
    print("\n" + "="*55)
    print("  SYNTHETIC DEMO — Lung Nodule AI Pipeline")
    print("="*55)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create models
    detector   = UNet3D(use_checkpoint=False).to(device).eval()
    classifier = ResNet3D(use_se=True, dropout=0.0).to(device).eval()

    # Synthetic volume (128³)
    rng = np.random.RandomState(42)
    volume = np.clip(rng.randn(128, 128, 128).astype(np.float32) * 0.05 + 0.15,
                     0, 1)

    # Plant 3 synthetic nodules
    planted = [
        {"zyx": np.array([40., 60., 70.]), "r": 8},
        {"zyx": np.array([80., 50., 80.]), "r": 5},
        {"zyx": np.array([60., 90., 40.]), "r": 12}
    ]
    for nod in planted:
        cz, cy, cx = nod["zyx"].astype(int)
        r = nod["r"]
        zz, yy, xx = np.ogrid[max(0,cz-r):min(128,cz+r),
                                max(0,cy-r):min(128,cy+r),
                                max(0,cx-r):min(128,cx+r)]
        sphere = (zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 < r**2
        volume[max(0,cz-r):min(128,cz+r),
                max(0,cy-r):min(128,cy+r),
                max(0,cx-r):min(128,cx+r)][sphere] = rng.uniform(0.5, 0.8)

    print(f"Synthetic volume: {volume.shape}  | {len(planted)} nodules planted")

    # Stage 1: Detector (sliding-window on small crop for demo)
    print("Running detector (sliding-window)...")
    vol_t = torch.from_numpy(volume[np.newaxis]).to(device)
    with torch.no_grad(), autocast(enabled=(device == "cuda")):
        prob_map = detector.predict_volume(vol_t, device=device)
    prob_np    = prob_map.squeeze().cpu().numpy()
    candidates = extract_candidates_from_probmap(prob_np, threshold=0.3)
    candidates = nms_3d(candidates)
    print(f"  Detected {len(candidates)} candidate(s) (random weights — not tuned)")

    # If no candidates found with random weights, use planted nodule locations
    if not candidates:
        print("  (Using planted locations as candidates for demo)")
        candidates = [{"zyx": p["zyx"], "radius_mm": p["r"],
                        "prob": 0.7} for p in planted]

    # Stage 2 + 3: Classifier + Grad-CAM
    cam = GradCAM3D(classifier, target_layer=classifier.layer3)
    from data.preprocessing import extract_patch

    for i, cand in enumerate(candidates[:5]):
        crop = extract_patch(volume, cand["zyx"], cfg.CLASSIFIER_CROP_SIZE)
        crop_t = torch.from_numpy(crop[np.newaxis, np.newaxis]).float().to(device)
        with torch.no_grad(), autocast(enabled=(device == "cuda")):
            logit = classifier(crop_t)
            prob  = float(torch.sigmoid(logit).item())

        heatmap, _ = cam(crop_t.float())
        fig = visualise_gradcam(crop, heatmap, prob,
                                  uid=f"Synthetic Candidate {i+1}",
                                  save_path=save_path / f"demo_gradcam_{i+1}.png")
        plt.close(fig)
        print(f"  Candidate {i+1}: p_malignant={prob:.3f}  "
              f"CAM saved → demo_gradcam_{i+1}.png")

    cam.remove_hooks()

    # Summary figure
    mid = 64
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[mid], cmap='gray')
    axes[0].set_title("CT Slice (axial)", fontsize=11)
    axes[1].imshow(prob_np[mid], cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Detection Prob Map", fontsize=11)
    axes[2].imshow(volume[mid], cmap='gray')
    for nod in planted:
        cz, cy, cx = nod["zyx"]
        if abs(cz - mid) < nod["r"] * 2:
            c = plt.Circle((cx, cy), nod["r"], color='lime', fill=False, lw=2)
            axes[2].add_patch(c)
    axes[2].set_title("Ground Truth Nodules", fontsize=11)
    for ax in axes:
        ax.axis('off')
    plt.suptitle("Lung Nodule AI — Synthetic Demo", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path / "demo_overview.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Demo complete. All outputs saved to {save_path}/")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Lung CT Nodule Detection & Malignancy Prediction")
    parser.add_argument("--scan",          type=str,  default=None,
                        help="Path to CT scan (.mhd/.nii.gz)")
    parser.add_argument("--detector-ckpt", type=str,
                        default=str(cfg.CHECKPOINTS_DIR / "detector_best.pth"))
    parser.add_argument("--classifier-ckpt", type=str,
                        default=str(cfg.CHECKPOINTS_DIR / "classifier_best.pth"))
    parser.add_argument("--save-dir",      type=str,
                        default=str(cfg.RESULTS_DIR / "inference"))
    parser.add_argument("--device",        type=str,  default="auto")
    parser.add_argument("--demo",          action="store_true",
                        help="Run synthetic demo (no real CT needed)")
    args = parser.parse_args()

    if args.demo or args.scan is None:
        run_synthetic_demo(save_dir=str(cfg.RESULTS_DIR / "demo"))
    else:
        pipeline = NoduleDetectionPipeline(
            detector_ckpt   = args.detector_ckpt,
            classifier_ckpt = args.classifier_ckpt,
            device          = args.device
        )
        report = pipeline(args.scan, save_dir=args.save_dir)
        pipeline.cleanup()

        print("\n── Final Report ──")
        print(json.dumps({k: v for k, v in report.items()
                          if k != "candidates"}, indent=2))
        print(f"  Detailed report: {args.save_dir}/report.json")
