"""
main.py
────────
Master entry point for the Lung Nodule AI project.

Stages:
  preprocess  → run LUNA16 preprocessing pipeline
  train-det   → train 3D U-Net detector
  train-cls   → train 3D ResNet-10 classifier
  evaluate    → FROC + classification metrics on val set
  demo        → synthetic end-to-end demo (no data needed)
  infer       → run on a single CT scan
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lung CT Nodule Detection & Malignancy Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick sanity test (no data needed, synthetic volumes):
  python main.py demo

  # Full pipeline with LUNA16 data:
  python main.py preprocess
  python main.py train-det --epochs 60
  python main.py train-cls --epochs 80
  python main.py evaluate
  python main.py infer --scan /path/to/scan.mhd
        """
    )
    parser.add_argument("stage", choices=[
        "preprocess", "train-det", "train-cls",
        "evaluate", "demo", "infer", "test"
    ])

    # Shared flags
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use synthetic data (no LUNA16 needed)")

    # Infer-specific
    parser.add_argument("--scan",       default=None)
    parser.add_argument("--save-dir",   default=None)

    # Preprocess-specific
    parser.add_argument("--max-scans",  type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print(f"  Lung Nodule AI  —  Stage: {args.stage.upper()}")
    print(f"{'═'*60}\n")

    # ── DEMO ──────────────────────────────────────────────────────
    if args.stage == "demo":
        from inference import run_synthetic_demo
        run_synthetic_demo(save_dir=str(cfg.RESULTS_DIR / "demo"))

    # ── UNIT TEST ─────────────────────────────────────────────────
    elif args.stage == "test":
        run_unit_tests()

    # ── PREPROCESS ────────────────────────────────────────────────
    elif args.stage == "preprocess":
        from data.preprocessing import LUNA16Preprocessor
        print("Starting LUNA16 preprocessing...")
        print(f"Data directory: {cfg.DATA_DIR}")
        if not cfg.DATA_DIR.exists():
            print(f"\n⚠  LUNA16 data not found at {cfg.DATA_DIR}")
            print("Please download LUNA16 from:")
            print("  https://luna16.grand-challenge.org/")
            print("and place the subset folders + CSV files there.\n")
            return
        preprocessor = LUNA16Preprocessor()
        preprocessor.run(max_scans=args.max_scans)

    # ── TRAIN DETECTOR ────────────────────────────────────────────
    elif args.stage == "train-det":
        from training.train_detector import train_detector
        train_detector(
            use_synthetic = args.synthetic,
            resume_from   = args.resume,
            epochs        = args.epochs or cfg.DETECTOR_EPOCHS,
            lr            = args.lr    or cfg.DETECTOR_LR,
            batch_size    = args.batch_size or cfg.DETECTOR_BATCH_SIZE,
            device        = args.device
        )

    # ── TRAIN CLASSIFIER ──────────────────────────────────────────
    elif args.stage == "train-cls":
        from training.train_classifier import train_classifier
        train_classifier(
            use_synthetic = args.synthetic,
            resume_from   = args.resume,
            epochs        = args.epochs or cfg.CLASSIFIER_EPOCHS,
            lr            = args.lr    or cfg.CLASSIFIER_LR,
            batch_size    = args.batch_size or cfg.CLASSIFIER_BATCH_SIZE,
            device        = args.device
        )

    # ── EVALUATE ──────────────────────────────────────────────────
    elif args.stage == "evaluate":
        run_evaluation(args)

    # ── INFERENCE ─────────────────────────────────────────────────
    elif args.stage == "infer":
        if args.scan is None:
            print("Error: --scan required for infer stage")
            return
        from inference import NoduleDetectionPipeline
        pipeline = NoduleDetectionPipeline(
            detector_ckpt   = str(cfg.CHECKPOINTS_DIR / "detector_best.pth"),
            classifier_ckpt = str(cfg.CHECKPOINTS_DIR / "classifier_best.pth"),
            device          = args.device
        )
        pipeline(args.scan, save_dir=args.save_dir or str(cfg.RESULTS_DIR / "inference"))
        pipeline.cleanup()


# ═══════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════════════════

def run_evaluation(args):
    import json
    import torch
    import numpy as np
    from torch.amp import autocast
    from tqdm import tqdm
    from models.unet3d import UNet3D
    from models.resnet3d import ResNet3D
    from data.dataset import get_classifier_loaders, SyntheticNoduleDataset
    from torch.utils.data import DataLoader
    from evaluation.metrics import (compute_classification_metrics,
                                     plot_roc, plot_training_history)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    print("Running evaluation on validation set...\n")

    # ── Classifier evaluation ──
    cls_ckpt = cfg.CHECKPOINTS_DIR / "classifier_best.pth"
    classifier = ResNet3D(use_se=True, dropout=0.0).to(device)

    if cls_ckpt.exists():
        ckpt = torch.load(cls_ckpt, map_location=device, weights_only=False)
        classifier.load_state_dict(ckpt["model"])
        print(f"Loaded classifier: {cls_ckpt}")
    else:
        print("⚠  No classifier checkpoint found — using random weights")

    classifier.eval()

    # Data
    if args.synthetic:
        full_ds = SyntheticNoduleDataset(n_samples=200, patch_size=(32,32,32),
                                          mode="classifier")
        val_loader = DataLoader(full_ds, batch_size=16, shuffle=False)
    else:
        _, val_loader = get_classifier_loaders()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for vols, labels, _ in tqdm(val_loader, desc="Classifying"):
            vols   = vols.to(device)
            with autocast("cuda", enabled=(device == "cuda")):
                probs = torch.sigmoid(classifier(vols)).cpu().squeeze().float()
            all_probs.append(probs)
            all_labels.append(labels.float())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_classification_metrics(all_labels, all_probs)

    print("\n── Malignancy Classifier ──────────────────────")
    print(f"  AUC-ROC        : {metrics['auc']:.4f}")
    print(f"  Average Prec   : {metrics['average_prec']:.4f}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Balanced Acc   : {metrics['balanced_acc']:.4f}")
    print(f"  Sensitivity    : {metrics['sensitivity']:.4f}")
    print(f"  Specificity    : {metrics['specificity']:.4f}")
    print(f"  PPV (Precision): {metrics['ppv']:.4f}")
    print(f"  NPV            : {metrics['npv']:.4f}")
    print(f"  F1 Score       : {metrics['f1']:.4f}")
    print(f"  Youden Thresh  : {metrics['youden_threshold']:.4f}")
    print(f"  Confusion matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("──────────────────────────────────────────────\n")

    # Save metrics
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    with open(cfg.RESULTS_DIR / "eval_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if not isinstance(v, list)}, f, indent=2)

    # ROC plot
    import matplotlib.pyplot as plt
    fpr = np.array(metrics["fpr"])
    tpr = np.array(metrics["tpr"])
    fig = plot_roc(fpr, tpr, metrics["auc"],
                    save_path=cfg.RESULTS_DIR / "roc_curve.png")
    plt.close(fig)

    # Training history plots
    for stage in ["detector", "classifier"]:
        hist_file = cfg.RESULTS_DIR / f"{stage}_history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                hist = json.load(f)
            fig = plot_training_history(hist, stage=stage,
                       save_path=cfg.RESULTS_DIR / f"{stage}_training_curves.png")
            plt.close(fig)

    print(f"✓ Evaluation complete. Results saved to {cfg.RESULTS_DIR}/")


# ═══════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════

def run_unit_tests():
    """Quick sanity tests for all modules."""
    import torch
    print("Running unit tests...\n")

    tests_passed = 0

    # Test 1: Models
    try:
        from models.unet3d import UNet3D, FocalDiceLoss
        m = UNet3D()
        x = torch.randn(1, 1, 64, 64, 64)
        y = m(x)
        assert y.shape == (1, 1, 64, 64, 64), f"Wrong shape: {y.shape}"
        print("  ✓ UNet3D forward pass")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ UNet3D: {e}")

    try:
        from models.resnet3d import ResNet3D
        m = ResNet3D()
        x = torch.randn(4, 1, 32, 32, 32)
        y = m(x)
        assert y.shape == (4, 1), f"Wrong shape: {y.shape}"
        f = m.forward_features(x)
        assert f.ndim == 5
        print("  ✓ ResNet3D forward pass + features")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ ResNet3D: {e}")

    # Test 2: Losses
    try:
        from models.unet3d import FocalDiceLoss
        from models.resnet3d import LabelSmoothingBCE
        fl = FocalDiceLoss()
        logits = torch.randn(2, 1, 64, 64, 64)
        labels = torch.zeros(2, 1, 64, 64, 64)
        loss = fl(logits, labels)
        assert loss.item() > 0
        print("  ✓ FocalDiceLoss")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FocalDiceLoss: {e}")

    # Test 3: Grad-CAM
    try:
        from explainability.gradcam3d import GradCAM3D
        from models.resnet3d import ResNet3D
        model = ResNet3D().eval()
        cam   = GradCAM3D(model, model.layer3)
        x     = torch.randn(1, 1, 32, 32, 32)
        h, p  = cam(x)
        assert h.shape == (32, 32, 32)
        assert 0.0 <= h.max() <= 1.0
        cam.remove_hooks()
        print("  ✓ Grad-CAM 3D")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Grad-CAM: {e}")

    # Test 4: Metrics
    try:
        import numpy as np
        from evaluation.metrics import compute_classification_metrics
        y_true = np.array([0,0,1,1,0,1,0,1])
        y_prob = np.array([0.1,0.2,0.8,0.9,0.3,0.7,0.4,0.6])
        m = compute_classification_metrics(y_true, y_prob)
        assert 0.5 < m["auc"] <= 1.0
        print("  ✓ Classification metrics")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Metrics: {e}")

    # Test 5: Preprocessing (synthetic)
    try:
        import numpy as np
        from data.preprocessing import (resample_volume, normalise_hu,
                                          extract_patch, make_gaussian_sphere)
        vol = np.random.randn(64, 128, 128).astype(np.float32) * 200 - 500
        vol_r, _ = resample_volume(vol, np.array([1., 1., 2.]), (1., 1., 1.))
        vol_n    = normalise_hu(vol_r)
        assert 0 <= vol_n.min() and vol_n.max() <= 1.0
        patch = extract_patch(vol_n, np.array([32., 64., 64.]), (32, 32, 32))
        assert patch.shape == (32, 32, 32)
        blob  = make_gaussian_sphere((32,32,32), np.array([16.,16.,16.]), 5.0)
        assert blob.max() <= 1.0
        print("  ✓ Preprocessing utilities")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Preprocessing: {e}")

    print(f"\n{'─'*40}")
    print(f"  Tests passed: {tests_passed}/6")
    if tests_passed == 6:
        print("  All tests passed!")
    else:
        print("  ⚠  Some tests failed — check errors above.")


if __name__ == "__main__":
    main()
