"""
explainability/gradcam3d.py
────────────────────────────
3D Gradient-weighted Class Activation Mapping (Grad-CAM) for the
ResNet3D malignancy classifier.

Theory:
  α_k^c  = (1 / Z) Σ_{z,y,x} ∂ŷ^c / ∂A^k_{z,y,x}   (global avg of grads)
  L_c    = ReLU( Σ_k  α_k^c · A^k )                  (weighted sum of feature maps)
  L_c is then trilinearly upsampled to input size (32³)

References:
  Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks…" ICCV 2017
  Zhou et al. "Learning Deep Features for Discriminative Localisation" CVPR 2016
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from models.resnet3d import ResNet3D


# ═══════════════════════════════════════════════════════
# GRAD-CAM HOOK MANAGER
# ═══════════════════════════════════════════════════════

class GradCAM3D:
    """
    Grad-CAM for any 3D CNN with a `forward_features` method.

    Usage:
        cam = GradCAM3D(model, target_layer=model.layer3)
        heatmap = cam(input_tensor)   # (D, H, W) in [0,1]
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self._activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._fwd_hook = self.target_layer.register_forward_hook(save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(save_gradient)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self,
                 input_tensor: torch.Tensor,
                 class_idx:    Optional[int] = None,
                 smooth:       bool  = True,
                 relu:         bool  = True) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : (1, 1, D, H, W) float32 tensor on correct device
        class_idx    : target class (None = use argmax)
        smooth       : apply Gaussian smoothing to the final map
        relu         : apply ReLU (keep only positive contributions)

        Returns
        -------
        heatmap : (D, H, W) numpy array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)            # (1, 1) logit
        prob   = torch.sigmoid(output)

        # Determine target class
        if class_idx is None:
            class_idx = int(output.squeeze().item() > 0)

        # Backward pass to target class score
        self.model.zero_grad()
        score = output.squeeze()                     # scalar
        score.backward(retain_graph=False)

        # Activations and gradients: (1, C, d, h, w)
        grads = self._gradients                      # (1, C, d, h, w)
        acts  = self._activations                    # (1, C, d, h, w)

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not fire. Check target_layer.")

        # Global average pooling of gradients → (1, C, 1, 1, 1)
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted combination of feature maps → (1, 1, d, h, w)
        cam = (weights * acts).sum(dim=1, keepdim=True)

        if relu:
            cam = F.relu(cam)

        # Upsample to input spatial size
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='trilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()   # (D, H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if smooth:
            from scipy.ndimage import gaussian_filter
            cam = gaussian_filter(cam, sigma=1.0)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, float(prob.squeeze().item())


# ═══════════════════════════════════════════════════════
# SCORE-CAM (gradient-free, more stable) 
# ═══════════════════════════════════════════════════════

class ScoreCAM3D:
    """
    Score-CAM (gradient-free alternative to Grad-CAM).
    Masks the input with each channel of the target feature map,
    then uses the change in output probability as the weight.

    Slower than Grad-CAM but more stable for medical imaging.
    Wang et al. "Score-CAM: Score-Weighted Visual Explanations…" CVPR 2020
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 max_channels: int = 32):
        self.model        = model
        self.target_layer = target_layer
        self.max_channels = max_channels  # cap for memory
        self._activations = None
        self._fwd_hook    = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_activations', o.detach())
        )

    def remove_hooks(self):
        self._fwd_hook.remove()

    @torch.no_grad()
    def __call__(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        self.model.eval()
        B, C, D, H, W = input_tensor.shape

        # Baseline output
        logit_base = self.model(input_tensor).squeeze()
        prob_base  = torch.sigmoid(logit_base).item()

        # Get feature maps
        _ = self.model(input_tensor)  # triggers hook
        acts = self._activations      # (1, n_ch, d, h, w)
        n_ch = min(acts.shape[1], self.max_channels)

        weights    = []
        for k in range(n_ch):
            # Upsample channel k to input size
            act_k  = acts[:, k:k+1, ...]     # (1, 1, d, h, w)
            act_up = F.interpolate(act_k, size=(D, H, W),
                                   mode='trilinear', align_corners=False)
            # Normalise to [0, 1]
            a_min, a_max = act_up.min(), act_up.max()
            if a_max > a_min:
                act_up = (act_up - a_min) / (a_max - a_min)
            # Masked input
            masked_input = input_tensor * act_up
            logit_masked = self.model(masked_input).squeeze()
            prob_masked  = torch.sigmoid(logit_masked).item()
            weights.append(prob_masked - prob_base)   # delta probability

        # Weighted sum of upsampled activations
        weights  = torch.tensor(weights, device=acts.device)
        weights  = F.softmax(weights, dim=0)
        cam      = torch.zeros(D, H, W, device=acts.device)

        for k in range(n_ch):
            act_k  = acts[:, k:k+1, ...]
            act_up = F.interpolate(act_k, size=(D, H, W),
                                   mode='trilinear', align_corners=False).squeeze()
            cam   += weights[k] * act_up

        cam = F.relu(cam).cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, prob_base


# ═══════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════

def visualise_gradcam(
        volume:   np.ndarray,   # (D, H, W) normalised CT crop
        heatmap:  np.ndarray,   # (D, H, W) Grad-CAM in [0, 1]
        prob:     float,
        uid:      str = "",
        save_path: Optional[Path] = None,
        n_slices: int = 5
) -> plt.Figure:
    """
    Visualise Grad-CAM overlaid on CT slices.
    Shows n_slices axial, coronal, and sagittal views.
    """
    D, H, W = volume.shape
    label   = "Malignant" if prob >= 0.5 else "Benign"
    colour  = "#d62728" if prob >= 0.5 else "#2ca02c"

    # Sample evenly-spaced slices
    axial_idxs   = np.linspace(D//4, 3*D//4, n_slices, dtype=int)
    coronal_idxs = np.linspace(H//4, 3*H//4, n_slices, dtype=int)
    sagittal_idxs= np.linspace(W//4, 3*W//4, n_slices, dtype=int)

    fig = plt.figure(figsize=(n_slices * 3, 9))
    fig.suptitle(
        f"Grad-CAM Explainability  |  {uid}\n"
        f"Prediction: {label}  (p = {prob:.3f})",
        fontsize=13, fontweight='bold', color=colour
    )
    gs = gridspec.GridSpec(3, n_slices, hspace=0.05, wspace=0.05)

    def overlay(ax, ct_slice, cam_slice, view_title=None):
        ax.imshow(ct_slice, cmap='gray', vmin=0, vmax=1)
        ax.imshow(cam_slice, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        ax.axis('off')
        if view_title:
            ax.set_title(view_title, fontsize=8, color='white',
                         backgroundcolor='black', pad=2)

    views = [
        ("Axial",    axial_idxs,
         lambda idx: (volume[idx, :, :], heatmap[idx, :, :])),
        ("Coronal",  coronal_idxs,
         lambda idx: (volume[:, idx, :], heatmap[:, idx, :])),
        ("Sagittal", sagittal_idxs,
         lambda idx: (volume[:, :, idx], heatmap[:, :, idx]))
    ]

    for row_i, (view_name, indices, slicer) in enumerate(views):
        for col_i, idx in enumerate(indices):
            ax = fig.add_subplot(gs[row_i, col_i])
            ct_sl, cam_sl = slicer(idx)
            title = view_name if col_i == 0 else None
            overlay(ax, ct_sl, cam_sl, title)

    # Add colour bar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    sm = plt.cm.ScalarMappable(cmap='jet',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention', fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight',
                    facecolor='#1a1a1a')
        print(f"Grad-CAM saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════
# BATCH EXPLAINABILITY SUMMARY
# ═══════════════════════════════════════════════════════

def explain_batch(model: nn.Module,
                  volumes: torch.Tensor,
                  labels: torch.Tensor,
                  device: str,
                  save_dir: Path,
                  method: str = "gradcam") -> List:
    """
    Run Grad-CAM on a batch of classifier inputs and save visualisations.
    method: "gradcam" or "scorecam"
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    if method == "gradcam":
        cam_fn = GradCAM3D(model, target_layer=model.layer3)
    else:
        cam_fn = ScoreCAM3D(model, target_layer=model.layer3)

    results = []
    for i in range(min(len(volumes), 8)):   # explain up to 8 samples
        vol_t   = volumes[i:i+1].to(device)    # (1, 1, 32, 32, 32)
        label   = int(labels[i].item())

        try:
            heatmap, prob = cam_fn(vol_t)
            vol_np = volumes[i, 0].cpu().numpy()

            fig = visualise_gradcam(
                vol_np, heatmap, prob,
                uid=f"sample_{i}",
                save_path=save_dir / f"gradcam_{i}_label{label}.png"
            )
            plt.close(fig)
            results.append({"idx": i, "prob": prob, "label": label,
                             "heatmap_max": heatmap.max()})
        except Exception as e:
            print(f"  Grad-CAM failed for sample {i}: {e}")

    if hasattr(cam_fn, "remove_hooks"):
        cam_fn.remove_hooks()

    return results


# ═══════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    from typing import List
    print("Grad-CAM 3D — Smoke Test")
    print("=" * 40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ResNet3D(use_se=True).to(device)
    model.eval()

    # Fake input: batch of 1, single channel, 32³
    inp = torch.randn(1, 1, 32, 32, 32).to(device)

    print("Testing Grad-CAM...")
    cam_fn  = GradCAM3D(model, target_layer=model.layer3)
    heatmap, prob = cam_fn(inp)
    print(f"  Heatmap shape: {heatmap.shape}  max: {heatmap.max():.3f}")
    print(f"  Probability:   {prob:.4f}")
    cam_fn.remove_hooks()

    print("Testing Score-CAM...")
    scam    = ScoreCAM3D(model, target_layer=model.layer3, max_channels=16)
    heatmap2, prob2 = scam(inp)
    print(f"  Heatmap shape: {heatmap2.shape}  max: {heatmap2.max():.3f}")
    scam.remove_hooks()

    print("Testing visualisation...")
    vol_np  = inp[0, 0].cpu().numpy()
    fig = visualise_gradcam(vol_np, heatmap, prob, uid="test_uid",
                             save_path=Path("results/gradcam_test.png"))
    plt.close(fig)
    print("✓ Grad-CAM module functional.")
