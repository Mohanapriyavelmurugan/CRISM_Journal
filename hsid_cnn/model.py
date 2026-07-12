"""
model.py  —  HSID-CNN PyTorch Implementation
=============================================
Faithful implementation of Yuan et al. (IEEE TGRS 2019):
  "Hyperspectral Image Denoising Employing a Spatial–Spectral
   Deep Residual Convolutional Neural Network"

Architecture:
  1. Joint Spatial–Spectral Multiscale Feature Extraction
     • 2D branch: 3 parallel Conv2d (k=3,5,7) on target band
     • 3D branch: 3 parallel Conv3d (k=3,5,7) on K adjacent bands
     • Outputs concatenated → 120 channels
  2. Deep CNN Backbone (9 Conv-BN-ReLU layers)
     • Feature maps saved at layers 3, 5, 7, 9
  3. Multilevel Feature Representation
     • Concat(f3, f5, f7, f9) → 1×1 Conv → residual output
  4. Residual output: x̂ = y_spatial − φ  (noise subtraction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hsid_cnn.config import (
    K_ADJACENT, N_FEAT_SCALE, N_BACKBONE_CH, N_SKIP_LAYERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MULTISCALE FEATURE EXTRACTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleFeatureExtractor(nn.Module):
    """
    Joint spatial–spectral multiscale feature extraction.

    2D branch  : three 2D convolutions (k=3,5,7) on the single target band.
                 Input:  (B, 1,    H, W)
                 Output: (B, 3×Nf, H, W)

    3D branch  : three 3D convolutions (k=3,5,7 in H,W; full depth K in spectral).
                 Input:  (B, 1, K, H, W)     (K is the "depth" dimension for Conv3d)
                 Output: (B, 3×Nf, H, W)     (depth dimension collapsed → 1 after 3D conv)

    Concatenated output: (B, 6×Nf, H, W)  →  120 channels when Nf=20.
    """

    def __init__(self, K: int = K_ADJACENT, n_feat: int = N_FEAT_SCALE):
        super().__init__()
        self.K = K

        # ── 2D Spatial Branch ──────────────────────────────────────────────
        # Each conv preserves H, W via appropriate padding
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True),
        )
        self.conv2d_5 = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True),
        )
        self.conv2d_7 = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True),
        )

        # ── 3D Spectral–Spatial Branch ─────────────────────────────────────
        # Input layout for Conv3d: (B, C_in=1, D=K, H, W)
        # Kernel: (kD, kH, kW).  We use kD=K to collapse the spectral dim,
        # kH=kW=k for multiscale spatial.
        # After conv: (B, n_feat, 1, H, W) → squeeze → (B, n_feat, H, W)
        #
        # For small K: ensure kernel depth ≤ K via min(K, k) trick.
        kD = min(K, K)   # full spectral depth → always K
        pad_D = 0        # no padding in spectral direction (collapse fully)

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(1, n_feat, kernel_size=(kD, 3, 3),
                      padding=(pad_D, 1, 1), bias=False),
            nn.BatchNorm3d(n_feat),
            nn.ReLU(inplace=True),
        )
        self.conv3d_5 = nn.Sequential(
            nn.Conv3d(1, n_feat, kernel_size=(kD, 5, 5),
                      padding=(pad_D, 2, 2), bias=False),
            nn.BatchNorm3d(n_feat),
            nn.ReLU(inplace=True),
        )
        self.conv3d_7 = nn.Sequential(
            nn.Conv3d(1, n_feat, kernel_size=(kD, 7, 7),
                      padding=(pad_D, 3, 3), bias=False),
            nn.BatchNorm3d(n_feat),
            nn.ReLU(inplace=True),
        )

        self.out_channels = 6 * n_feat   # 120 when n_feat=20

    def forward(self, spatial: torch.Tensor,
                spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
          spatial : (B, 1,  H, W)   — single target band
          spectral: (B, K,  H, W)   — K adjacent bands
        Returns:
          (B, 6*n_feat, H, W)
        """
        # ── 2D branch ──────────────────────────────────────────────────────
        f2d_3 = self.conv2d_3(spatial)   # (B, n_feat, H, W)
        f2d_5 = self.conv2d_5(spatial)
        f2d_7 = self.conv2d_7(spatial)

        # ── 3D branch: reshape spectral → (B, 1, K, H, W) ─────────────────
        B, K, H, W = spectral.shape
        sp3d = spectral.unsqueeze(1)     # (B, 1, K, H, W)

        f3d_3 = self.conv3d_3(sp3d)     # (B, n_feat, 1, H, W) if kD=K
        f3d_5 = self.conv3d_5(sp3d)
        f3d_7 = self.conv3d_7(sp3d)

        # Collapse spectral dimension if it remains (size=1 after full-depth conv)
        f3d_3 = f3d_3.squeeze(2)        # (B, n_feat, H, W)
        f3d_5 = f3d_5.squeeze(2)
        f3d_7 = f3d_7.squeeze(2)

        return torch.cat([f2d_3, f2d_5, f2d_7, f3d_3, f3d_5, f3d_7], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DEEP CNN BACKBONE  (9 Conv-BN-ReLU layers)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k,
                      padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DeepBackbone(nn.Module):
    """
    Nine Conv-BN-ReLU layers. Saves feature maps at layers 3, 5, 7, 9
    for the multilevel skip-connection block.

    Input : (B, 120, H, W)
    Output: (B, n_ch, H, W) + list of skip tensors [(B, n_ch, H, W) × 4]
    """

    def __init__(self, in_ch: int = 120, n_ch: int = N_BACKBONE_CH,
                 skip_layers: list[int] = N_SKIP_LAYERS):
        super().__init__()
        self.skip_layers = set(skip_layers)
        layers = []
        for i in range(1, 10):      # layers 1..9
            ic = in_ch if i == 1 else n_ch
            layers.append(_ConvBNReLU(ic, n_ch))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        skips = {}
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if i in self.skip_layers:
                skips[i] = x
        ordered_skips = [skips[k] for k in sorted(self.skip_layers)]
        return x, ordered_skips


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MULTILEVEL FEATURE REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────

class MultilevelHead(nn.Module):
    """
    Concatenates 4 skip feature maps, projects to 1-channel noise estimate φ.

    f_c = Concat{f3, f5, f7, f9}   (4 × n_ch channels)
    φ   = Conv1×1(f_c)              (1 channel)
    """

    def __init__(self, n_ch: int = N_BACKBONE_CH, n_skips: int = 4):
        super().__init__()
        in_ch = n_ch * n_skips
        self.proj = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        # All skips have same H, W; ensure alignment
        f_c = torch.cat(skips, dim=1)   # (B, 4*n_ch, H, W)
        return self.proj(f_c)           # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL HSID-CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HSIDCNN(nn.Module):
    """
    Full HSID-CNN as described in Yuan et al. (IEEE TGRS 2019).

    Forward pass:
      1. MultiScaleFeatureExtractor(spatial, spectral) → 120-ch features
      2. DeepBackbone(features)   → 9 layers, skip maps at 3,5,7,9
      3. MultilevelHead(skips)    → noise estimate φ  (B, 1, H, W)
      4. Residual: x̂ = spatial − φ

    Args:
      K        : number of adjacent bands (default 24)
      n_feat   : channels per kernel in multiscale block (default 20)
      n_ch     : backbone channels (default 64)
    """

    def __init__(self,
                 K:      int = K_ADJACENT,
                 n_feat: int = N_FEAT_SCALE,
                 n_ch:   int = N_BACKBONE_CH):
        super().__init__()
        self.K = K
        self.extractor = MultiScaleFeatureExtractor(K, n_feat)
        self.backbone  = DeepBackbone(
            in_ch      = self.extractor.out_channels,
            n_ch       = n_ch,
            skip_layers= N_SKIP_LAYERS,
        )
        self.head      = MultilevelHead(n_ch, n_skips=len(N_SKIP_LAYERS))

    def forward(self, spatial: torch.Tensor,
                spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
          spatial : (B, 1, H, W)
          spectral: (B, K, H, W)
        Returns:
          denoised: (B, 1, H, W)
        """
        feats          = self.extractor(spatial, spectral)  # (B,120,H,W)
        _, skips       = self.backbone(feats)               # list of 4 skip maps
        noise_est      = self.head(skips)                   # (B,1,H,W)
        return spatial - noise_est                          # residual subtraction

    def noise_estimate(self, spatial: torch.Tensor,
                       spectral: torch.Tensor) -> torch.Tensor:
        """Return the raw noise estimate φ (useful for analysis)."""
        feats     = self.extractor(spatial, spectral)
        _, skips  = self.backbone(feats)
        return self.head(skips)

    # ── Parameter count ───────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = self.count_parameters()
        return (f"HSIDCNN(K={self.K}, "
                f"extractor_out={self.extractor.out_channels}, "
                f"backbone_ch={self.backbone.layers[0][0].out_channels}, "
                f"params={total:,})")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  WEIGHT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_model(K: int = K_ADJACENT,
                n_feat: int = N_FEAT_SCALE,
                n_ch:   int = N_BACKBONE_CH) -> HSIDCNN:
    """Build and initialise a fresh HSID-CNN model."""
    model = HSIDCNN(K=K, n_feat=n_feat, n_ch=n_ch)
    model.apply(_init_weights)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = build_model()
    print(model)

    B, K, H, W = 4, K_ADJACENT, 15, 32
    spatial  = torch.randn(B, 1, H, W)
    spectral = torch.randn(B, K, H, W)

    out = model(spatial, spectral)
    print(f"Input  spatial : {spatial.shape}")
    print(f"Input  spectral: {spectral.shape}")
    print(f"Output denoised: {out.shape}")
    assert out.shape == (B, 1, H, W), "Shape mismatch!"
    print("✓ Forward pass OK")
