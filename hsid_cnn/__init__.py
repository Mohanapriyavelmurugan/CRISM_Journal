"""
hsid_cnn  —  HSID-CNN implementation for CRISM / M³ hyperspectral denoising
=============================================================================
Based on: Yuan et al., "Hyperspectral Image Denoising Employing a
Spatial–Spectral Deep Residual CNN", IEEE TGRS 2019.

Adapted for:
  • CRISM TRR3 (L-sensor 438 bands, S-sensor 107 bands)
  • M³ (Moon Mineralogy Mapper)
  • Self-supervised training on CRISM S3→S4 pairs
"""

from .model   import HSIDCNN
from .metrics import compute_mpsnr, compute_mssim, compute_msa, compute_sam, compute_snr

__version__ = "1.0.0"
__all__ = ["HSIDCNN", "compute_mpsnr", "compute_mssim", "compute_msa",
           "compute_sam", "compute_snr"]
