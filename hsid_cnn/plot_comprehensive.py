"""
plot_comprehensive.py  —  Comprehensive Evaluation Figures for HSI Denoising
=============================================================================
Generates:
  1. Multi-panel comparison: Original -> Physics -> HSID -> Proposed -> Differences -> Gradients -> Zoom
  2. Spectral Signature Preservation: Extracts and compares spectra across regions.
"""

import os, sys, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hsid_cnn.config import FIGURES_DIR

# ── matplotlib setup for IEEE/Springer ────────────────────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 600,
})

def compute_gradient_mag(img):
    dx = sobel(img, axis=1)
    dy = sobel(img, axis=0)
    return np.hypot(dx, dy)

def extract_regions(cube):
    H, W, _ = cube.shape
    return [
        (H//2, W//2),
        (H//4, W//4),
        (3*H//4, W//4),
        (H//4, 3*W//4),
        (3*H//4, 3*W//4)
    ]

def plot_comprehensive_figure(s2, s3, hsid, s4, stem, band_idx, out_dir, n2n=None):
    fig, axes = plt.subplots(4, 5, figsize=(12.5, 10))
    
    images = [s2[:,:,band_idx] if s2 is not None else np.zeros_like(s3[:,:,band_idx]), 
              s3[:,:,band_idx], 
              hsid[:,:,band_idx], 
              s4[:,:,band_idx],
              n2n[:,:,band_idx] if n2n is not None else np.zeros_like(s3[:,:,band_idx])]
              
    titles = ["Original Noisy (S2)" if s2 is not None else "N/A", 
              "Physics Output (S3)", 
              "HSID-CNN", 
              "Proposed Hybrid (S4)",
              "N2N4M (Platt 2025)" if n2n is not None else "N/A"]
    
    vmin = np.nanpercentile(s3[:,:,band_idx], 2)
    vmax = np.nanpercentile(s3[:,:,band_idx], 98)
    
    for i in range(5):
        ax = axes[0, i]
        im = ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
        ax.axis('off')
        if i == 4: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ref_idx = 0 if s2 is not None else 1
    diff_max = np.nanpercentile(np.abs(images[3] - images[ref_idx]), 99)
    for i in range(5):
        ax = axes[1, i]
        if i == ref_idx or (s2 is None and i == 0) or (i == 4 and n2n is None):
            ax.axis('off')
            ax.set_title("N/A")
            continue
        diff = images[i] - images[ref_idx]
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        ax.axis('off')
        if i == 4: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    grad_vmax = np.nanpercentile(compute_gradient_mag(images[3]), 95)
    for i in range(5):
        ax = axes[2, i]
        if (s2 is None and i == 0) or (i == 4 and n2n is None):
            ax.axis('off')
            continue
        grad = compute_gradient_mag(images[i])
        im = ax.imshow(grad, cmap='viridis', vmin=0, vmax=grad_vmax)
        ax.axis('off')
        if i == 4: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    regs = extract_regions(s3)
    r, c = regs[0]
    w = 16
    for i in range(5):
        ax = axes[3, i]
        if (s2 is None and i == 0) or (i == 4 and n2n is None):
            ax.axis('off')
            continue
        zoom = images[i][max(0, r-w):min(s3.shape[0], r+w), max(0, c-w):min(s3.shape[1], c+w)]
        im = ax.imshow(zoom, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if i == 4: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    savename = os.path.join(out_dir, f'{stem}_comprehensive_B{band_idx}.png')
    plt.savefig(savename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {savename}")

def plot_spectra(s2, s3, hsid, s4, stem, out_dir, n2n=None):
    regs = extract_regions(s3)
    fig, axes = plt.subplots(len(regs), 1, figsize=(6, 2 * len(regs)))
    
    B = s3.shape[2]
    waves = np.linspace(0.4, 2.5, B)
    
    for i, (r, c) in enumerate(regs):
        ax = axes[i]
        
        if s2 is not None:
            ax.plot(waves, s2[r, c, :], label="Original Noisy (S2)", color='black', alpha=0.3, linewidth=1)
        ax.plot(waves, s3[r, c, :], label="Physics (S3)", color='green', alpha=0.5, linewidth=1.5)
        ax.plot(waves, hsid[r, c, :], label="HSID-CNN", color='orange', alpha=0.8, linewidth=1.5)
        ax.plot(waves, s4[r, c, :], label="Proposed (S4)", color='red', alpha=0.9, linewidth=1.5)
        if n2n is not None:
            ax.plot(waves, n2n[r, c, :], label="N2N4M", color='blue', alpha=0.7, linewidth=1.5)
            
        ax.set_ylabel("Reflectance")
        if i == len(regs) - 1:
            ax.set_xlabel("Wavelength (µm)")
            
        idx_1um = np.argmin(np.abs(waves - 1.0))
        idx_start, idx_end = max(0, idx_1um - 10), min(B, idx_1um + 10)
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="30%", height="40%", loc='lower right')
        if s2 is not None:
            axins.plot(waves[idx_start:idx_end], s2[r, c, idx_start:idx_end], color='black', alpha=0.3)
        axins.plot(waves[idx_start:idx_end], s3[r, c, idx_start:idx_end], color='green', alpha=0.5)
        axins.plot(waves[idx_start:idx_end], hsid[r, c, idx_start:idx_end], color='orange', alpha=0.8)
        axins.plot(waves[idx_start:idx_end], s4[r, c, idx_start:idx_end], color='red', alpha=0.9)
        if n2n is not None:
            axins.plot(waves[idx_start:idx_end], n2n[r, c, idx_start:idx_end], color='blue', alpha=0.7)
            
        axins.tick_params(labelleft=False, labelbottom=False)
        ax.indicate_inset_zoom(axins, edgecolor="black")

    axes[0].legend(loc="upper left")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{stem}_spectral_preservation.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--s2', required=True, help="Path to S2 (noisy) or None")
    p.add_argument('--s3', required=True, help="Path to S3 (physics) cube")
    p.add_argument('--hsid', required=True, help="Path to HSID-CNN clean cube")
    p.add_argument('--s4', required=True, help="Path to S4 (Proposed) clean cube")
    p.add_argument('--stem', default="example")
    p.add_argument('--n2n', default=None, help="Path to N2N4M clean cube or None")
    args = p.parse_args()
    
    s2 = np.load(args.s2) if args.s2 != "None" and os.path.exists(args.s2) else None
    s3 = np.load(args.s3)
    hsid = np.load(args.hsid)
    s4 = np.load(args.s4)
    n2n = np.load(args.n2n) if args.n2n and os.path.exists(args.n2n) else None
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    B_TOTAL = s3.shape[2]
    eval_bands = [10, B_TOTAL//2, B_TOTAL-10]
    
    for b in eval_bands:
        plot_comprehensive_figure(s2, s3, hsid, s4, args.stem, b, FIGURES_DIR, n2n=n2n)
        
    plot_spectra(s2, s3, hsid, s4, args.stem, FIGURES_DIR, n2n=n2n)
