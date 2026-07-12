"""
plot_figures.py  —  Generate publication-quality figures for HSID-CNN
======================================================================
Generates the side-by-side comparisons of Noisy vs. Denoised and error residuals.
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hsid_cnn.config import FIGURES_DIR, TARGET_WL

# Set pub-quality parameters
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 600,
})

def plot_hsid_results(noisy_cube: np.ndarray, clean_cube: np.ndarray, stem: str, bands: list[int]):
    """
    Generates a figure showing Original, HSID-Denoised, and Removed Noise
    for selected bands.
    """
    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 3, figsize=(9, 2.5 * n_bands))
    
    for i, b in enumerate(bands):
        orig = noisy_cube[:, :, b]
        clean = clean_cube[:, :, b]
        noise = orig - clean
        
        vmin = np.nanpercentile(orig, 1)
        vmax = np.nanpercentile(orig, 99)
        n_max = np.nanpercentile(np.abs(noise), 99)
        
        # Original
        ax = axes[i, 0]
        im = ax.imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"Noisy (Band {b})")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Clean
        ax = axes[i, 1]
        im = ax.imshow(clean, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"HSID-CNN Denoised")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Noise Removed
        ax = axes[i, 2]
        im = ax.imshow(noise, cmap='RdBu_r', vmin=-n_max, vmax=n_max)
        ax.set_title(f"Removed Noise")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f'{stem}_hsid_pub.png')
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved publication figure: {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--noisy', required=True, help="Path to noisy cube (.npy)")
    p.add_argument('--clean', required=True, help="Path to HSID-CNN clean cube (.npy)")
    p.add_argument('--stem',  default='example', help="Output file prefix")
    p.add_argument('--bands', type=str, default='10,50,100', help="Comma-separated band indices")
    args = p.parse_args()
    
    noisy = np.load(args.noisy)
    clean = np.load(args.clean)
    bands = [int(x) for x in args.bands.split(',')]
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_hsid_results(noisy, clean, args.stem, bands)
