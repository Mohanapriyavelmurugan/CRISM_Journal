import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, "test")
PHYS_DIR = os.path.join(TEST_DIR, "physics_output")
HSID_DIR = os.path.join(TEST_DIR, "hsid_output")
ML_DIR = os.path.join(TEST_DIR, "ml_output")
N2N4M_DIR = os.path.join(SCRIPT_DIR, "results", "n2n4m_output")

sys.path.insert(0, SCRIPT_DIR)
from crism_utils import parse_pds3_label, load_crism_cube, find_label, get_sensor_waves

L_IMG = "frt000098fe_01_if156l_trr3"
S_IMG = "frt000098fe_01_if156s_trr3"

def load_data(img_prefix, is_s_sensor=False):
    img_path = os.path.join(TEST_DIR, img_prefix + ".img")
    lbl_path = find_label(img_path)
    meta = parse_pds3_label(lbl_path)
    orig = load_crism_cube(img_path, meta)
    waves = get_sensor_waves(orig.shape[2], meta.get("wavelengths"))
    
    # Load others
    phys = np.load(os.path.join(PHYS_DIR, img_prefix + "_S3.npy"))
    hsid = np.load(os.path.join(HSID_DIR, img_prefix + "_HSID.npy"))
    prop = np.load(os.path.join(ML_DIR, img_prefix + "_S4.npy"))
    
    if not is_s_sensor:
        n2n4m = np.load(os.path.join(N2N4M_DIR, img_prefix + "_N2N4M.npy"))
    else:
        n2n4m = None
        
    return orig, phys, hsid, prop, n2n4m, waves

print("Loading data...")
orig_L, phys_L, hsid_L, prop_L, n2n4m_L, waves_L = load_data(L_IMG, False)
orig_S, phys_S, hsid_S, prop_S, n2n4m_S, waves_S = load_data(S_IMG, True)

# Select a representative band
b_L = int(np.argmin(np.abs(waves_L - 1.509)))
b_S = int(np.argmin(np.abs(waves_S - 0.750)))

print(f"L band: {b_L}, wavelength: {waves_L[b_L]}")
print(f"S band: {b_S}, wavelength: {waves_S[b_S]}")

L_data = [orig_L[:,:,b_L], phys_L[:,:,b_L], hsid_L[:,:,b_L], prop_L[:,:,b_L], n2n4m_L[:,:,b_L]]
S_data = [orig_S[:,:,b_S], phys_S[:,:,b_S], hsid_S[:,:,b_S], prop_S[:,:,b_S], None]

# Global normalization (across all valid data for identical normalization)
all_valid = []
for d in L_data + S_data:
    if d is not None:
        v = d.ravel()
        all_valid.extend(v[np.isfinite(v)])

vmin = float(np.percentile(all_valid, 1))
vmax = float(np.percentile(all_valid, 99))
print(f"Global norm: [{vmin}, {vmax}]")

# Setup figure
fig, axes = plt.subplots(2, 5, figsize=(14, 4))
fig.suptitle("Visual Comparison of CRISM Denoising Methods", fontsize=14, fontweight='bold', y=0.98)

col_titles = [
    "Original CRISM\nObservation",
    "Physics-Based\nCorrection (S3)",
    "HSID-CNN",
    "Proposed Hybrid\nPipeline",
    "N2N4M"
]

row_labels = ["L Sensor", "S Sensor"]

roi_x, roi_y, roi_w, roi_h = 30, 5, 10, 4

for r in range(2):
    for c in range(5):
        ax = axes[r, c]
        data = L_data[c] if r == 0 else S_data[c]
        
        if data is not None:
            im = ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            
            # ROI rectangle
            rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        else:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, "Not Supported", color="white", ha="center", va="center", transform=ax.transAxes, fontsize=10, fontweight="bold")
            
        # Title for first row
        if r == 0:
            ax.set_title(col_titles[c], fontsize=10, pad=10)
            
        # Row label for first col
        if c == 0:
            ax.set_ylabel(row_labels[r], fontsize=11, fontweight='bold', labelpad=10)
            
        ax.set_xticks([])
        ax.set_yticks([])

# Add a single global colorbar
fig.subplots_adjust(right=0.9, wspace=0.05, hspace=0.1)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Reflectance (I/F)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

OUT_PNG = "visual_comparison.png"
OUT_PDF = "visual_comparison.pdf"
OUT_SVG = "visual_comparison.svg"

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
fig.savefig(OUT_SVG, dpi=600, bbox_inches="tight")

print("Saved figure successfully.")
