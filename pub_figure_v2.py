#!/usr/bin/env python3
"""
pub_figure_v2.py  —  Publication-Quality CRISM Denoising Figure (v2)
=====================================================================
Layout: 4 rows (wavelengths) x 5 columns (metrics)
  Col 1: Before denoising    — grayscale, shared reflectance limits
  Col 2: After denoising     — grayscale, shared reflectance limits
  Col 3: Removed component   — diverging RdBu_r, symmetric around 0
  Col 4: Gradient mag before — Viridis, shared gradient limits
  Col 5: Gradient mag after  — Viridis, shared gradient limits

Rows: 750 nm, 1010 nm, 1509 nm, 2018 nm

Each panel has its own individual colorbar.
Column titles on first row only; wavelength labels on left.
Times New Roman font, 600 dpi, white background.
"""

import os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import sobel

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Publication style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":    "stix",        # matches Times in math mode
    "axes.labelsize":      8,
    "axes.titlesize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "figure.dpi":          100,           # screen preview
    "savefig.dpi":         600,
    "pdf.fonttype":        42,            # TrueType embedding for Acrobat
    "ps.fonttype":         42,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.5,
    "ytick.major.width":   0.5,
    "xtick.major.size":    2,
    "ytick.major.size":    2,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "test")
ML_DIR     = os.path.join(DATA_DIR,   "ml_output")

sys.path.insert(0, SCRIPT_DIR)
from crism_utils import (
    parse_pds3_label, load_crism_cube, find_label, get_sensor_waves
)

TARGET_WL_UM = [0.750, 1.010, 1.509, 2.018]
TARGET_WL_NM = [750,   1010,  1509,  2018 ]

L_IMG = "frt0001458f_01_if156l_trr3.img"   # 15 x 64 x 438  L-sensor
S_IMG = "frt0001458f_01_if156s_trr3.img"   # 15 x 64 x 107  S-sensor

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cube loader
# ─────────────────────────────────────────────────────────────────────────────
def load_pair(img_name):
    img_path = os.path.join(DATA_DIR, img_name)
    lbl_path = find_label(img_path)
    meta     = parse_pds3_label(lbl_path)
    orig     = load_crism_cube(img_path, meta)
    waves    = get_sensor_waves(orig.shape[2], meta.get("wavelengths"))
    npy_path = os.path.join(ML_DIR, os.path.splitext(img_name)[0] + "_S4.npy")
    den      = np.load(npy_path).astype(np.float32)
    return orig, den, waves

print("Loading L-sensor ..."); orig_L, den_L, waves_L = load_pair(L_IMG)
print("Loading S-sensor ..."); orig_S, den_S, waves_S = load_pair(S_IMG)
print(f"L: {orig_L.shape}, waves [{waves_L[0]:.3f}, {waves_L[-1]:.3f}] um")
print(f"S: {orig_S.shape}, waves [{waves_S[0]:.3f}, {waves_S[-1]:.3f}] um")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Band extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def nearest_band(waves, wl_um):
    return int(np.argmin(np.abs(waves - wl_um)))

def sobel_mag(img2d):
    """Sobel gradient magnitude; NaN-safe."""
    work = np.where(np.isfinite(img2d), img2d, 0.0)
    mag  = np.hypot(sobel(work, axis=1), sobel(work, axis=0))
    mag[~np.isfinite(img2d)] = np.nan
    return mag

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build per-row data
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for wl_um, wl_nm in zip(TARGET_WL_UM, TARGET_WL_NM):
    if wl_um < 1.07:
        orig, den, waves = orig_S, den_S, waves_S
    else:
        orig, den, waves = orig_L, den_L, waves_L

    b         = nearest_band(waves, wl_um)
    actual_nm = float(waves[b]) * 1000.0
    print(f"  {wl_nm} nm -> band {b}, actual {actual_nm:.1f} nm")

    o2d = orig[:, :, b].copy()
    d2d = den [:, :, b].copy()
    rem = o2d - d2d
    go  = sobel_mag(o2d)
    gd  = sobel_mag(d2d)

    rows.append(dict(
        wl_nm=wl_nm, actual_nm=actual_nm,
        before=o2d, after=d2d, removed=rem,
        grad_before=go, grad_after=gd,
    ))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Global colour limits
# ─────────────────────────────────────────────────────────────────────────────
# Shared reflectance: 1st–99th percentile across all before + after images
all_ref = np.concatenate([
    np.concatenate([r["before"].ravel(), r["after"].ravel()])
    for r in rows
])
valid_ref = all_ref[np.isfinite(all_ref)]
ref_vmin  = float(np.percentile(valid_ref,  1))
ref_vmax  = float(np.percentile(valid_ref, 99))

# Shared gradient limits
all_grad  = np.concatenate([
    np.concatenate([r["grad_before"].ravel(), r["grad_after"].ravel()])
    for r in rows
])
valid_grad = all_grad[np.isfinite(all_grad)]
grad_vmax  = float(np.percentile(valid_grad, 99))

# Per-row symmetric limits for removed component
for r in rows:
    v = r["removed"].ravel()
    v = v[np.isfinite(v)]
    r["rem_abs"] = float(np.percentile(np.abs(v), 99)) if len(v) else 1e-6

print(f"\nReflectance limits : [{ref_vmin:.4f}, {ref_vmax:.4f}] I/F")
print(f"Gradient vmax      : {grad_vmax:.4f}")
rem_abs_vals = [f"{r['rem_abs']:.4f}" for r in rows]
print(f"Removed abs-max    : {rem_abs_vals}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure geometry
# ─────────────────────────────────────────────────────────────────────────────
N_ROWS   = 4    # wavelengths
N_COLS   = 5    # metrics

# Each image panel (inches)
IMG_W    = 1.40
IMG_H    = 0.70   # spatial: 15 rows x 64 cols  → tall/narrow; we show as landscape
CB_FRAC  = 0.06   # colorbar width as fraction of IMG_W
CB_PAD   = 0.03   # pad between image and colorbar (inches)

# Margins (inches)
LEFT_M   = 0.90   # for wavelength labels
RIGHT_M  = 0.20
TOP_M    = 0.60   # for column titles
BOT_M    = 0.25

HGAP     = 0.55   # vertical gap between rows (inches), to accommodate per-panel cbars
WGAP     = 0.55   # horizontal gap between columns

# Derived figure size
FIG_W = LEFT_M + N_COLS * IMG_W + (N_COLS - 1) * WGAP + RIGHT_M
FIG_H = TOP_M  + N_ROWS * IMG_H + (N_ROWS - 1) * HGAP + BOT_M

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Compute axes positions manually (gives pixel-perfect control)
# ─────────────────────────────────────────────────────────────────────────────
def frac(x, total):
    return x / total

# x positions of image left edges (fraction of fig width)
col_x = [frac(LEFT_M + c * (IMG_W + WGAP), FIG_W) for c in range(N_COLS)]
# y positions of image bottom edges (fraction of fig height) — row 0 at top
row_y = [frac(FIG_H - TOP_M - (r + 1) * IMG_H - r * HGAP, FIG_H)
         for r in range(N_ROWS)]

img_w_frac = frac(IMG_W, FIG_W)
img_h_frac = frac(IMG_H, FIG_H)

ax_grid = np.empty((N_ROWS, N_COLS), dtype=object)
for r in range(N_ROWS):
    for c in range(N_COLS):
        ax = fig.add_axes([col_x[c], row_y[r], img_w_frac, img_h_frac])
        ax_grid[r, c] = ax

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Column metadata
# ─────────────────────────────────────────────────────────────────────────────
COL_TITLES = [
    "Before Denoising",
    "After Denoising",
    "Removed Component\n(Before $-$ After)",
    "Gradient Magnitude\n(Before)",
    "Gradient Magnitude\n(After)",
]
COL_CMAPS  = ["gray", "gray", "RdBu_r", "viridis", "viridis"]
COL_LABELS = ["I/F", "I/F", "dI/F", "grad I/F", "grad I/F"]

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Plot every panel + individual colorbar
# ─────────────────────────────────────────────────────────────────────────────
for r, row in enumerate(rows):
    data_list = [
        row["before"],
        row["after"],
        row["removed"],
        row["grad_before"],
        row["grad_after"],
    ]

    for c, (data2d, cmap, cb_label) in enumerate(
            zip(data_list, COL_CMAPS, COL_LABELS)):

        ax = ax_grid[r, c]

        # ── colour limits ──────────────────────────────────────────────────
        if c in (0, 1):
            norm = Normalize(vmin=ref_vmin, vmax=ref_vmax)
        elif c == 2:
            ra = row["rem_abs"]
            norm = TwoSlopeNorm(vmin=-ra, vcenter=0, vmax=ra)
        else:
            norm = Normalize(vmin=0, vmax=grad_vmax)

        # ── image ─────────────────────────────────────────────────────────
        im = ax.imshow(
            data2d,
            cmap          = cmap,
            norm          = norm,
            origin        = "upper",
            aspect        = "auto",
            interpolation = "nearest",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#444444")

        # ── column title (first row only) ──────────────────────────────────
        if r == 0:
            ax.set_title(
                COL_TITLES[c],
                fontsize   = 7.5,
                fontweight = "bold",
                pad        = 4,
                linespacing = 1.3,
            )

        # ── wavelength label (first column only) ──────────────────────────
        if c == 0:
            ax.set_ylabel(
                f"{row['wl_nm']} nm\n({row['actual_nm']:.1f} nm)",
                fontsize  = 7.5,
                labelpad  = 6,
                rotation  = 90,
                va        = "center",
                ha        = "right",
                fontstyle = "italic",
            )

        # ── individual colorbar ────────────────────────────────────────────
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="7%", pad=0.04)
        cb      = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax         = cax,
            orientation = "vertical",
        )
        cb.ax.tick_params(
            labelsize = 5.5,
            width     = 0.4,
            length    = 2,
            pad       = 1,
        )
        cb.outline.set_linewidth(0.4)

        # Tick formatting
        if c in (0, 1):
            ticks = np.linspace(ref_vmin, ref_vmax, 4)
        elif c == 2:
            ra    = row["rem_abs"]
            ticks = np.linspace(-ra, ra, 5)
        else:
            ticks = np.linspace(0, grad_vmax, 4)

        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.3f}" for t in ticks])

        # Colorbar label on last row only, to avoid clutter
        if r == N_ROWS - 1:
            cb.set_label(
                cb_label,
                fontsize  = 6,
                labelpad  = 3,
                rotation  = 270,
                va        = "bottom",
            )

# ─────────────────────────────────────────────────────────────────────────────
# 10. Figure-level text
# ─────────────────────────────────────────────────────────────────────────────
fig.text(
    (LEFT_M - 0.55) / FIG_W, 0.5,
    "Wavelength",
    fontsize  = 8,
    fontweight= "bold",
    rotation  = 90,
    ha        = "center",
    va        = "center",
)

fig.text(
    0.5,
    (FIG_H - TOP_M * 0.30) / FIG_H,
    "CRISM Hyperspectral Denoising — Reflectance and Gradient Comparison",
    fontsize   = 9,
    fontweight = "bold",
    ha         = "center",
    va         = "bottom",
)

fig.text(
    0.5,
    0.007,
    r"MRO/CRISM TRR3 · FRT0001458F · Stage-4 ML denoising (Spectral DAE + Stripe CNN) · Gradient: Sobel operator",
    fontsize  = 5.5,
    ha        = "center",
    va        = "bottom",
    color     = "#555555",
    fontstyle = "italic",
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────────────────
OUT_PNG = os.path.join(SCRIPT_DIR, "crism_pub_figure_v2.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "crism_pub_figure_v2.pdf")

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
print(f"\nPNG saved -> {OUT_PNG}")

fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", facecolor="white")
print(f"PDF saved -> {OUT_PDF}")

plt.close(fig)
print("Done.")
