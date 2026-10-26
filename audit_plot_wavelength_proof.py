#!/usr/bin/env python3
"""Figure: scene median I/F plotted on the pipeline's wavelength axis vs the
empirically recovered (reversed) axis, with known Mars/CRISM features marked."""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crism_utils import get_sensor_waves  # noqa: E402

SCENE = sys.argv[1] if len(sys.argv) > 1 else r"D:\CRISM scenes\scenes\Scene 1"
img = sorted(glob.glob(os.path.join(SCENE, "*l_trr3.img")))[0]
L, S, B = 450, 640, 438

raw = np.memmap(img, dtype="<f4", mode="r", shape=(L * S * B,))
cube = np.transpose(np.asarray(raw).reshape(L, B, S), (0, 2, 1))
cube = np.where((cube > -10) & (cube < 10), cube, np.nan)
med = np.nanmedian(cube[:, 60:580, :].reshape(-1, B), axis=0)

w_pipe = get_sensor_waves(B, None)
w_rev = 3.9312 - 0.006765 * np.arange(B)

FEATURES = [
    (2.01, "CO2 2.01"),
    (2.70, "CO2 2.70 (saturated)"),
    (3.00, "H2O / 3 um"),
    (3.70, "thermal upturn"),
]

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
})
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharey=True)

ax[0].plot(w_pipe, med, color="#f85149", lw=1.3)
ax[0].set_title("Scene 1 median I/F on the wavelength axis the pipeline currently uses\n"
                "(crism_utils.WAVES_L, built from 11 hand-guessed anchors)",
                fontsize=11)
for wl, nm in FEATURES:
    if w_pipe[0] <= wl <= w_pipe[-1]:
        ax[0].axvline(wl, color="#8b949e", ls=":", lw=1)
        ax[0].text(wl, 0.21, nm, rotation=90, va="bottom", ha="right", fontsize=7)
ax[0].text(1.55, 0.02, "near-zero trough the code calls\n'the 1.55 um artifact'",
           fontsize=8, color="#f0883e")

ax[1].plot(w_rev, med, color="#3fb950", lw=1.3)
ax[1].set_title("Same data on the empirically recovered axis "
                "(band 0 = 3.93 um, band 437 = 0.97 um, -6.77 nm/band)",
                fontsize=11)
for wl, nm in FEATURES:
    ax[1].axvline(wl, color="#8b949e", ls=":", lw=1)
    ax[1].text(wl, 0.21, nm, rotation=90, va="bottom", ha="right", fontsize=7)
ax[1].text(2.70, 0.03, "trough is the saturated\n2.70 um CO2 band",
           fontsize=8, color="#3fb950", ha="center")

for a in ax:
    a.set_xlim(0.9, 4.0)
    a.set_ylabel("median I/F")
    a.grid(alpha=0.15)
ax[1].set_xlabel("wavelength (um)")
fig.tight_layout()
out = os.path.join(SCENE, "mineral_output", "fig_wavelength_axis_proof.png")
fig.savefig(out, dpi=150)
print("wrote", out)
