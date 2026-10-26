#!/usr/bin/env python3
"""Dump scene median spectrum per band to check wavelength/radiometry sanity."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crism_utils import (  # noqa: E402
    build_bad_mask, find_label, get_sensor_waves, load_crism_cube, parse_pds3_label,
)

SCENE = sys.argv[1] if len(sys.argv) > 1 else r"D:\CRISM scenes\scenes\Scene 1"

img_path = sorted(glob.glob(os.path.join(SCENE, "*l_trr3.img")))[0]
lbl = find_label(img_path)
meta = parse_pds3_label(lbl)
print("label wavelengths present:", meta.get("wavelengths") is not None,
      "n =", None if meta.get("wavelengths") is None else len(meta["wavelengths"]))
cube = load_crism_cube(img_path, meta).astype(np.float32)
cube = np.where(np.isfinite(cube), np.clip(cube, -0.05, 1.30), np.nan)
H, W, B = cube.shape
waves = get_sensor_waves(B, meta.get("wavelengths"))
bad = build_bad_mask(waves)

if meta.get("wavelengths") is not None and len(meta["wavelengths"]) == B:
    lw = np.array(meta["wavelengths"], dtype=float)
    print(f"label waves: {lw[0]:.4f} .. {lw[-1]:.4f}  (unit guess: "
          f"{'micron' if lw[-1] < 10 else 'nanometre'})")
    print("first 8:", np.round(lw[:8], 4), " last 5:", np.round(lw[-5:], 4))

# central ROI to avoid edges
cs = cube[:, 60:580, :]
med = np.nanmedian(cs.reshape(-1, B), axis=0)
p10 = np.nanpercentile(cs.reshape(-1, B), 10, axis=0)
p90 = np.nanpercentile(cs.reshape(-1, B), 90, axis=0)
finite = 100 * np.mean(np.isfinite(cs.reshape(-1, B)), axis=0)

print("\n band   wl(um)  bad  finite%     p10      med      p90")
for bi in range(B):
    flag = "BAD" if bad[bi] else "   "
    mark = ""
    if np.isfinite(med[bi]) and bi > 0 and np.isfinite(med[bi - 1]):
        if med[bi - 1] > 1e-6 and abs(med[bi] / med[bi - 1] - 1) > 0.15:
            mark = "  <== JUMP"
    print(f"{bi:>5}  {waves[bi]:>7.4f}  {flag}  {finite[bi]:>6.1f}  "
          f"{p10[bi]:>7.4f}  {med[bi]:>7.4f}  {p90[bi]:>7.4f}{mark}")

print("\nmedian I/F at key wavelengths:")
for wl in (1.05, 1.10, 1.15, 1.21, 1.25, 1.30, 1.33, 1.40, 1.47, 1.50,
           1.55, 1.60, 1.70, 1.75, 1.80, 2.00, 2.12, 2.21, 2.30, 2.50):
    bi = int(np.argmin(np.abs(waves - wl)))
    print(f"  {wl:.2f} um -> band {bi:>3} (w={waves[bi]:.4f}) med={med[bi]:.4f}")
