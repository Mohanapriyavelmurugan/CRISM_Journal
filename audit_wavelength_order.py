#!/usr/bin/env python3
"""
Determine the true band->wavelength mapping of the TRR3 L cube empirically.

Anchors on physics that is unambiguous in CRISM L I/F:
  * very deep atmospheric CO2 band  ~2.70 um
  * narrower atmospheric CO2 band   ~2.01 um
  * dark 3.0-3.5 um region with thermal upturn toward 3.9 um
Then checks which linear mapping (forward or reversed) is consistent.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

SCENE = sys.argv[1] if len(sys.argv) > 1 else r"D:\CRISM scenes\scenes\Scene 1"
img = sorted(glob.glob(os.path.join(SCENE, "*l_trr3.img")))[0]
L, S, B = 450, 640, 438

raw = np.memmap(img, dtype="<f4", mode="r", shape=(L * S * B,))
cube = np.transpose(np.asarray(raw).reshape(L, B, S), (0, 2, 1))
cube = np.where((cube > -10) & (cube < 10), cube, np.nan)
med = np.nanmedian(cube[:, 60:580, :].reshape(-1, B), axis=0)

print("full median spectrum by band index:")
for i in range(0, B, 1):
    if i % 4 == 0:
        print(f"  {i:>3}:{med[i]:7.4f}", end="")
        if (i // 4) % 7 == 6:
            print()
print("\n")

ok = np.isfinite(med)
idx = np.where(ok)[0]
print(f"finite bands: {idx.min()}..{idx.max()}  (n={len(idx)})")
print(f"global minimum: band {int(idx[np.argmin(med[idx])])} value {np.nanmin(med):.5f}")

# local minima of a lightly smoothed curve
sm = med.copy()
for _ in range(2):
    sm = np.where(np.isfinite(sm), sm, np.nan)
    sm = np.array([np.nanmean(sm[max(0, i - 1):i + 2]) for i in range(B)])
mins, maxs = [], []
for i in range(2, B - 2):
    w = sm[i - 2:i + 3]
    if not np.all(np.isfinite(w)):
        continue
    if sm[i] == np.min(w) and sm[i] < 0.95 * max(sm[i - 2], sm[i + 2]):
        mins.append(i)
    if sm[i] == np.max(w) and sm[i] > 1.05 * min(sm[i - 2], sm[i + 2]):
        maxs.append(i)


def group(a):
    out = []
    for v in a:
        if out and v - out[-1][-1] <= 3:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(np.mean(g)) for g in out]


mins, maxs = group(mins), group(maxs)
print(f"\nlocal minima bands : {mins}")
print(f"  values           : {[round(float(sm[m]), 4) for m in mins]}")
print(f"local maxima bands : {maxs}")

# Candidate mappings
print("\n--- candidate mapping tests ---")
for label, w0, dw in (
    ("forward  1.0021 + 6.55nm", 1.0021, +0.00655),
    ("forward  1.0021 + 6.63nm", 1.0021, +0.00663),
    ("reversed 3.9200 - 6.55nm", 3.9200, -0.00655),
    ("reversed 3.9200 - 6.63nm", 3.9200, -0.00663),
):
    w = w0 + dw * np.arange(B)
    print(f"\n{label}:  band0={w[0]:.3f}  band437={w[-1]:.3f}")
    print(f"   deepest min  band {int(idx[np.argmin(med[idx])])} -> {w[int(idx[np.argmin(med[idx])])]:.3f} um")
    for m in mins:
        print(f"   local min    band {m:>3} -> {w[m]:.3f} um   (I/F={sm[m]:.4f})")

# Free fit: assume the two deepest minima are the 2.70 and 2.01 um CO2 bands
if len(mins) >= 2:
    order = sorted(mins, key=lambda m: sm[m])
    b_deep = order[0]
    cands = [m for m in mins if abs(m - b_deep) > 40]
    if cands:
        b_sec = sorted(cands, key=lambda m: sm[m])[0]
        dw_fit = (2.01 - 2.70) / (b_sec - b_deep)
        w0_fit = 2.70 - dw_fit * b_deep
        print(f"\n--- free 2-point fit (band {b_deep}=2.70um, band {b_sec}=2.01um) ---")
        print(f"   dispersion = {1000*dw_fit:+.3f} nm/band")
        print(f"   band0 = {w0_fit:.4f} um, band437 = {w0_fit + dw_fit*437:.4f} um")
        wf = w0_fit + dw_fit * np.arange(B)
        for m in mins:
            print(f"   local min band {m:>3} -> {wf[m]:.3f} um  (I/F={sm[m]:.4f})")
