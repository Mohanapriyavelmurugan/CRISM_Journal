#!/usr/bin/env python3
"""
1. Confirm reversed band order using anchors NOT used in the fit
   (all-NaN saturated CO2 core, 3 um hydration, thermal upturn).
2. Quantify, for every Viviano-Beck index the pipeline computes, which band it
   currently reads and what wavelength that band truly is.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crism_utils import build_bad_mask, get_sensor_waves  # noqa: E402

SCENE = sys.argv[1] if len(sys.argv) > 1 else r"D:\CRISM scenes\scenes\Scene 1"
img = sorted(glob.glob(os.path.join(SCENE, "*l_trr3.img")))[0]
L, S, B = 450, 640, 438

raw = np.memmap(img, dtype="<f4", mode="r", shape=(L * S * B,))
cube = np.transpose(np.asarray(raw).reshape(L, B, S), (0, 2, 1))
cube = np.where((cube > -10) & (cube < 10), cube, np.nan)

core = cube[:, 60:580, :]
finite_frac = np.mean(np.isfinite(core.reshape(-1, B)), axis=0)
allnan = np.where(finite_frac < 0.01)[0]
print("bands with <1% finite data (instrument-flagged invalid):")
runs = []
for bi in allnan:
    if runs and bi == runs[-1][1] + 1:
        runs[-1][1] = bi
    else:
        runs.append([bi, bi])
print("  ", [(int(a), int(b)) for a, b in runs])

# Reversed axis from the earlier 2-point CO2 fit
DW, W0 = -0.006765, 3.9312
w_rev = W0 + DW * np.arange(B)
w_fwd = get_sensor_waves(B, None)  # what the pipeline currently uses

print("\nindependent anchor check (not used in the fit):")
for a, b in runs:
    print(f"  invalid run bands {a:>3}-{b:<3} -> reversed {w_rev[b]:.3f}-{w_rev[a]:.3f} um"
          f"   | pipeline says {w_fwd[a]:.3f}-{w_fwd[b]:.3f} um")

med = np.nanmedian(core.reshape(-1, B), axis=0)
print("\n  3 um hydration region (reversed bands ~130-160):")
for bi in (130, 140, 150, 160):
    print(f"    band {bi}: reversed {w_rev[bi]:.3f} um  I/F={med[bi]:.4f}"
          f"   | pipeline {w_fwd[bi]:.3f} um")
print("  thermal upturn (reversed bands ~50 -> 4):")
for bi in (50, 40, 30, 20, 10, 4):
    print(f"    band {bi:>3}: reversed {w_rev[bi]:.3f} um  I/F={med[bi]:.4f}")

# ---- damage table ----
SPM_INDICES = {
    "BD1900_2": (1.930, 1.857, 2.067),
    "BD2100_2": (2.132, 1.930, 2.250),
    "BD2210_2": (2.210, 2.165, 2.250),
    "BD2290": (2.290, 2.250, 2.350),
    "BD2300": (2.295, 2.120, 2.530),
    "BD2500_2": (2.500, 2.380, 2.570),
    "OLINDEX3": (1.330, 1.054, 1.750),
    "LCPINDEX2": (1.810, 1.560, 2.450),
    "HCPINDEX2": (2.120, 1.810, 2.530),
    "SINDEX2": (2.100, 1.815, 2.200),
    "BD1500_2": (1.510, 1.367, 1.810),
    "BD1435": (1.435, 1.370, 1.470),
}
bad = build_bad_mask(w_fwd)
print("\n" + "=" * 104)
print("WHICH WAVELENGTH EACH INDEX ACTUALLY SAMPLES")
print(f"{'index':<11}{'role':<8}{'want':>7}{'band':>6}{'pipeline wl':>13}{'TRUE wl':>10}{'error':>9}")
print("=" * 104)
for nm, (c, l, r) in SPM_INDICES.items():
    for role, wl in (("centre", c), ("left", l), ("right", r)):
        cand = np.where(~bad)[0]
        bi = int(cand[np.argmin(np.abs(w_fwd[cand] - wl))])
        print(f"{nm:<11}{role:<8}{wl:>7.3f}{bi:>6}{w_fwd[bi]:>13.3f}{w_rev[bi]:>10.3f}"
              f"{w_rev[bi]-wl:>+9.3f}")
    print("-" * 104)
