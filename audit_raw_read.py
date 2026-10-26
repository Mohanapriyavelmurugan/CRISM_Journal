#!/usr/bin/env python3
"""Independent BIL read of the TRR3 .img, compared against crism_utils.load_crism_cube."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crism_utils import find_label, load_crism_cube, parse_pds3_label  # noqa: E402

SCENE = sys.argv[1] if len(sys.argv) > 1 else r"D:\CRISM scenes\scenes\Scene 1"
img = sorted(glob.glob(os.path.join(SCENE, "*l_trr3.img")))[0]

L, S, B = 450, 640, 438
n_img = L * S * B

# Ground truth per label: RECORD_BYTES=2560, BIL => (line, band, sample)
raw = np.memmap(img, dtype="<f4", mode="r", shape=(n_img,))
bil = np.asarray(raw).reshape(L, B, S)          # line, band, sample
cube_true = np.transpose(bil, (0, 2, 1))        # line, sample, band
cube_true = np.where((cube_true > -10) & (cube_true < 10), cube_true, np.nan)

meta = parse_pds3_label(find_label(img))
print("parsed meta:", {k: meta.get(k) for k in
                       ("lines", "line_samples", "bands", "sample_type",
                        "band_storage", "sample_bits", "unit")})
cube_pipe = load_crism_cube(img, meta).astype(np.float32)
print("pipeline cube shape:", cube_pipe.shape, " independent BIL shape:", cube_true.shape)

a = np.nan_to_num(cube_pipe, nan=-999)
b = np.nan_to_num(cube_true.astype(np.float32), nan=-999)
print("identical to independent BIL read:", bool(np.array_equal(a, b)))
if not np.array_equal(a, b):
    d = np.abs(np.where((a > -900) & (b > -900), a - b, 0))
    print("  max abs diff:", float(d.max()), " frac differing:",
          float(np.mean(d > 1e-6)))

# Median spectrum from the independent read
mid = cube_true[:, 60:580, :].reshape(-1, B)
med = np.nanmedian(mid, axis=0)
print("\nindependent-BIL median I/F by band index (every 10th):")
for bi in range(0, B, 10):
    print(f"  band {bi:>3}: {med[bi]:.4f}")

print("\nROWNUM table (detector rows actually downlinked):")
tail = np.memmap(img, dtype=">u2", mode="r", offset=197100 * 2560, shape=(438,))
rows = np.asarray(tail) & 0x1FF
print("  first 12:", rows[:12], " last 12:", rows[-12:])
print("  monotonic increasing:", bool(np.all(np.diff(rows.astype(int)) > 0)))
print("  min/max:", int(rows.min()), int(rows.max()))
