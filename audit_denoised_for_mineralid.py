#!/usr/bin/env python3
"""
Audit: is the Stage-4 denoised cube (S4) fit for mineral identification?

Compares raw TRR3 .img, physics-corrected S3 and ML-denoised S4 on the
wavelengths that actually drive the Viviano-Beck indices and library SAM.

Read-only. Writes nothing.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crism_utils import (  # noqa: E402
    build_bad_mask, find_label, get_sensor_waves, load_crism_cube, parse_pds3_label,
)

SCENE = sys.argv[1] if len(sys.argv) > 1 else r"D:\CRISM scenes\scenes\Scene 1"

DIAG = {
    "OLINDEX3 RB1054": 1.054,
    "OLINDEX3 RB1211": 1.211,
    "OLINDEX3 RB1329": 1.329,
    "OLINDEX3 RB1474": 1.474,
    "OLINDEX3 R1750": 1.750,
    "LCPINDEX2 ctr 1810": 1.810,
    "BD1900 ctr 1930": 1.930,
    "HCP/BD2100 2120": 2.120,
    "BD2210 ctr 2210": 2.210,
    "BD2290 ctr 2290": 2.290,
    "BD2300 ctr 2295": 2.295,
    "BD2500 ctr 2500": 2.500,
    "LCP shoulder 2450": 2.450,
    "HCP shoulder 2530": 2.530,
}


def col_stripe_metric(band: np.ndarray) -> float:
    """Std of column means after removing a smooth cross-track trend."""
    cm = np.nanmean(band, axis=0)
    ok = np.isfinite(cm)
    if ok.sum() < 50:
        return float("nan")
    x = np.arange(len(cm))[ok]
    y = cm[ok]
    trend = np.polyval(np.polyfit(x, y, 3), x)
    return float(np.nanstd(y - trend))


def spectral_roughness(cube, bi: np.ndarray, sl) -> float:
    """Mean |2nd spectral difference| at band bi (noise proxy)."""
    a = cube[sl[0], sl[1], bi - 1].astype(np.float64)
    b = cube[sl[0], sl[1], bi].astype(np.float64)
    c = cube[sl[0], sl[1], bi + 1].astype(np.float64)
    d2 = np.abs(b - 0.5 * (a + c))
    return float(np.nanmean(d2))


def spatial_roughness(band: np.ndarray) -> float:
    """Mean |along-track 2nd difference| (noise proxy, cross-track preserved)."""
    d2 = np.abs(band[1:-1, :] - 0.5 * (band[:-2, :] + band[2:, :]))
    return float(np.nanmean(d2))


def main():
    img_files = sorted(glob.glob(os.path.join(SCENE, "*l_trr3.img")))
    s3_files = sorted(glob.glob(os.path.join(SCENE, "physics_output", "**", "*_S3.npy"), recursive=True))
    s4_files = sorted(glob.glob(os.path.join(SCENE, "ml_output", "**", "*_S4.npy"), recursive=True))
    print(f"scene dir : {SCENE}")
    print(f"img       : {img_files}")
    print(f"S3        : {s3_files}")
    print(f"S4        : {s4_files}")
    if not (img_files and s4_files):
        raise SystemExit("need both .img and _S4.npy")

    meta = parse_pds3_label(find_label(img_files[0]))
    img = load_crism_cube(img_files[0], meta).astype(np.float32)
    img = np.where(np.isfinite(img), np.clip(img, -0.05, 1.30), np.nan).astype(np.float32)
    s4 = np.load(s4_files[0], mmap_mode="r")
    s3 = np.load(s3_files[0], mmap_mode="r") if s3_files else None

    H, W, B = img.shape
    waves = get_sensor_waves(B, meta.get("wavelengths"))
    bad = build_bad_mask(waves)
    print(f"\nshape IMG={img.shape}  S4={s4.shape}  S3={None if s3 is None else s3.shape}")
    print(f"waves {waves[0]:.3f}-{waves[-1]:.3f} um | build_bad_mask marks {bad.sum()}/{B} bad")

    # Valid pixels (from good bands of IMG)
    gb = np.where(~bad)[0]
    valid = np.sum(np.isfinite(img[:, :, gb]), axis=2) >= 0.5 * len(gb)
    print(f"valid pixels: {int(valid.sum())}/{H*W}")
    rr, cc = np.where(valid)
    sub = np.random.default_rng(0).choice(len(rr), min(40000, len(rr)), replace=False)
    sl = (rr[sub], cc[sub])

    print("\n" + "=" * 118)
    print(f"{'diagnostic':<22}{'band':>5}{'wl':>8}{'bad?':>6}"
          f"{'finIMG%':>9}{'finS4%':>8}{'medIMG':>9}{'medS4':>9}"
          f"{'S4==S3?':>9}{'spcIMG':>9}{'spcS4':>9}{'spaIMG':>9}{'spaS4':>9}"
          f"{'strIMG':>9}{'strS4':>9}")
    print("=" * 118)
    rows = []
    for nm, wl in DIAG.items():
        bi = int(np.argmin(np.abs(waves - wl)))
        bi = max(1, min(B - 2, bi))
        bimg = img[:, :, bi]
        bs4 = np.asarray(s4[:, :, bi], dtype=np.float32)
        fin_i = 100 * float(np.mean(np.isfinite(bimg[valid])))
        fin_4 = 100 * float(np.mean(np.isfinite(bs4[valid])))
        med_i = float(np.nanmedian(bimg[valid]))
        med_4 = float(np.nanmedian(bs4[valid]))
        if s3 is not None:
            bs3 = np.asarray(s3[:, :, bi], dtype=np.float32)
            d = np.abs(bs4[valid] - bs3[valid])
            same = "YES" if np.nanmax(d) < 1e-7 else f"{np.nanmean(d):.2e}"
        else:
            same = "n/a"
        row = dict(
            name=nm, band=bi, wl=float(waves[bi]), bad=bool(bad[bi]),
            fin_img=fin_i, fin_s4=fin_4, med_img=med_i, med_s4=med_4, s4_eq_s3=same,
            spc_img=spectral_roughness(img, bi, sl), spc_s4=spectral_roughness(s4, bi, sl),
            spa_img=spatial_roughness(bimg), spa_s4=spatial_roughness(bs4),
            str_img=col_stripe_metric(bimg), str_s4=col_stripe_metric(bs4),
        )
        rows.append(row)
        print(f"{nm:<22}{bi:>5}{row['wl']:>8.3f}{('BAD' if row['bad'] else '-'):>6}"
              f"{fin_i:>9.1f}{fin_4:>8.1f}{med_i:>9.4f}{med_4:>9.4f}{same:>9}"
              f"{row['spc_img']:>9.2e}{row['spc_s4']:>9.2e}"
              f"{row['spa_img']:>9.2e}{row['spa_s4']:>9.2e}"
              f"{row['str_img']:>9.2e}{row['str_s4']:>9.2e}")

    # Whole-cube: how many bands were left untouched by Stage 4?
    if s3 is not None:
        print("\nper-band S4-vs-S3 identity scan (untouched = denoiser skipped it):")
        untouched = []
        for bi in range(B):
            a = np.asarray(s3[:, :, bi], dtype=np.float32)[valid]
            b = np.asarray(s4[:, :, bi], dtype=np.float32)[valid]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() == 0:
                untouched.append(bi)
                continue
            if float(np.max(np.abs(a[m] - b[m]))) < 1e-7:
                untouched.append(bi)
        print(f"  untouched bands: {len(untouched)}/{B}")
        # group into wavelength runs
        runs = []
        for bi in untouched:
            if runs and bi == runs[-1][1] + 1:
                runs[-1][1] = bi
            else:
                runs.append([bi, bi])
        for a, b in runs:
            print(f"    bands {a:>3}-{b:>3}  {waves[a]:.3f}-{waves[b]:.3f} um"
                  f"{'   <-- bad_mask' if bad[a] else ''}")

    # NaN geometry
    print("\nNaN / zero structure:")
    for nm, cube in (("IMG", img), ("S4", s4)):
        c = np.asarray(cube[sl[0], sl[1], :], dtype=np.float32)
        print(f"  {nm}: finite={100*np.mean(np.isfinite(c)):.2f}%  "
              f"exact-zero={100*np.mean(c == 0):.2f}%  "
              f"min={np.nanmin(c):.4f} max={np.nanmax(c):.4f}")

    # Scene SNR proxy per cube on good bands
    print("\nSNR proxy (median / spectral-roughness) on good bands:")
    for nm, cube in (("IMG", img), ("S3", s3), ("S4", s4)):
        if cube is None:
            continue
        vals = []
        for bi in gb[::8]:
            if bi < 1 or bi > B - 2:
                continue
            sig = abs(float(np.nanmedian(np.asarray(cube[sl[0], sl[1], bi], dtype=np.float32))))
            nz = spectral_roughness(cube, int(bi), sl)
            if nz > 0 and np.isfinite(sig):
                vals.append(sig / nz)
        print(f"  {nm}: median SNR proxy = {np.nanmedian(vals):.1f}  (n={len(vals)} bands)")

    out = os.path.join(SCENE, "mineral_output", "denoise_audit.json")
    try:
        with open(out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"\n(could not write json: {e})")


if __name__ == "__main__":
    main()
