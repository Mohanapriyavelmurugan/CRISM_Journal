"""
metrics.py  —  Evaluation metrics for HSI denoising
=====================================================
Implements:
  • MPSNR  — Mean Peak Signal-to-Noise Ratio  (higher = better)
  • MSSIM  — Mean Structural Similarity Index (higher = better)
  • MSA    — Mean Spectral Angle (radians)     (lower  = better)
  • SAM    — Spectral Angle Mapper (per-pixel mean, radians)
  • SNR    — Signal-to-Noise Ratio (|mean|/std, higher = better)
  • PSNR   — Per-band PSNR for band-level plots
  • SSIM   — Per-band SSIM for band-level plots
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _valid_mask(cube: np.ndarray,
                if_min: float = -0.05,
                if_max: float =  1.30) -> np.ndarray:
    """Boolean mask: finite and within physical I/F range."""
    return np.isfinite(cube) & (cube > if_min) & (cube < if_max)


def _band_range(cube: np.ndarray) -> float:
    """Peak signal value: 99th percentile of valid pixels."""
    v = cube[np.isfinite(cube)]
    return float(np.percentile(v, 99)) if v.size else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PER-BAND METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr_per_band(clean: np.ndarray,
                           restored: np.ndarray,
                           data_range: float | None = None) -> np.ndarray:
    """
    Per-band PSNR (dB).
    clean, restored : (H, W, B) float arrays.
    """
    H, W, B  = clean.shape
    if data_range is None:
        data_range = max(_band_range(clean), 1e-6)

    psnr = np.zeros(B)
    for b in range(B):
        c  = clean[:, :, b]
        r  = restored[:, :, b]
        ok = np.isfinite(c) & np.isfinite(r)
        if ok.sum() == 0:
            psnr[b] = np.nan
            continue
        mse = float(np.mean((c[ok] - r[ok]) ** 2))
        psnr[b] = (10.0 * np.log10(data_range ** 2 / (mse + 1e-12))
                   if mse > 0 else 100.0)
    return psnr


def compute_ssim_per_band(clean: np.ndarray,
                           restored: np.ndarray,
                           data_range: float | None = None) -> np.ndarray:
    """Per-band SSIM."""
    _, _, B = clean.shape
    if data_range is None:
        data_range = max(_band_range(clean), 1e-6)

    ssim = np.zeros(B)
    for b in range(B):
        c  = clean[:, :, b]
        r  = restored[:, :, b]
        ok = np.isfinite(c) & np.isfinite(r)
        if ok.sum() < 4 or c.shape[0] < 2 or c.shape[1] < 2:
            ssim[b] = np.nan
            continue
        try:
            ssim[b] = _ssim(c, r, data_range=data_range)
        except Exception:
            ssim[b] = np.nan
    return ssim


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mpsnr(clean: np.ndarray, restored: np.ndarray,
                  data_range: float | None = None) -> float:
    """
    MPSNR — Mean PSNR across all bands (NaN bands excluded).
    Defined in [45] of the paper.
    """
    psnr = compute_psnr_per_band(clean, restored, data_range)
    valid = psnr[np.isfinite(psnr)]
    return float(valid.mean()) if valid.size else np.nan


def compute_mssim(clean: np.ndarray, restored: np.ndarray,
                  data_range: float | None = None) -> float:
    """MSSIM — Mean SSIM across all bands."""
    ssim = compute_ssim_per_band(clean, restored, data_range)
    valid = ssim[np.isfinite(ssim)]
    return float(valid.mean()) if valid.size else np.nan


def compute_msa(clean: np.ndarray, restored: np.ndarray,
                good_mask: np.ndarray | None = None) -> float:
    """
    MSA — Mean Spectral Angle (radians).
    Computed on the scene-median spectra (as per the paper's Eq. in Sec IV).

    Lower is better; 0 = identical spectral shape.
    """
    H, W, B = clean.shape
    ok = np.isfinite(clean) & np.isfinite(restored)
    if good_mask is not None:
        # Only use good (non-bad) bands
        ok[:, :, ~good_mask] = False

    c_flat = clean.reshape(-1, B).astype(float)
    r_flat = restored.reshape(-1, B).astype(float)

    pix_ok = np.all(ok.reshape(-1, B), axis=1)
    c_med  = np.nanmedian(c_flat[pix_ok], axis=0)
    r_med  = np.nanmedian(r_flat[pix_ok], axis=0)

    dot  = float(np.dot(c_med, r_med))
    norm = (np.linalg.norm(c_med) * np.linalg.norm(r_med) + 1e-12)
    return float(np.arccos(np.clip(dot / norm, -1 + 1e-6, 1 - 1e-6)))


def compute_sam(clean: np.ndarray, restored: np.ndarray,
                good_mask: np.ndarray | None = None) -> float:
    """
    SAM — per-pixel Spectral Angle Mapper, returned as mean over all pixels.
    This is the metric used by the CRISM Stage-4 pipeline.
    """
    H, W, B = clean.shape
    c_flat  = clean.reshape(-1, B).astype(float)
    r_flat  = restored.reshape(-1, B).astype(float)

    dot  = np.nansum(c_flat * r_flat, axis=1)
    norm = (np.sqrt(np.nansum(c_flat**2, axis=1)) *
            np.sqrt(np.nansum(r_flat**2, axis=1)) + 1e-12)
    angles = np.arccos(np.clip(dot / norm, -1 + 1e-6, 1 - 1e-6))
    return float(np.nanmean(angles))


def compute_snr(cube: np.ndarray,
                good_mask: np.ndarray | None = None,
                if_min: float = -0.05,
                if_max: float = 1.30) -> float:
    """
    Mean SNR = |mean(band)| / std(band), averaged over good bands.
    Used by the CRISM Stage-4 pipeline for validation.
    """
    _, _, B = cube.shape
    snrs = []
    for b in range(B):
        if good_mask is not None and not good_mask[b]:
            continue
        v = cube[:, :, b]
        v = v[np.isfinite(v) & (v > if_min) & (v < if_max)]
        if v.size > 1:
            snrs.append(abs(v.mean()) / (v.std() + 1e-8))
    return float(np.mean(snrs)) if snrs else np.nan


def compute_band_depth(spectrum: np.ndarray, waves: np.ndarray,
                        centre_um: float, left_um: float,
                        right_um:  float) -> float:
    """
    Compute mineral band depth from a 1D spectrum.
    BD = 1 - Rc / Rcont
    where Rcont is the linear continuum between left and right shoulders.
    """
    def _r(wl):
        bi = int(np.argmin(np.abs(waves - wl)))
        v  = float(spectrum[bi])
        return np.nan if not np.isfinite(v) else v

    rc, rl, rr = _r(centre_um), _r(left_um), _r(right_um)
    if any(np.isnan(x) for x in (rc, rl, rr)):
        return np.nan
    frac = (centre_um - left_um) / (right_um - left_um + 1e-9)
    cont = rl + frac * (rr - rl)
    return float(1.0 - rc / cont) if cont > 1e-6 else np.nan


def compute_all_metrics(clean:     np.ndarray,
                         restored:  np.ndarray,
                         waves:     np.ndarray | None = None,
                         good_mask: np.ndarray | None = None,
                         label:     str = "") -> dict:
    """
    Compute the full set of metrics and return as a dict.
    Prints a formatted summary.
    """
    data_range = max(_band_range(clean), 1e-6)

    mpsnr = compute_mpsnr(clean, restored, data_range)
    mssim = compute_mssim(clean, restored, data_range)
    msa   = compute_msa  (clean, restored, good_mask)
    sam   = compute_sam  (clean, restored, good_mask)
    snr_b = compute_snr  (clean,           good_mask)
    snr_a = compute_snr  (restored,        good_mask)
    snr_d = (snr_a - snr_b) / (snr_b + 1e-8) * 100.0

    metrics = {
        'mpsnr':    round(mpsnr, 3),
        'mssim':    round(mssim, 4),
        'msa_rad':  round(msa,   4),
        'sam_rad':  round(sam,   4),
        'snr_clean':    round(snr_b, 3),
        'snr_restored': round(snr_a, 3),
        'snr_delta_pct': round(snr_d, 1),
    }

    # Status (mirrors Stage-4 logic)
    if sam < 0.05 and snr_a >= snr_b * 0.99:
        metrics['status'] = 'PASS'
    elif sam < 0.10 and snr_a >= snr_b * 0.99:
        metrics['status'] = 'REVIEW'
    else:
        metrics['status'] = 'FAIL'

    if label:
        print(f"\n  ── Metrics: {label} ──")
        print(f"  MPSNR : {mpsnr:7.3f} dB")
        print(f"  MSSIM : {mssim:7.4f}")
        print(f"  MSA   : {msa:7.4f} rad")
        print(f"  SAM   : {sam:7.4f} rad  → {metrics['status']}")
        print(f"  SNR   : {snr_b:.3f} → {snr_a:.3f}  (Δ={snr_d:+.1f}%)")

    return metrics
