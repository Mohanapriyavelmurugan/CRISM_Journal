#!/usr/bin/env python3
"""
crism_utils.py  —  Shared utilities for the CRISM pipeline
===========================================================
Handles:
  - PDS3 label parsing (object-scoped, not naïve regex)
  - Proper wavelength-axis loading from label BAND_BIN tables
  - Cube loading with file-size validation
  - Physical mask construction (bad-band + I/F bounds)
  - Shared constants

Wavelength note
---------------
CRISM TRR3 L labels often omit BAND_BIN_CENTER (wavelengths live in a
separate WA CDR).  For 438-band L cubes we build a descending axis from
the USGS CRISM-convolved wavelength table (cvCRISMjMTR3) — this is the
CRISM channel µm list, not Earth geology.  Band 0 ≈ long-λ (~3.9 µm),
band 437 ≈ short-λ (~1.0 µm).  Cached per-scene as wavelengths_L438.npy.
"""

import os, re, glob
import numpy as np

# ─────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────
IF_MAX =  1.30   # Lambertian upper bound for Mars
IF_MIN = -0.05   # dark-current noise floor

# USGS CRISM-convolved L grid (µm / nm). Dispersion from
# s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt.
_USGS_L_STEP = 0.0066101          # µm per channel
_USGS_L_SHORT = 1.00364           # short-λ end of L grid
_DEFAULT_USGS_WAVE_FILE = os.path.join(
    r"D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata"
    r"\ASCIIdata_splib07b_cvCRISMjMTR3",
    "s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt",
)


def _parse_usgs_wave_file(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, ln in enumerate(f):
            if i == 0:
                continue
            ln = ln.strip()
            if not ln:
                continue
            try:
                vals.append(float(ln.split()[0]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float64)


def build_crism_L_waves_usgs(n_bands: int = 438, wave_file: str = None) -> np.ndarray:
    """
    Build CRISM L-sensor wavelength axis (µm), descending in band index.

    Uses the USGS CRISM-convolved joined wavelength table for dispersion,
    then places the short-λ end so the saturated ~2.70 µm CO₂ core falls
    on the known invalid mid-detector run (Scene 1 / typical TRR3 L).

    This is a *sensor channel list*, not a mineral library match.
    """
    if n_bands != 438:
        raise ValueError(f"USGS L builder expects 438 bands, got {n_bands}")

    step = _USGS_L_STEP
    short = _USGS_L_SHORT
    path = wave_file or os.environ.get("USGS_CRISM_WAVES", _DEFAULT_USGS_WAVE_FILE)
    if path and os.path.isfile(path):
        try:
            all_w = _parse_usgs_wave_file(path)
            L = all_w[all_w > 1.0]
            if len(L) >= 50:
                d = np.diff(L)
                step = float(np.median(d[d > 0]))
                short = float(L.min())
        except Exception as e:
            print(f"  [WARN] USGS wave file unreadable ({e}); using defaults")

    # Place short-λ at last band; ascending channel index from short→long,
    # then reverse so band 0 = long wavelength (matches TRR3 L byte order).
    # Offset calibrated so mid invalid run (~band 175.5) ≈ 2.75 µm CO₂.
    # w[b] = w_short + step*(n_bands-1-b)
    # Solve: 2.75 = w_short + step*(437-175.5)  →  w_short ≈ 1.022
    co2_band = 175.5
    co2_um = 2.75
    w_short = co2_um - step * ((n_bands - 1) - co2_band)
    # Prefer USGS short if within 30 nm of calibrated value
    if abs(w_short - short) < 0.030:
        w_short = short
    waves = w_short + step * np.arange(n_bands - 1, -1, -1, dtype=np.float64)
    return waves.astype(np.float32)


def assert_crism_L_wavelengths(waves: np.ndarray, cube: np.ndarray = None) -> dict:
    """
    Physics gate: fail loudly if the L axis is ascending/wrong.
    Returns a small diagnostics dict on success.
    """
    waves = np.asarray(waves, dtype=np.float64)
    if waves.size < 100:
        raise ValueError("Wavelength axis too short for CRISM L")
    if waves[0] < waves[-1]:
        raise ValueError(
            f"CRISM L wavelength axis must be DESCENDING in band index "
            f"(got band0={waves[0]:.3f} < bandN={waves[-1]:.3f}). "
            f"Refusing to run with a flipped ruler."
        )
    if not (3.5 < waves[0] < 4.2 and 0.90 < waves[-1] < 1.15):
        raise ValueError(
            f"CRISM L endpoints look wrong: [{waves[0]:.3f}, {waves[-1]:.3f}] µm "
            f"(expected ~3.9 → ~1.0)."
        )
    info = {
        "band0_um": float(waves[0]),
        "bandN_um": float(waves[-1]),
        "descending": True,
    }
    if cube is not None:
        H, W, B = cube.shape
        if B != len(waves):
            raise ValueError(f"Cube bands {B} != waves {len(waves)}")
        # Scene median on interior columns
        c0, c1 = max(0, W // 10), min(W, W - W // 10)
        med = np.nanmedian(cube[:, c0:c1, :].reshape(-1, B), axis=0)
        ok = np.isfinite(med)
        if ok.sum() < 50:
            print("  [WARN] wavelength assert: too few finite median bands")
            return info
        # Deepest absorption should be near 2.70 µm
        bi_deep = int(np.nanargmin(med))
        w_deep = float(waves[bi_deep])
        info["deepest_band"] = bi_deep
        info["deepest_um"] = w_deep
        if not (2.55 <= w_deep <= 2.85):
            raise ValueError(
                f"Deepest median absorption at band {bi_deep} = {w_deep:.3f} µm "
                f"(expected ~2.70 µm CO2). Wavelength axis still wrong."
            )
        # Local min near 2.01 µm CO2
        near = np.where((waves >= 1.95) & (waves <= 2.10) & ok)[0]
        if len(near):
            bi_201 = int(near[np.argmin(med[near])])
            info["co2_201_band"] = bi_201
            info["co2_201_um"] = float(waves[bi_201])
        print(f"  [WAVES OK] descending {waves[0]:.3f}→{waves[-1]:.3f} µm | "
              f"deepest@{w_deep:.3f} µm")
    return info


def resolve_sensor_waves(n_bands: int, label_waves=None, data_dir: str = None,
                         cube: np.ndarray = None) -> np.ndarray:
    """
    Resolve wavelength axis with caching.
    Priority: label → cached scene file → USGS CRISM L builder → S linspace.
    """
    if label_waves is not None and len(label_waves) == n_bands:
        waves = np.array(label_waves, dtype=np.float32)
        if n_bands == 438:
            assert_crism_L_wavelengths(waves, cube=cube)
        return waves

    if data_dir and n_bands == 438:
        cache = os.path.join(data_dir, "wavelengths_L438.npy")
        if os.path.isfile(cache):
            waves = np.load(cache).astype(np.float32)
            if waves.shape == (438,):
                assert_crism_L_wavelengths(waves, cube=cube)
                return waves

    if n_bands == 438:
        waves = build_crism_L_waves_usgs(438)
        assert_crism_L_wavelengths(waves, cube=cube)
        if data_dir:
            try:
                os.makedirs(data_dir, exist_ok=True)
                np.save(os.path.join(data_dir, "wavelengths_L438.npy"), waves)
            except Exception:
                pass
        return waves

    if n_bands == 107:
        return WAVES_S.copy()

    raise ValueError(
        f"No wavelength table for n_bands={n_bands}. "
        f"Supply label BAND_BIN_CENTER or a cached axis."
    )


def get_sensor_waves(n_bands: int, label_waves=None, data_dir: str = None,
                     cube: np.ndarray = None):
    """
    Return the wavelength array for a cube with `n_bands` bands.
    For 438-band L cubes the axis is DESCENDING (long→short).
    """
    return resolve_sensor_waves(n_bands, label_waves=label_waves,
                                data_dir=data_dir, cube=cube)


WAVES_L = build_crism_L_waves_usgs(438)                    # shape (438,)
WAVES_S = np.linspace(0.3620, 1.0530, 107).astype(np.float32)  # shape (107,)

# ─────────────────────────────────────────────────────────────
# BAD-BAND REGIONS  (hard excludes only — keep mineral diagnostics)
# Keep ~1.05 (olivine), ~1.9 (hydration), ~2.14–2.30 (clays).
# Blank true CO2 cores, saturated 2.7 µm, thermal tail, cut-on edge.
# ─────────────────────────────────────────────────────────────
BAD_BAND_REGIONS_L = [
    (0.970, 1.025),   # below / at L cut-on
    (1.370, 1.420),   # telluric H2O (narrow)
    (1.980, 2.070),   # atmospheric CO2 core (keep ~1.93 for BD1900)
    (2.650, 2.800),   # saturated CO2 / deep atm
    (3.500, 4.000),   # thermal emission tail
]

BAD_BAND_REGIONS_S = [
    (0.362, 0.410),   # detector startup
    (0.930, 0.960),   # atmospheric O2
]

# Extra-hard excludes for SPM channel picking (same as bad for L now)
SPM_HARD_EXCLUDE_REGIONS = list(BAD_BAND_REGIONS_L)


def build_bad_mask(waves: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True = bad / unreliable band.
    Automatically selects L vs S region list by wavelength range.
    For L, uses endpoints: descending axis has waves[0] > waves[-1].
    """
    is_L = float(np.nanmax(waves)) > 1.5
    regions = BAD_BAND_REGIONS_L if is_L else BAD_BAND_REGIONS_S
    mask = np.zeros(len(waves), dtype=bool)
    for lo, hi in regions:
        mask |= (waves >= lo) & (waves <= hi)
    return mask


# ─────────────────────────────────────────────────────────────
# PDS3 LABEL PARSER  (object-scoped)
# ─────────────────────────────────────────────────────────────
def parse_pds3_label(label_path: str) -> dict:
    """
    Parse a PDS3 .lbl file.

    Key improvements over naïve regex:
      - Extracts values from within the correct OBJECT block
        (IMAGE vs SPECTRUM_QUBE etc.) to avoid picking up
        duplicate keywords at the wrong scope.
      - Parses BAND_BIN_CENTER value-sequence into a float list,
        giving us the actual wavelength axis.
      - Returns a dict with the following guaranteed keys:
          lines, line_samples, bands, sample_bits, sample_type,
          band_storage, sensor_id, unit, product_id,
          wavelengths (list[float] or None)
    """
    meta = {
        'lines': 0, 'line_samples': 0, 'bands': 0,
        'sample_bits': 32, 'sample_type': 'PC_REAL',
        'band_storage': 'LINE_INTERLEAVED',
        'sensor_id': 'Unknown', 'unit': 'I/F',
        'product_id': 'Unknown', 'wavelengths': None,
    }
    try:
        with open(label_path, 'r', errors='replace') as f:
            text = f.read()
    except OSError as e:
        meta['parse_error'] = str(e)
        return meta

    def _first_int(pattern, src):
        m = re.search(pattern, src, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _first_str(pattern, src):
        m = re.search(pattern, src, re.IGNORECASE)
        return m.group(1).strip('"').strip() if m else None

    # ── Top-level scalar keywords ────────────────────────────
    for key, pat in [
        ('sensor_id',   r'MRO:SENSOR_ID\s*=\s*"?(\S+)"?'),
        ('product_id',  r'PRODUCT_ID\s*=\s*"?([^"\r\n]+)"?'),
        ('unit',        r'\bUNIT\s*=\s*"?([^"\r\n,)]+)"?'),
    ]:
        v = _first_str(pat, text)
        if v:
            meta[key] = v

    # ── IMAGE object block ───────────────────────────────────
    # Scope to the first IMAGE object to avoid mis-picking
    img_match = re.search(
        r'OBJECT\s*=\s*IMAGE(.*?)END_OBJECT\s*=\s*IMAGE',
        text, re.IGNORECASE | re.DOTALL)
    scope = img_match.group(1) if img_match else text

    for key, pat in [
        ('lines',        r'\bLINES\s*=\s*(\d+)'),
        ('line_samples', r'\bLINE_SAMPLES\s*=\s*(\d+)'),
        ('bands',        r'\bBANDS\s*=\s*(\d+)'),
        ('sample_bits',  r'\bSAMPLE_BITS\s*=\s*(\d+)'),
    ]:
        v = _first_int(pat, scope)
        if v is not None:
            meta[key] = v

    st = _first_str(r'\bSAMPLE_TYPE\s*=\s*(\S+)', scope)
    if st:
        meta['sample_type'] = st.strip('"')

    bs = _first_str(r'\bBAND_STORAGE_TYPE\s*=\s*(\S+)', scope)
    if bs:
        meta['band_storage'] = bs.strip('"')

    # ── BAND_BIN object (wavelengths) ───────────────────────
    bb_match = re.search(
        r'OBJECT\s*=\s*BAND_BIN(.*?)END_OBJECT\s*=\s*BAND_BIN',
        text, re.IGNORECASE | re.DOTALL)
    if bb_match:
        bb_text = bb_match.group(1)
        # BAND_BIN_CENTER = ( val, val, ... ) possibly multi-line
        bc = re.search(
            r'BAND_BIN_CENTER\s*=\s*\(([^)]+)\)',
            bb_text, re.IGNORECASE | re.DOTALL)
        if bc:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', bc.group(1))
            if nums:
                meta['wavelengths'] = [float(x) for x in nums]

    return meta


# ─────────────────────────────────────────────────────────────
# CUBE LOADER
# ─────────────────────────────────────────────────────────────
def load_crism_cube(img_path: str, meta: dict) -> np.ndarray:
    """
    Load a CRISM BIL/BSQ/BIP .img file → float32 (H, W, B).

    File-size validation: raises ValueError if the label dims
    disagree with the actual file by more than 1 % and cannot
    be explained by a ROWNUM table suffix.
    """
    file_size = os.path.getsize(img_path)
    L = int(meta.get('lines',        0))
    S = int(meta.get('line_samples', 0))
    B = int(meta.get('bands',        0))
    bits = int(meta.get('sample_bits', 32))
    bps  = bits // 8

    # ── Heuristic dims if label was empty ────────────────────
    if not (L and S and B):
        total = file_size // bps
        for b_try in [438, 107, 545, 248]:
            if total % b_try != 0:
                continue
            rem = total // b_try
            for w_try in [640, 64, 320, 128, 256]:
                h_try = rem // w_try
                if h_try * w_try * b_try == total and h_try > 0:
                    L, S, B = h_try, w_try, b_try
                    print(f"  [HEURISTIC] dims inferred: {L}×{S}×{B}")
                    break
            if L: break

    if not (L and S and B):
        raise ValueError(f"Cannot determine cube dims for {os.path.basename(img_path)}")

    # ── File-size validation ──────────────────────────────────
    expected = L * S * B * bps
    remainder = file_size - expected
    rownum_v1 = L * B * 4   # TRR3 ROWNUM table per BIL row
    rownum_v2 = L * 4       # per spatial line

    if remainder == 0:
        pass
    elif abs(remainder) in (rownum_v1, rownum_v2):
        print(f"  [INFO] ROWNUM table present ({abs(remainder)} bytes), ignored.")
    elif abs(remainder) / file_size < 0.01:
        print(f"  [WARN] size mismatch {remainder:+d} bytes (<1%), proceeding.")
    else:
        raise ValueError(
            f"File-size mismatch for {os.path.basename(img_path)}: "
            f"label says {expected} bytes, file is {file_size} bytes "
            f"(diff {remainder:+d}).  "
            f"Check LINES={L}, LINE_SAMPLES={S}, BANDS={B}.")

    raw = np.fromfile(img_path, dtype='<f4', count=L * S * B)
    if raw.size != L * S * B:
        raise ValueError(f"Read {raw.size} samples, expected {L*S*B}")

    storage = meta.get('band_storage', 'LINE_INTERLEAVED').upper()
    if 'BIL' in storage or 'LINE_INTERLEAVED' in storage:
        cube = raw.reshape(L, B, S).transpose(0, 2, 1)   # → (L,S,B)
    elif 'BSQ' in storage or 'BAND_SEQUENTIAL' in storage:
        cube = raw.reshape(B, L, S).transpose(1, 2, 0)
    elif 'BIP' in storage or 'BAND_INTERLEAVED_BY_PIXEL' in storage:
        cube = raw.reshape(L, S, B)
    else:
        cube = raw.reshape(L, B, S).transpose(0, 2, 1)

    cube = cube.astype(np.float32)
    cube[cube >= 65534.0] = np.nan   # PDS3 sentinel
    return cube


def find_label(img_path: str):
    base = os.path.splitext(img_path)[0]
    for ext in ('.lbl', '.LBL', '.lbl.txt'):
        c = base + ext
        if os.path.exists(c):
            return c
    return None