#!/usr/bin/env python3
"""
Stage5_minerals.py  —  CRISM Mineral Identification Pipeline
=============================================================
Linear pipeline (Stages 0–7) implementing the methodology in
CRISM_MINERAL_IDENTIFICATION_GUIDE.md:

  0  Load denoised cube (_S4.npy), wavelengths, bad/dead masks
  1  USGS CRISM-convolved library -> 438-band subset (NO SRF)
  2  Continuum removal + 12 SPM indices on full scene
  3  Contrastive SpectralEncoder (NT-Xent + SPM anchor)
  4  HDBSCAN clustering + SPM/library naming
  5  Physics-constrained AbundanceNetwork unmixing (softmax)
  6  SLIC superpixel CRF-style spatial smoothing
  7  Validation, rare-mineral detection, publication figures

Usage
-----
  # New full-height FRT scenes (default Scene 1)
  set CRISM_DATA_DIR=D:\\CRISM scenes\\scenes\\Scene 1
  python Stage5_minerals.py

  # Or helper
  python run_mineral_fullfrt.py --scene 1

  # Via orchestrator
  python Run_pipeline.py --stages 5
"""

from __future__ import annotations

import os
# Must be set before numba/hdbscan first import (Windows stability)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gc
import glob
import json
import time
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

from crism_utils import (
    parse_pds3_label,
    find_label,
    get_sensor_waves,
    build_bad_mask,
    load_crism_cube,
    assert_crism_L_wavelengths,
)

# -------------------------------------------------------------
# PATHS / CONSTANTS
# -------------------------------------------------------------
DATA_DIR = os.environ.get(
    "CRISM_DATA_DIR",
    r"D:\CRISM scenes\scenes\Scene 1",
)
USGS_DIR = os.environ.get(
    "USGS_CRISM_LIB",
    r"D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata\ASCIIdata_splib07b_cvCRISMjMTR3",
)
OUTPUT_DIR = os.path.join(DATA_DIR, "mineral_output")

N_BANDS = 438
LATENT_DIM = 128          # richer fingerprint — reduce mineral-feature loss in clustering
N_SPM = 12
BLOCK_ROWS = 32
BATCH_INF = 1024          # safer for 4 GB GPUs + W=640 cubes
ENCODER_BATCH = 128
ENCODER_EPOCHS = 80
UNMIX_EPOCHS_A = 40
UNMIX_EPOCHS_B = 8
ENCODER_PATIENCE = 12
UNMIX_PATIENCE = 10
# Full mode: do not early-stop before this many epochs (avoids under-separated latents)
ENCODER_MIN_EPOCHS_FULL = 30
ENCODER_MIN_EPOCHS_ADAPT = 8
# ---------------------------------------------------------------------------
# v6 FREEZE — do not retune without a new Scene1 regression baseline:
#   CLUSTER_SPM/DIAG weights, noise-rescue ×1.85, leaf keep bars, PROOF_DEPTH_*,
#   SAM_MARGIN_STRONG, MEGA_CLUSTER_FRAC (primary). Safe adds only below
#   (CLUSTER_DEPTH_* fusion is additive; does not retune SPM/DIAG).
# ---------------------------------------------------------------------------
# Mega-cluster fraction of SPM-active domain → last-resort physics split only
MEGA_CLUSTER_FRAC = 0.40
# Competing SPM pairs for mega-unit splitting (hydration vs sulfate continuum, etc.)
SPM_COMPETE_PAIRS = (
    ("BD1900_2", "SINDEX2"),
    ("BD1900_2", "BD2100_2"),
    ("BD2210_2", "BD2290"),
    ("BD2210_2", "BD2300"),
    ("BD2290", "BD2300"),
)
# Secondary clay-only split (post-HDBSCAN): always attempted after primary.
# Relative-strength mode allows Al-OH vs Fe/Mg split when hotspots overlap
# (exclusive-only gate was blocking Scene2 hydrate mega-units).
MEGA_SECONDARY_FRAC = 0.12
MEGA_SECONDARY_PAIRS = (
    ("BD2210_2", "BD2290"),
    ("BD2210_2", "BD2300"),
    ("BD2290", "BD2300"),
)
# Min hotspot fraction inside an Unresolved hydrate unit to show clay display_label
CLAY_DISPLAY_HOTSPOT_FRAC = 0.15
CLAY_DISPLAY_MARGIN = 0.05  # need clear Al-OH vs Fe/Mg winner for display
# Prefer PRESENT Fe/Mg-OH (BD2290) / Al-OH; exclude BD2300 (often collapsed, not PRESENT)
CLAY_DISPLAY_SPM = ("BD2210_2", "BD2290")
# How many recursive mega-split rounds (BD1900↔SINDEX, then clay pairs inside units)
MEGA_SPLIT_MAX_ROUNDS = 2
# Soft priority when picking SPM group for a cluster (hydration before SINDEX proxy)
SPM_LABEL_PRIORITY = {
    "BD1900_2": 1.20,
    "BD2210_2": 1.10,
    "BD2290": 1.10,
    "BD2300": 1.10,
    "BD2500_2": 1.05,
    "OLINDEX3": 1.10,
    "LCPINDEX2": 1.05,
    "HCPINDEX2": 1.05,
    "SINDEX2": 0.80,   # continuum/sulfate proxy — downweight vs BD1900
    "BD2100_2": 1.00,
}
# Transfer mode (overridden by run_mineral_identification / env)
TRAIN_MODE = "full"  # full | adapt
SHARED_PRIOR_DIR = None  # set at runtime for adapt
TARGET_TRAINING_SAMPLES = 12000
SYNTH_MIXES = 4000
LIB_SAMPLE_N = 2000
RARE_ANGLE_THRESH = 0.25
MINOR_ABUND_THRESH = 0.30   # secondary rare only when clearly present
MIN_CLUSTER_KEEP = 30       # minority clusters preserved in smoothing
FULL_HEIGHT_MIN_LINES = 100
MIN_ENDMEMBER_ANGLE = 0.10  # rad; keep Mars-prior FPS endmembers distinct
NOISE_UNC_FLOOR = 0.70      # soft floor for HDBSCAN noise (was hard 0.85)
# Fit-set size for HDBSCAN (smaller → avoids 20k×20k float64 OOM on Windows)
HDBSCAN_MAX_FIT = int(os.environ.get("CRISM_HDBSCAN_MAX_FIT", "10000"))
DBCV_MAX_SAMPLES = int(os.environ.get("CRISM_DBCV_MAX_SAMPLES", "4000"))
# Discovery-first clustering fusion: encoder latent is backbone; SPM is soft only
# FROZEN at v6: starving SPM to 0.15 collapsed Scene1 to 1 mixed cluster
CLUSTER_SPM_WEIGHT = 0.30
CLUSTER_DIAG_WEIGHT = 0.30
CLUSTER_XY_WEIGHT = 0.08
# Continuum band depths (proof defs) fused into HDBSCAN — ice keys excluded.
# Does not retune CLUSTER_SPM/DIAG freeze. Adapt uses a milder weight so
# transfer scenes (e.g. Scene2) are not dominated by noisy depth axes.
CLUSTER_DEPTH_WEIGHT = 0.30
CLUSTER_DEPTH_WEIGHT_ADAPT = 0.15
CLUSTER_DEPTH_MIN_STD = 0.002  # mute near-flat depth channels after compute
# Adapt-only: if domain noise stays high with depth, trial without depth
ADAPT_DEPTH_DOMAIN_NOISE_RETRY = 0.55
ADAPT_DEPTH_NOISE_IMPROVE = 0.05
CLUSTER_DEPTH_KEYS = (
    "d1.05",   # olivine
    "d1.75",   # gypsum OH
    "d1.93",   # hydrated H2O
    "d2.10",   # mono-sulfate
    "d2.12",   # pyroxene
    "d2.21",   # Al-OH
    "d2.26",   # jarosite
    "d2.29",   # Fe/Mg-OH
    "d2.30",   # Fe/Mg phyllosilicate / carbonate shoulder
    "d2.40",   # polyhydrate sulfate
    "d2.50",   # carbonate
)
# Explicitly never cluster on ice depths (proof-only)
CLUSTER_DEPTH_EXCLUDE = frozenset({"d1.43", "d1.50"})
# Inter-cluster mean SAM gap (rad) that can lift clustering_quality to MODERATE
CLUSTER_INTER_SAM_GAP_MODERATE = 0.020
# Continuum-trap families blocked from open-set USGS naming
OPEN_USGS_BLOCKLIST = frozenset({
    "quartz", "hematite", "goethite", "maghemite", "ferrihydrite",
})
# Absolute CR depth floors for PRESENT claims (noise-aware) — FROZEN at v6
PROOF_DEPTH_OLIVINE = 0.010
PROOF_DEPTH_PYROXENE = 0.008
PROOF_DEPTH_BD = 0.008
PROOF_DEPTH_BD_MIN = 0.006  # clay/adaptive floor never below this
PROOF_DEPTH_BD_STRONG = 0.015  # both Fe/Mg indices need this to stay dual-PRESENT
PROOF_HOT_MINUS_BLAND = 0.005
# Clay BD indices use scene-adaptive floor (see adaptive_bd_depth_floor)
CLAY_BD_SPM = frozenset({"BD2210_2", "BD2290", "BD2300"})
# Sulfate CR diagnostics for SINDEX2 PRESENT (not d2.10 alone)
SULFATE_PROOF_DEPTH_KEYS = ("d2.40", "d1.75", "d2.26", "d2.10")
SULFATE_PURE_FAMILIES = frozenset({
    "gypsum", "jarosite", "alunite", "epsomite", "kieserite",
    "szomolnokite", "bassanite", "copiapite",
})
# Primary endmembers above this mean abundance are not "rare" secondaries
RARE_PRIMARY_MEAN_ABUND_EXCLUDE = 0.15
# Continuum-removed reflectance must stay near [0, 2]; CO₂ / fill glitches
# previously leaked into unmixing targets and blew MSE to ~1e13.
CR_SANITIZE_LO = 0.0
CR_SANITIZE_HI = 2.0
# Explicit SPM → diagnostic depth key (avoids BD2290/BD2500 collapsing to d2.30)
# SINDEX2 is a continuum-convexity proxy — proof uses multi-band sulfate keys below.
SPM_PROOF_DEPTH_KEY = {
    "BD1900_2": "d1.93",
    "BD2100_2": "d2.10",
    "BD2210_2": "d2.21",
    "BD2290": "d2.29",
    "BD2300": "d2.30",
    "BD2500_2": "d2.50",
    "OLINDEX3": "d1.05",
    "LCPINDEX2": "d2.12",
    "HCPINDEX2": "d2.12",
    "SINDEX2": "d2.40",  # primary sulfate diagnostic; fallbacks in SULFATE_PROOF_DEPTH_KEYS
    "BD1500_2": "d1.50",  # H2O ice (polar proof only)
    "BD1435": "d1.43",    # CO2 ice (polar proof only)
}
# Primary + fallback (center, left, right) tuples — S4 often has dead 2.067 µm
PROOF_DEPTH_DEFS = {
    "d1.05": [(1.054, 1.030, 1.250)],
    "d1.33": [(1.330, 1.054, 1.750)],
    "d1.43": [(1.435, 1.370, 1.470)],  # CO2 ice
    "d1.50": [(1.510, 1.367, 1.810), (1.500, 1.360, 1.800)],  # H2O ice
    "d1.75": [(1.750, 1.700, 1.850), (1.750, 1.650, 1.900)],  # gypsum OH
    "d1.93": [(1.930, 1.857, 2.067), (1.930, 1.857, 2.100), (1.930, 1.860, 2.120)],
    "d2.10": [(2.100, 1.930, 2.250), (2.132, 1.930, 2.250)],
    "d2.12": [(2.120, 1.810, 2.530)],
    "d2.21": [(2.210, 2.165, 2.250)],
    "d2.26": [(2.265, 2.210, 2.300), (2.260, 2.210, 2.300)],  # jarosite
    "d2.29": [(2.290, 2.250, 2.350)],
    "d2.30": [(2.300, 2.120, 2.530), (2.295, 2.120, 2.530)],
    "d2.40": [(2.400, 2.300, 2.500), (2.430, 2.300, 2.530)],  # polyhydrated sulfate
    "d2.50": [(2.500, 2.380, 2.570)],
}
# Preferred USGS family stems to seed endmembers per SPM mineral group
SPM_GROUP_SEED_FAMILIES = {
    "Hydrated Minerals": (
        "montmorillonite", "nontronite", "kaolinite", "saponite",
        "gypsum", "epsomite", "illite", "chlorite",
    ),
    "Monohydrated Sulfates": ("kieserite", "szomolnokite", "epsomite"),
    "Al-OH Clays": ("montmorillonite", "kaolinite", "illite", "beidellite"),
    "Fe/Mg-OH Phyllosilicates": ("nontronite", "saponite", "chlorite", "serpentine", "clinochlore"),
    "Fe/Mg Phyllosilicates": ("nontronite", "saponite", "chlorite", "clinochlore"),
    "Carbonates": ("calcite", "dolomite", "magnesite", "siderite"),
    "Olivine": ("olivine", "forsterite", "fayalite"),
    "Low-Ca Pyroxene": ("enstatite", "hypersthene", "bronzite", "pigeonite"),
    "High-Ca Pyroxene": ("diopside", "augite"),
    "Sulfates": ("gypsum", "alunite", "jarosite", "epsomite"),
}
# Max spectral angle (rad) for a family-restricted library hit to count as corroboration
FAMILY_SAM_HIT_MAX = 0.25
# Mafic indices: if TRR3-img SPM is artifact-dead, retry on ratioed / S4 I/F
MAFIC_SPM_NAMES = {"OLINDEX3", "LCPINDEX2", "HCPINDEX2"}
# Hydration indices gated when scene-flat (atm residual risk on TRR3 I/F)
FLAT_HYDRATION_SPM = {"BD1900_2", "BD2100_2"}
# Classic CRISM bland-region ratio (Mustard / Pelkey practice): suppress atm +
# instrument continuum before SPM / CR / library SAM. Methodology unchanged —
# Viviano-Beck indices still run on reflectance-like I/F (ratioed).
BLAND_PCT = 20
# Exclude ONLY confirmed atmospheric cores from library SAM (true µm).
SAM_EXCLUDE_WINDOWS = [
    (1.980, 2.070),  # CO2 core
    (2.650, 2.800),  # saturated CO2
]
# Mineral CR / encoder / SAM: keep Viviano diagnostic bands.
# Hard excludes only (aligned with crism_utils.BAD_BAND_REGIONS_L).
MINERAL_CR_EXCLUDE = [
    (0.970, 1.025),   # L cut-on edge
    (2.650, 2.800),   # saturated CO2
    (3.500, 4.000),   # thermal tail
]
# Stripe-like HDBSCAN units (pushbroom columns) → treat as noise
STRIPE_X_STD_MAX = 18.0
STRIPE_Y_RANGE_MIN = 200
# Detector edge columns (FRT left/right margins) — mask from all stages
EDGE_COL_MARGIN = 20
# Softmax unmixing: keep library endmembers nearly fixed (ASC abundances learn)
UNMIX_LIB_WEIGHT = 1.0
UNMIX_FREEZE_E_PHASE_A = True
# Library SAM: suspiciously tiny angles on bland FRT → prefer SPM group
SAM_SUSPICIOUS_TIGHT = 0.05
SAM_STRONG_MAX = 0.12
SAM_MARGIN_STRONG = 0.015  # best must beat 2nd-best by this (rad)
# Clustering: ice SPM channels must NOT steer HDBSCAN (proof-only on polar).
# Weight 0 = present in array but zeroed in fusion (universal, all scenes).
SPM_CLUSTER_WEIGHT = {
    "BD1500_2": 0.0,
    "BD1435": 0.0,
}
# Ice PRESENT sparsity (universal): hotspot / max-component vs valid pixels
# cannot exceed max(floor, 2× median non-ice gate n_kept/valid).
ICE_PROOF_HOT_FRAC_FLOOR = 0.15
ICE_PROOF_HOT_FRAC_MULT = 2.0
# Training / rescue: prefer SPM-active pixels over bland background
SPM_ACTIVE_FLOOR = 0.02
HDBSCAN_NOISE_RETRY = 0.75  # if noise fraction above this, retry smaller mcs
# Fix B: scene-adaptive SPM detection — absolute threshold is primary;
# percentile only ranks within hotspots (cannot alone declare presence).
SPM_ADAPTIVE_PERCENTILE = 90
SPM_MIN_COMPONENT = 50  # connected pixels required for a positive SPM unit
SPM_GATE_USE_PERCENTILE_FLOOR = False  # True = old max(abs,p90) tautology

# Hard-exclude only for SPM channel selection (true atmospheric cores).
SPM_HARD_EXCLUDE = [
    (0.970, 1.025),
    (1.980, 2.070),
    (2.650, 2.800),
    (3.500, 4.000),
]

# Mars-prior mineral names (USGS filename stem match, case-insensitive)
MARS_PRIOR_MINERALS = [
    "Olivine", "Forsterite", "Fayalite",
    "Enstatite", "Bronzite", "Hypersthene", "Diopside", "Augite", "Pigeonite", "Pyroxene",
    "Nontronite", "Saponite", "Montmorillonite", "Kaolinite", "Illite", "Beidellite",
    "Halloysite", "Hectorite", "Vermiculite", "Talc",
    "Chlorite", "Clinochlore", "Chamosite", "Serpentine", "Antigorite", "Lizardite", "Chrysotile",
    "Gypsum", "Bassanite", "Kieserite", "Jarosite", "Alunite", "Anhydrite", "Epsomite",
    "Calcite", "Dolomite", "Magnesite", "Siderite",
    "Hematite", "Goethite", "Maghemite", "Ferrihydrite",
    "Opal", "Quartz", "Chalcedony", "Chert",
    "Prehnite", "Epidote", "Analcime", "Muscovite", "Biotite",
]

MARS_RARE_MINERALS = [
    "Prehnite", "Serpentine", "Antigorite", "Lizardite", "Chrysotile",
    "Opal", "Calcite", "Dolomite", "Magnesite", "Siderite",
    "Analcime", "Epidote", "Jarosite", "Alunite", "Kieserite",
]

# Ice indices gated unless spatially sparse/hotspot-like
ICE_SPM_NAMES = {"BD1500_2", "BD1435"}
# Never use ice indices to build the SPM-active clustering domain
SPM_ACTIVE_EXCLUDE = set(ICE_SPM_NAMES)
# Polar / explicit ice PRESENT screening (clustering domain stays ice-free)
# Enable via CRISM_ALLOW_ICE_PROOF=1 or Scene 4 data path (full mode only).
POLAR_SCENE_IDS = frozenset({4})


def _non_ice_spm_indices() -> List[int]:
    """Channel indices for mineral SPM (excludes ice — clustering domain hygiene)."""
    return [i for i, n in enumerate(SPM_NAMES) if n not in ICE_SPM_NAMES]


def nanmax_spm_exclude_ice(
    spm: np.ndarray,
    axis: int = -1,
    fill: float = -np.inf,
) -> np.ndarray:
    """
    Max over non-ice SPM channels only.

    Used for SPM-active fallbacks, domain rebuild, rescue ranking, and related
    proxies so ice cannot define the clustering domain on any scene.
    """
    idx = _non_ice_spm_indices()
    if not idx:
        return np.full(spm.shape[:axis] + spm.shape[axis + 1 :], np.nan)
    if axis < 0:
        axis = spm.ndim + axis
    # Take only non-ice channels along the SPM axis
    if axis == spm.ndim - 1:
        sub = spm[..., idx]
        return np.nanmax(np.nan_to_num(sub, nan=fill), axis=-1)
    if axis == 1 and spm.ndim == 2:
        # (N_pix, N_SPM)
        sub = spm[:, idx]
        return np.nanmax(np.nan_to_num(sub, nan=fill), axis=1)
    # Generic: swap axis to last
    spm_t = np.moveaxis(spm, axis, -1)
    sub = spm_t[..., idx]
    out = np.nanmax(np.nan_to_num(sub, nan=fill), axis=-1)
    return out

SPM_INDICES = {
    "BD1900_2": (1.930, 1.857, 2.067),
    "BD2100_2": (2.132, 1.930, 2.250),
    "BD2210_2": (2.210, 2.165, 2.250),
    "BD2290":   (2.290, 2.250, 2.350),
    "BD2300":   (2.295, 2.120, 2.530),
    "BD2500_2": (2.500, 2.380, 2.570),
    "OLINDEX3": (1.330, 1.054, 1.750),
    "LCPINDEX2": (1.810, 1.560, 2.450),
    "HCPINDEX2": (2.120, 1.810, 2.530),
    "SINDEX2":  (2.100, 1.815, 2.200),
    "BD1500_2": (1.510, 1.367, 1.810),
    "BD1435":   (1.435, 1.370, 1.470),
}
SPM_NAMES = list(SPM_INDICES.keys())
assert len(SPM_NAMES) == N_SPM

SPM_MINERAL_MAP = {
    "BD1900_2": "Hydrated Minerals",
    "BD2100_2": "Monohydrated Sulfates",
    "BD2210_2": "Al-OH Clays",
    "BD2290": "Fe/Mg-OH Phyllosilicates",
    "BD2300": "Fe/Mg Phyllosilicates",
    "BD2500_2": "Carbonates",
    "OLINDEX3": "Olivine",
    "LCPINDEX2": "Low-Ca Pyroxene",
    "HCPINDEX2": "High-Ca Pyroxene",
    "SINDEX2": "Sulfates",
    "BD1500_2": "Water Ice",
    "BD1435": "CO2 Ice",
}

# Viviano-Beck style positive-detection thresholds (guide)
SPM_DETECT_THRESH = {
    "BD1900_2": 0.01,
    "BD2100_2": 0.01,
    "BD2210_2": 0.005,
    "BD2290": 0.005,
    "BD2300": 0.01,
    "BD2500_2": 0.005,
    "OLINDEX3": 0.02,
    "LCPINDEX2": 0.01,
    "HCPINDEX2": 0.01,
    "SINDEX2": 0.005,
    "BD1500_2": 0.01,
    "BD1435": 0.01,
}

SPM_DIAGNOSTIC_WLS = [
    1.930, 1.857, 2.067, 2.132, 2.250, 2.210, 2.165, 2.290,
    2.350, 2.295, 2.120, 2.530, 2.500, 2.380, 2.570,
    1.330, 1.054, 1.750, 1.810, 1.560, 2.450,
    1.510, 1.367, 1.435, 1.370, 1.470, 2.100, 1.815, 2.200,
]

# Dark theme consistent with existing pipeline figures
_DARK_RC = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json_default(obj):
    """JSON serializer for numpy scalars; NaN/Inf → None."""
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _json_clean(obj):
    """Recursively replace NaN/Inf with None for strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def safe_np_save(path: str, arr: np.ndarray) -> None:
    """
    Windows-safe np.save.

    Direct open(..., 'wb') on a large existing .npy often raises
    OSError Errno 22 (Invalid argument) when the file is briefly locked,
    OneDrive-synced, or left mapped by a prior process. Write to a sibling
    temp file, remove the target if present, then rename into place.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}.npy"
    try:
        if not getattr(arr, "flags", None) or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        # Prefer delete+rename over in-place truncate (Windows-safe).
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                try:
                    os.replace(tmp, path)
                    return
                except OSError:
                    alt = path + ".new.npy"
                    os.replace(tmp, alt)
                    print(f"  [WARN] Could not overwrite {path}; wrote {alt}")
                    return
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------
# SPECTRAL UTILITIES
# -------------------------------------------------------------
def parse_usgs_ascii_values(filepath: str) -> np.ndarray:
    """Parse USGS wavelength/bandpass ASCII (skip header, 489 floats)."""
    values = []
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        val = float(line)
        if val < -1e30:
            values.append(np.nan)
        else:
            values.append(val)
    return np.asarray(values, dtype=np.float32)


def parse_usgs_spectrum(filepath: str) -> Tuple[str, np.ndarray]:
    """Parse USGS ASCII spectrum file. Returns (header, values_489)."""
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()
    header = lines[0].strip()
    values = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        val = float(line)
        if val < -1e30:
            values.append(np.nan)
        else:
            values.append(val)
    return header, np.asarray(values, dtype=np.float32)


def extract_mineral_name(filepath: str) -> str:
    """Parse mineral name from s07CRSMj_{MineralName}_{rest}.txt."""
    base = os.path.basename(filepath)
    name = base.replace("s07CRSMj_", "").replace(".txt", "")
    return name.split("_")[0]


def fill_nans_interp(spectrum: np.ndarray, waves: np.ndarray) -> Optional[np.ndarray]:
    """Fill NaN values using linear interpolation from valid neighbors."""
    finite = np.isfinite(spectrum)
    if finite.sum() < 10:
        return None
    if finite.all():
        return spectrum.astype(np.float32, copy=True)
    f = interp1d(
        waves[finite], spectrum[finite],
        kind="linear", fill_value="extrapolate", bounds_error=False,
    )
    filled = spectrum.astype(np.float32, copy=True)
    filled[~finite] = f(waves[~finite]).astype(np.float32)
    return filled


def continuum_removal(spectrum: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """
    Modified upper convex hull continuum removal.
    Traces the upper envelope by steepest-slope walk from left to right.
    """
    finite = np.isfinite(spectrum) & (spectrum > 0)
    if finite.sum() < 5:
        return np.full_like(spectrum, np.nan, dtype=np.float32)

    w_valid = waves[finite]
    s_valid = spectrum[finite]
    n = len(w_valid)

    # Modified upper hull: left -> right steepest upward slope walk
    hull_idx = [0]
    while hull_idx[-1] < n - 1:
        i = hull_idx[-1]
        best_j = i + 1
        best_slope = -np.inf
        for j in range(i + 1, n):
            dw = w_valid[j] - w_valid[i]
            if dw <= 0:
                continue
            slope = (s_valid[j] - s_valid[i]) / dw
            if slope > best_slope:
                best_slope = slope
                best_j = j
        hull_idx.append(best_j)

    hull_x = w_valid[hull_idx]
    hull_y = s_valid[hull_idx]
    continuum = np.interp(waves, hull_x, hull_y, left=hull_y[0], right=hull_y[-1])

    result = np.full_like(spectrum, np.nan, dtype=np.float32)
    valid = (continuum > 1e-8) & np.isfinite(spectrum)
    result[valid] = (spectrum[valid] / continuum[valid]).astype(np.float32)
    return np.clip(result, 0.0, 2.0)


# Matching-consistent CR window (scene + library must use the same algorithm)
CR_MATCH_FILTER_SIZE = 41


def continuum_removal_match(
    spectrum: np.ndarray,
    waves: np.ndarray,
    size: int = CR_MATCH_FILTER_SIZE,
) -> np.ndarray:
    """
    Continuum removal used for encoder / library SAM.

    Same algorithm for scene and library (moving upper envelope) so spectral
    angles are comparable. Hull remains available via continuum_removal().
    """
    from scipy.ndimage import maximum_filter1d
    finite = np.isfinite(spectrum) & (spectrum > 0)
    if finite.sum() < 5:
        return np.full_like(spectrum, np.nan, dtype=np.float32)
    filled = spectrum.astype(np.float32, copy=True)
    if not finite.all():
        # linear fill for envelope stability
        idx = np.arange(len(spectrum))
        filled[~finite] = np.interp(idx[~finite], idx[finite], spectrum[finite]).astype(np.float32)
    env = maximum_filter1d(filled, size=size, mode="nearest")
    env = np.maximum(env, filled)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = filled / np.maximum(env, 1e-8)
    cr = np.clip(cr, 0.0, 2.0).astype(np.float32)
    cr[~np.isfinite(spectrum)] = np.nan
    return cr


def continuum_removal_match_cube(
    cube: np.ndarray,
    valid: np.ndarray,
    mineral_bad: np.ndarray,
    size: int = CR_MATCH_FILTER_SIZE,
    block_rows: int = BLOCK_ROWS,
) -> np.ndarray:
    """Vectorized matching-consistent CR for a full cube."""
    from scipy.ndimage import maximum_filter1d
    H, W, B = cube.shape
    out = np.full_like(cube, np.nan, dtype=np.float32)
    bad = mineral_bad.astype(bool)
    for r_start in range(0, H, block_rows):
        r_end = min(r_start + block_rows, H)
        block = cube[r_start:r_end].astype(np.float32).copy()
        bh, bw, _ = block.shape
        flat = block.reshape(-1, B)
        vflat = valid[r_start:r_end].reshape(-1)
        for bi in range(B):
            if bad[bi]:
                flat[:, bi] = np.nan
                continue
            col = flat[:, bi]
            m = np.isfinite(col) & vflat
            if m.any():
                fillv = float(np.nanmedian(col[m]))
                miss = (~np.isfinite(col)) & vflat
                col[miss] = fillv
                flat[:, bi] = col
        # Bad bands → 0 for envelope only (excluded from CR output below)
        flat_env = np.nan_to_num(flat, nan=0.0)
        flat_env[:, bad] = 0.0
        env = maximum_filter1d(flat_env, size=size, axis=1, mode="nearest")
        env = np.maximum(env, flat_env)
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = flat / np.maximum(env, 1e-8)
        cr = np.clip(cr, CR_SANITIZE_LO, CR_SANITIZE_HI).astype(np.float32)
        cr[:, bad] = np.nan
        cr[~vflat] = np.nan
        out[r_start:r_end] = cr.reshape(bh, bw, B)
        if (r_start // block_rows) % 5 == 0:
            print(f"  CR progress: {r_end}/{H} rows ({100.0 * r_end / H:.1f}%)")
    return out


def sanitize_cr_array(
    cr: np.ndarray,
    lo: float = CR_SANITIZE_LO,
    hi: float = CR_SANITIZE_HI,
    mineral_bad: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Force continuum-removed spectra into a physical [lo, hi] range.

    Replaces non-finite and out-of-range values with NaN (or 0 for 2D training
    pools via caller). Returns (sanitized, stats).
    """
    out = np.asarray(cr, dtype=np.float32).copy()
    finite = np.isfinite(out)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return out, {"n_finite": 0, "n_clipped": 0, "p99_abs": None, "max_abs_before": None}
    abs_before = float(np.max(np.abs(out[finite])))
    p99_before = float(np.percentile(np.abs(out[finite]), 99))
    bad = (~finite) | (out < lo) | (out > hi)
    n_clipped = int(((finite) & ((out < lo) | (out > hi))).sum())
    out[bad] = np.nan
    if mineral_bad is not None:
        mb = mineral_bad.astype(bool)
        if out.ndim == 3 and mb.shape[0] == out.shape[2]:
            out[:, :, mb] = np.nan
        elif out.ndim == 2 and mb.shape[0] == out.shape[1]:
            out[:, mb] = np.nan
        elif out.ndim == 1 and mb.shape[0] == out.shape[0]:
            out[mb] = np.nan
    fin2 = np.isfinite(out)
    p99_after = float(np.percentile(np.abs(out[fin2]), 99)) if fin2.any() else None
    stats = {
        "n_finite": int(fin2.sum()),
        "n_clipped": n_clipped,
        "max_abs_before": abs_before,
        "p99_abs_before": p99_before,
        "p99_abs_after": p99_after,
        "lo": lo,
        "hi": hi,
    }
    return out, stats


def build_diagnostic_sam_mask(
    waves: np.ndarray,
    good_mask: np.ndarray,
    half_width: float = 0.06,
) -> np.ndarray:
    """
    SAM mask restricted to diagnostic mineral windows (plus small margins).

    Full-spectrum SAM collapses to hematite/quartz continuum shape; windowed
    SAM emphasizes Viviano diagnostic absorptions for proof corroboration.
    """
    base = build_sam_band_mask(waves, good_mask)
    diag = np.zeros(len(waves), dtype=bool)
    centers = set(SPM_DIAGNOSTIC_WLS)
    for _key, triples in PROOF_DEPTH_DEFS.items():
        for c, l, r in triples:
            centers.update((c, l, r))
    for w0 in centers:
        diag |= (waves >= (w0 - half_width)) & (waves <= (w0 + half_width))
    mask = base & diag
    # Fallback if too few bands survive
    if int(mask.sum()) < 20:
        return base
    return mask


def l2_normalize(spectrum: np.ndarray) -> np.ndarray:
    """L2-normalize a spectrum (NaNs -> 0 for the norm)."""
    finite = np.isfinite(spectrum)
    if not finite.any():
        return spectrum.astype(np.float32, copy=True)
    s = np.where(finite, spectrum, 0.0).astype(np.float32)
    norm = float(np.linalg.norm(s))
    if norm < 1e-8:
        return s
    return s / norm


def build_spm_bad_mask(waves: np.ndarray) -> np.ndarray:
    """
    Relaxed bad-mask for Viviano-Beck SPM computation.
    Allows caution-zone diagnostic bands (e.g. ~1.93, ~2.21 µm) that the
    full pipeline bad-mask excludes for continuum/encoding.
    """
    mask = np.zeros(len(waves), dtype=bool)
    for lo, hi in SPM_HARD_EXCLUDE:
        mask |= (waves >= lo) & (waves <= hi)
    return mask


def build_mineral_bad_mask(waves: np.ndarray) -> np.ndarray:
    """
    Bad-mask for mineral CR / encoder / library SAM (Fix A).

    Keeps Viviano-Beck diagnostic bands that build_bad_mask() blanks
    (≈1.05, 1.81–1.95, 2.14–2.26 µm). Still excludes detector startup,
    the 1.55 µm artifact notch, deep CO2/H2O, and thermal tail.
    """
    mask = np.zeros(len(waves), dtype=bool)
    for lo, hi in MINERAL_CR_EXCLUDE:
        mask |= (waves >= lo) & (waves <= hi)
    return mask


def suppress_cr_artifact_notch(
    cube_cr: np.ndarray,
    waves: np.ndarray,
    valid: np.ndarray,
    window: Tuple[float, float] = (1.450, 1.650),
) -> np.ndarray:
    """
    Fix E: linear-interpolate continuum-removed spectra across the known
    ~1.55 µm artifact notch so it cannot dominate encoder / SAM.
    Vectorized over pixels for full-FRT speed.
    """
    out = cube_cr.astype(np.float32, copy=True)
    lo, hi = window
    notch_idx = np.where((waves >= lo) & (waves <= hi))[0]
    if len(notch_idx) == 0:
        return out
    keep_idx = np.where(~((waves >= lo) & (waves <= hi)))[0]
    if len(keep_idx) < 10:
        return out
    # Nearest keep bands on each side of the notch
    left = int(keep_idx[keep_idx < notch_idx[0]][-1]) if np.any(keep_idx < notch_idx[0]) else int(keep_idx[0])
    right = int(keep_idx[keep_idx > notch_idx[-1]][0]) if np.any(keep_idx > notch_idx[-1]) else int(keep_idx[-1])
    w_left = float(waves[left])
    w_right = float(waves[right])
    denom = max(w_right - w_left, 1e-9)
    left_band = out[:, :, left]
    right_band = out[:, :, right]
    for bi in notch_idx:
        w = float(waves[bi])
        t = (w - w_left) / denom
        interp = (1.0 - t) * left_band + t * right_band
        # Only overwrite where both shoulders finite and pixel valid
        ok = valid & np.isfinite(left_band) & np.isfinite(right_band)
        band = out[:, :, bi]
        band = np.where(ok, interp, band)
        out[:, :, bi] = band.astype(np.float32)
    # Keep physical CR range after interpolation
    finite = np.isfinite(out)
    out = np.where(finite, np.clip(out, CR_SANITIZE_LO, CR_SANITIZE_HI), out).astype(np.float32)
    return out


def build_adaptive_spm_active(
    spm_map: np.ndarray,
    valid: np.ndarray,
    spm_notes: Optional[dict] = None,
    percentile: float = SPM_ADAPTIVE_PERCENTILE,
    min_component: int = SPM_MIN_COMPONENT,
) -> Tuple[np.ndarray, dict, List[str]]:
    """
    SPM-active domain for clustering.

    A pixel is SPM-active if ANY non-ice index passes
    value >= Viviano absolute threshold (primary) AND lies in a connected
    component of size >= min_component.

    Percentile is recorded for ranking only — it does NOT raise the gate
    (avoids the tautology that top-10% always "passes").
    """
    from scipy import ndimage as ndi

    H, W, n_spm = spm_map.shape
    active = np.zeros((H, W), dtype=bool)
    info: dict = {}
    passed: List[str] = []
    notes = spm_notes or {}

    for j, name in enumerate(SPM_NAMES):
        if name in SPM_ACTIVE_EXCLUDE or name in ICE_SPM_NAMES:
            info[name] = {"passed": False, "reason": "ice_or_excluded"}
            continue
        if notes and not str(notes.get(name, "ok")).startswith("ok"):
            info[name] = {"passed": False, "reason": f"spm_note={notes.get(name)}"}
            continue
        col = spm_map[:, :, j]
        fin = valid & np.isfinite(col)
        if fin.sum() < 100:
            info[name] = {"passed": False, "reason": "insufficient_finite"}
            continue
        vals = col[fin]
        p_adapt = float(np.percentile(vals, percentile))
        thr_abs = float(SPM_DETECT_THRESH[name])
        # Primary: absolute Viviano-style threshold. Optional legacy floor.
        if SPM_GATE_USE_PERCENTILE_FLOOR:
            thr = max(thr_abs, p_adapt)
        else:
            thr = thr_abs
        raw = fin & (col >= thr)
        # Drop tiny components (noise salt)
        labeled, ncomp = ndi.label(raw)
        keep = np.zeros_like(raw)
        n_kept = 0
        max_sz = 0
        if ncomp > 0:
            sizes = np.bincount(labeled.ravel())
            for lab_id in range(1, ncomp + 1):
                sz = int(sizes[lab_id]) if lab_id < len(sizes) else 0
                max_sz = max(max_sz, sz)
                if sz >= min_component:
                    keep |= labeled == lab_id
                    n_kept += 1
        n_pix = int(keep.sum())
        # Also require hot-vs-scene contrast: p90 of hot region should exceed
        # scene median by a meaningful margin (rejects flat residual maps).
        contrast_ok = True
        if n_pix >= min_component:
            hot_med = float(np.nanmedian(col[keep]))
            scene_med = float(np.nanmedian(vals))
            scene_std = float(np.nanstd(vals))
            if not (hot_med > scene_med + max(0.5 * scene_std, 0.5 * thr_abs)):
                contrast_ok = False
        ok = n_pix >= min_component and contrast_ok
        info[name] = {
            "passed": bool(ok),
            "thr_abs": thr_abs,
            "thr_adaptive": thr,
            "percentile": float(percentile),
            "p_adapt": p_adapt,
            "n_raw": int(raw.sum()),
            "n_kept": n_pix,
            "n_components_kept": int(n_kept),
            "max_component": int(max_sz),
            "mineral_group": SPM_MINERAL_MAP[name],
            "contrast_ok": bool(contrast_ok),
        }
        if ok:
            active |= keep
            passed.append(name)
            print(f"  SPM-active {name}: thr={thr:.4f} "
                  f"(abs={thr_abs:.4f}, p{percentile:.0f}={p_adapt:.4f}) "
                  f"pixels={n_pix} comps>={min_component}:{n_kept}")
        else:
            reason = "low_contrast" if n_pix >= min_component and not contrast_ok else "kept"
            print(f"  SPM-gate skip {name}: thr={thr:.4f} kept={n_pix} "
                  f"({reason}; need ≥{min_component} + contrast)")

    print(f"  SPM-active domain: {int(active.sum())} pixels | "
          f"passing indices: {passed or ['(none)']}")
    return active, info, passed


def ice_proof_enabled(ctx: Optional[dict] = None, data_dir: Optional[str] = None) -> bool:
    """
    Polar / explicit ice PRESENT screening.

    Clustering domain stays ice-free (SPM_ACTIVE_EXCLUDE). Ice enters proof only
    when Scene 4 path, POLAR_SCENE_IDS, or CRISM_ALLOW_ICE_PROOF=1.
    """
    env = str(os.environ.get("CRISM_ALLOW_ICE_PROOF", "")).strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    path = data_dir or (ctx or {}).get("data_dir") or ""
    path_l = str(path).replace("\\", "/").lower()
    for sid in POLAR_SCENE_IDS:
        if f"scene {sid}" in path_l or f"scene_{sid}" in path_l:
            return True
    # Also honor explicit ctx flag
    if ctx and ctx.get("ice_proof_enabled"):
        return True
    return False


def evaluate_ice_spm_gates_for_proof(
    spm_map: np.ndarray,
    valid: np.ndarray,
    spm_notes: Optional[dict] = None,
    percentile: float = SPM_ADAPTIVE_PERCENTILE,
    min_component: int = SPM_MIN_COMPONENT,
) -> Tuple[dict, List[str]]:
    """
    Gate BD1500_2 / BD1435 for PRESENT screening only (not clustering domain).

    Same absolute thr + connected-component + contrast rules as non-ice indices.
    """
    from scipy import ndimage as ndi

    notes = spm_notes or {}
    info: dict = {}
    passed: List[str] = []
    for name in sorted(ICE_SPM_NAMES):
        if name not in SPM_NAMES:
            continue
        j = SPM_NAMES.index(name)
        if notes and not str(notes.get(name, "ok")).startswith("ok"):
            info[name] = {
                "passed": False,
                "reason": f"spm_note={notes.get(name)}",
                "clustering_excluded": True,
                "proof_candidate": False,
            }
            continue
        col = spm_map[:, :, j]
        fin = valid & np.isfinite(col)
        if fin.sum() < 100:
            info[name] = {
                "passed": False,
                "reason": "insufficient_finite",
                "clustering_excluded": True,
                "proof_candidate": False,
            }
            continue
        vals = col[fin]
        p_adapt = float(np.percentile(vals, percentile))
        thr_abs = float(SPM_DETECT_THRESH[name])
        thr = thr_abs if not SPM_GATE_USE_PERCENTILE_FLOOR else max(thr_abs, p_adapt)
        raw = fin & (col >= thr)
        labeled, ncomp = ndi.label(raw)
        keep = np.zeros_like(raw)
        n_kept = 0
        max_sz = 0
        if ncomp > 0:
            sizes = np.bincount(labeled.ravel())
            for lab_id in range(1, ncomp + 1):
                sz = int(sizes[lab_id]) if lab_id < len(sizes) else 0
                max_sz = max(max_sz, sz)
                if sz >= min_component:
                    keep |= labeled == lab_id
                    n_kept += 1
        n_pix = int(keep.sum())
        contrast_ok = True
        if n_pix >= min_component:
            hot_med = float(np.nanmedian(col[keep]))
            scene_med = float(np.nanmedian(vals))
            scene_std = float(np.nanstd(vals))
            if not (hot_med > scene_med + max(0.5 * scene_std, 0.5 * thr_abs)):
                contrast_ok = False
        ok = n_pix >= min_component and contrast_ok
        info[name] = {
            "passed": bool(ok),
            "thr_abs": thr_abs,
            "thr_adaptive": thr,
            "percentile": float(percentile),
            "p_adapt": p_adapt,
            "n_raw": int(raw.sum()),
            "n_kept": n_pix,
            "n_components_kept": int(n_kept),
            "max_component": int(max_sz),
            "mineral_group": SPM_MINERAL_MAP[name],
            "contrast_ok": bool(contrast_ok),
            "clustering_excluded": True,
            "proof_candidate": bool(ok),
            "reason": None if ok else (
                "low_contrast" if n_pix >= min_component and not contrast_ok else "kept"
            ),
        }
        if ok:
            passed.append(name)
            print(
                f"  Ice-proof gate {name}: thr={thr:.4f} pixels={n_pix} "
                f"(clustering_excluded; PRESENT screening only)"
            )
        else:
            print(
                f"  Ice-proof gate skip {name}: kept={n_pix} "
                f"({info[name].get('reason')}; need >={min_component} + contrast)"
            )
    return info, passed


def simple_band_depth(
    spectrum: np.ndarray,
    waves: np.ndarray,
    center: float,
    left: float,
    right: float,
    bad_mask: Optional[np.ndarray] = None,
) -> float:
    """Continuum-interpolated band depth at center; NaN if bands unavailable."""
    if bad_mask is None:
        bad_mask = np.zeros(len(waves), dtype=bool)
    ci = nearest_good_band(waves, center, bad_mask)
    li = nearest_good_band(waves, left, bad_mask)
    ri = nearest_good_band(waves, right, bad_mask)
    if ci is None or li is None or ri is None:
        return float("nan")
    # Require finite CR at all three bands (S4 may be fully NaN at ~2.067 µm)
    vals = spectrum[[ci, li, ri]]
    if not np.all(np.isfinite(vals)):
        return float("nan")
    cont = spectrum[li] + (spectrum[ri] - spectrum[li]) * (
        (waves[ci] - waves[li]) / (waves[ri] - waves[li] + 1e-9)
    )
    if not np.isfinite(cont) or abs(cont) < 1e-8:
        return float("nan")
    return float(1.0 - spectrum[ci] / cont)


def robust_band_depth(
    spectrum: np.ndarray,
    waves: np.ndarray,
    depth_key: str,
    bad_mask: Optional[np.ndarray] = None,
) -> float:
    """Try primary then fallback (center,left,right) triples for a depth key."""
    triples = PROOF_DEPTH_DEFS.get(depth_key)
    if not triples:
        return float("nan")
    for center, left, right in triples:
        d = simple_band_depth(spectrum, waves, center, left, right, bad_mask)
        if np.isfinite(d):
            return d
    return float("nan")


def compute_proof_depths(
    spectrum: np.ndarray,
    waves: np.ndarray,
    bad_mask: Optional[np.ndarray] = None,
) -> dict:
    """All proof diagnostic depths with fallbacks for dead shoulders."""
    return {k: robust_band_depth(spectrum, waves, k, bad_mask) for k in PROOF_DEPTH_DEFS}


def _band_depth_rows(
    specs: np.ndarray,
    waves: np.ndarray,
    center: float,
    left: float,
    right: float,
    bad_mask: np.ndarray,
) -> np.ndarray:
    """
    Vectorized continuum band depth for (N, B) CR spectra.
    Returns (N,) float64 with NaN where bands unavailable / non-finite.
    """
    ci = nearest_good_band(waves, center, bad_mask)
    li = nearest_good_band(waves, left, bad_mask)
    ri = nearest_good_band(waves, right, bad_mask)
    out = np.full(specs.shape[0], np.nan, dtype=np.float64)
    if ci is None or li is None or ri is None:
        return out
    rc = specs[:, ci].astype(np.float64)
    rl = specs[:, li].astype(np.float64)
    rr = specs[:, ri].astype(np.float64)
    ok = np.isfinite(rc) & np.isfinite(rl) & np.isfinite(rr)
    if not ok.any():
        return out
    denom = (waves[ri] - waves[li]) + 1e-9
    t = (waves[ci] - waves[li]) / denom
    cont = rl + (rr - rl) * t
    ok &= np.isfinite(cont) & (np.abs(cont) >= 1e-8)
    depth = np.full(specs.shape[0], np.nan, dtype=np.float64)
    depth[ok] = 1.0 - rc[ok] / cont[ok]
    return depth


def compute_cluster_depth_features(
    specs: np.ndarray,
    waves: np.ndarray,
    bad_mask: Optional[np.ndarray] = None,
    depth_keys: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Per-pixel diagnostic continuum depths for HDBSCAN fusion.

    Uses the same PROOF_DEPTH_DEFS triples as mineral proof (primary then
    fallbacks). Ice depths (d1.43 / d1.50) are never included.
    Returns (N, K) array and the key names used.
    """
    if bad_mask is None:
        bad_mask = np.zeros(len(waves), dtype=bool)
    keys = [
        k for k in (depth_keys or CLUSTER_DEPTH_KEYS)
        if k in PROOF_DEPTH_DEFS and k not in CLUSTER_DEPTH_EXCLUDE
    ]
    n = int(specs.shape[0])
    if not keys:
        return np.zeros((n, 0), dtype=np.float64), []
    feats = np.full((n, len(keys)), np.nan, dtype=np.float64)
    for j, key in enumerate(keys):
        col = np.full(n, np.nan, dtype=np.float64)
        for center, left, right in PROOF_DEPTH_DEFS[key]:
            d = _band_depth_rows(specs, waves, center, left, right, bad_mask)
            fill = ~np.isfinite(col) & np.isfinite(d)
            if fill.any():
                col[fill] = d[fill]
            if np.isfinite(col).all():
                break
        feats[:, j] = col
    return feats, keys


def mute_flat_depth_features(
    depth_raw: np.ndarray,
    depth_feat: np.ndarray,
    min_std: float = CLUSTER_DEPTH_MIN_STD,
    keys: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Zero z-scored depth columns with near-zero scene contrast.

    Prevents flat/noisy proof depths from dominating HDBSCAN after z-score
    (Scene2 adapt failure mode) without dropping informative channels.
    """
    out = np.array(depth_feat, dtype=np.float64, copy=True)
    muted: List[str] = []
    if out.ndim != 2 or out.shape[1] == 0:
        return out, muted
    for j in range(out.shape[1]):
        col = depth_raw[:, j]
        fin = np.isfinite(col)
        name = keys[j] if keys is not None and j < len(keys) else f"col{j}"
        if int(fin.sum()) < 50 or float(np.std(col[fin])) < float(min_std):
            out[:, j] = 0.0
            muted.append(str(name))
    return out, muted


def is_mixture_library_name(name: str) -> bool:
    """USGS mixture / doped spectra (Goeth+qtz.5+Jarosite.5, etc.)."""
    n = name or ""
    low = n.lower()
    return ("+" in n) or (".5" in n) or ("mix" in low) or ("+" in low)


def adaptive_bd_depth_floor(spm_name: str, bland_depth: float) -> float:
    """
    Scene-adaptive CR depth floor for PRESENT.

    Clays (Al-OH / Fe/Mg-OH): max(PROOF_DEPTH_BD_MIN, 0.75·PROOF_DEPTH_BD)
    so near-miss real absorptions (hot≫bland) are not blocked by a hard 0.008.
    Other BD indices keep the absolute PROOF_DEPTH_BD floor.
    """
    if spm_name in CLAY_BD_SPM:
        return float(max(PROOF_DEPTH_BD_MIN, 0.75 * PROOF_DEPTH_BD))
    return float(PROOF_DEPTH_BD)


def build_data_dead_mask(cube: np.ndarray, valid: np.ndarray, min_frac: float = 0.05) -> np.ndarray:
    """Bands with too little finite data among valid pixels."""
    H, W, B = cube.shape
    dead = np.zeros(B, dtype=bool)
    if not valid.any():
        return dead
    for bi in range(B):
        band = cube[:, :, bi]
        dead[bi] = float(np.mean(np.isfinite(band[valid]))) < min_frac
    return dead


def load_matching_img_cube(data_dir: str, sname: str, expect_shape: Tuple[int, int, int]):
    """Load TRR3 .img matching product name; return (cube, path) or (None, None)."""
    candidates = (
        glob.glob(os.path.join(data_dir, "**", f"{sname}.img"), recursive=True)
        + glob.glob(os.path.join(data_dir, f"{sname}.img"))
        + glob.glob(os.path.join(data_dir, "**", f"{sname}.IMG"), recursive=True)
    )
    if not candidates:
        return None, None
    img_path = candidates[0]
    lbl = find_label(img_path)
    if not lbl:
        return None, None
    meta = parse_pds3_label(lbl)
    try:
        cube = load_crism_cube(img_path, meta).astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Could not load SPM source img: {e}")
        return None, None
    cube = np.where(np.isfinite(cube), np.clip(cube, -0.05, 1.30), np.nan).astype(np.float32)
    if cube.shape != expect_shape:
        print(f"  [WARN] SPM img shape {cube.shape} != denoised {expect_shape}; ignoring img.")
        return None, None
    return cube, img_path


def nearest_good_band(
    waves: np.ndarray,
    target_wl: float,
    bad_mask: np.ndarray,
    tol_um: float = 0.030,
) -> Optional[int]:
    """Nearest good-band index within tolerance; None if unavailable."""
    dists = np.abs(waves - target_wl).astype(np.float64)
    dists[bad_mask] = np.inf
    bi = int(np.argmin(dists))
    if not np.isfinite(dists[bi]) or dists[bi] > tol_um:
        return None
    return bi


def _name_in_list(name: str, pool: Sequence[str]) -> bool:
    n = name.lower()
    return any(p.lower() in n or n in p.lower() for p in pool)


def mineral_family(name: str) -> str:
    """Collapse USGS specimen stems to a mineral family for unique endmember selection."""
    n = (name or "unknown").lower()
    for p in MARS_PRIOR_MINERALS:
        pl = p.lower()
        if pl in n:
            return pl
    stem = n.replace(" ", "_").split("_")[0].split(".")[0]
    return stem[:32] if stem else "unknown"


def _zscore_cols(arr: np.ndarray) -> np.ndarray:
    """Column-wise z-score; NaN -> 0. Used to fuse SPM into HDBSCAN features."""
    out = np.zeros_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        m = np.isfinite(col)
        if m.sum() < 10:
            continue
        mu = float(np.mean(col[m]))
        sd = float(np.std(col[m]))
        if sd < 1e-8:
            continue
        out[m, j] = (col[m] - mu) / sd
    return out


def build_sam_band_mask(waves: np.ndarray, good_mask: np.ndarray) -> np.ndarray:
    """Good bands for library SAM, excluding known artifact wavelength windows."""
    mask = good_mask.astype(bool).copy()
    for lo, hi in SAM_EXCLUDE_WINDOWS:
        mask &= ~((waves >= lo) & (waves <= hi))
    return mask


def estimate_bland_spectrum(
    cube: np.ndarray,
    valid: np.ndarray,
    bad_mask: np.ndarray,
    pct: float = BLAND_PCT,
) -> Tuple[np.ndarray, int]:
    """
    Estimate a bland (spectrally neutral) reference spectrum from the scene.

    Uses interior valid columns and the lowest-contrast spectra (low band-to-band
    std on good bands) — standard CRISM ratio denominator practice.
    """
    H, W, B = cube.shape
    # Avoid detector edge columns (common CRISM bad margins)
    margin = max(8, W // 40)
    interior = valid.copy()
    interior[:, :margin] = False
    interior[:, W - margin:] = False
    if interior.sum() < 100:
        interior = valid
    coords = np.argwhere(interior)
    if len(coords) == 0:
        bland = np.nanmedian(cube[valid], axis=0).astype(np.float32)
        return bland, 0

    # Subsample for speed on full FRT
    if len(coords) > 30000:
        sel = np.random.choice(len(coords), 30000, replace=False)
        coords = coords[sel]

    good = ~bad_mask
    contrast = np.full(len(coords), np.nan, dtype=np.float64)
    for i, (r, c) in enumerate(coords):
        spec = cube[r, c, good]
        fin = np.isfinite(spec)
        if fin.sum() < 20:
            continue
        contrast[i] = float(np.nanstd(spec[fin]))
    ok = np.isfinite(contrast)
    if ok.sum() < 20:
        bland = np.nanmedian(cube[valid], axis=0).astype(np.float32)
        return bland, 0

    thr = np.nanpercentile(contrast[ok], pct)
    bland_coords = coords[ok & (contrast <= thr)]
    specs = cube[bland_coords[:, 0], bland_coords[:, 1], :]
    bland = np.nanmedian(specs, axis=0).astype(np.float32)
    # Guard against zeros in bland denominator
    bland[~np.isfinite(bland) | (np.abs(bland) < 1e-6)] = np.nan
    return bland, int(len(bland_coords))


def apply_bland_ratio(
    cube: np.ndarray,
    bland: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Pixelwise I/F / bland (CRISM spectral ratio)."""
    out = np.full_like(cube, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = cube / bland[None, None, :]
    ratio = ratio.astype(np.float32)
    ratio[~valid] = np.nan
    # Keep physically plausible ratioed reflectance
    bad = ~np.isfinite(ratio) | (ratio < 0) | (ratio > 5)
    ratio[bad] = np.nan
    out[valid] = ratio[valid]
    return out


def mask_edge_columns(valid: np.ndarray, margin: int = EDGE_COL_MARGIN) -> np.ndarray:
    """Zero FRT detector edge columns (high uncertainty / bad optics)."""
    out = valid.copy()
    H, W = out.shape
    m = max(4, min(margin, W // 8))
    out[:, :m] = False
    out[:, W - m:] = False
    return out


def column_destripe(
    cube: np.ndarray,
    valid: np.ndarray,
    bad_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Additive pushbroom column-stripe correction (CRISM practice).

    For each band: subtract (column median − scene median) so vertical
    detector striping is suppressed without changing mineral methodology
    (still bland-ratio → Viviano-Beck SPM → CR → encoder → HDBSCAN).
    Returns (corrected cube, mean |residual| over corrected bands).
    """
    out = cube.astype(np.float32, copy=True)
    H, W, B = out.shape
    resid_sum = 0.0
    n_bands = 0
    for bi in range(B):
        if bad_mask[bi]:
            continue
        band = out[:, :, bi]
        masked = np.where(valid & np.isfinite(band), band, np.nan)
        col_med = np.nanmedian(masked, axis=0)
        fin = np.isfinite(col_med)
        if fin.sum() < max(8, W // 4):
            continue
        scene_med = float(np.nanmedian(col_med[fin]))
        if not np.isfinite(scene_med) or abs(scene_med) < 1e-8:
            continue
        residual = np.zeros(W, dtype=np.float32)
        residual[fin] = (col_med[fin] - scene_med).astype(np.float32)
        # Cap extreme column offsets (protect real spatial gradients)
        cap = float(np.nanpercentile(np.abs(residual[fin]), 98))
        if cap > 0:
            residual = np.clip(residual, -cap, cap)
        band = band - residual[None, :]
        band[~valid] = np.nan
        out[:, :, bi] = band
        resid_sum += float(np.mean(np.abs(residual[fin])))
        n_bands += 1
    mean_resid = resid_sum / max(1, n_bands)
    return out, mean_resid


def match_library_topk(
    mean_spec: np.ndarray,
    lib_cr: np.ndarray,
    lib_names: List[str],
    good_mask: np.ndarray,
    k: int = 5,
    prior_only: bool = True,
    sam_mask: Optional[np.ndarray] = None,
    family_stems: Optional[Sequence[str]] = None,
    require_family: bool = False,
    exclude_mixtures: bool = True,
) -> List[Tuple[str, float, int]]:
    """
    Top-k USGS SAM matches (ascending angle).

    If family_stems is set, only specimens whose name/family contains one of
    those stems are scored — prevents Quartz/Hematite continuum traps when
    matching clay/hydrate/carbonate SPM hotspots.
    Mixture spectra are excluded by default (authoritative labeling hygiene).
    """
    lib_mat = np.nan_to_num(lib_cr, nan=0.0)
    pix = np.nan_to_num(mean_spec, nan=0.0)
    finite_pix = np.isfinite(mean_spec) & good_mask
    if sam_mask is not None:
        finite_pix = finite_pix & sam_mask.astype(bool)
    stems = tuple(s.lower() for s in (family_stems or ()))

    def _in_family(name: str) -> bool:
        if not stems:
            return True
        n = (name or "").lower()
        fam = mineral_family(name)
        return any(s in n or s in fam for s in stems)

    scored: List[Tuple[str, float, int]] = []
    for li, name in enumerate(lib_names):
        if prior_only and not _name_in_list(name, MARS_PRIOR_MINERALS):
            continue
        if exclude_mixtures and is_mixture_library_name(name):
            continue
        if stems and not _in_family(name):
            continue
        m = finite_pix & np.isfinite(lib_cr[li])
        if m.sum() < 10:
            continue
        a, b = lib_mat[li, m], pix[m]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            continue
        ang = float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
        scored.append((name, ang, li))
    scored.sort(key=lambda t: t[1])
    if not scored and prior_only and not require_family:
        return match_library_topk(
            mean_spec, lib_cr, lib_names, good_mask,
            k=k, prior_only=False, sam_mask=sam_mask,
            family_stems=family_stems, require_family=require_family,
            exclude_mixtures=exclude_mixtures,
        )
    if not scored and stems and not require_family:
        return match_library_topk(
            mean_spec, lib_cr, lib_names, good_mask,
            k=k, prior_only=prior_only, sam_mask=sam_mask,
            family_stems=None, require_family=False,
            exclude_mixtures=exclude_mixtures,
        )
    if not scored and exclude_mixtures:
        return match_library_topk(
            mean_spec, lib_cr, lib_names, good_mask,
            k=k, prior_only=prior_only, sam_mask=sam_mask,
            family_stems=family_stems, require_family=require_family,
            exclude_mixtures=False,
        )
    return scored[:k]


def match_library_family_then_global(
    mean_spec: np.ndarray,
    lib_cr: np.ndarray,
    lib_names: List[str],
    good_mask: np.ndarray,
    spm_group: str,
    k: int = 5,
    sam_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[str, float, int]], List[Tuple[str, float, int]], str]:
    """
    Prefer family-restricted SAM for an SPM mineral group; keep global top-k
    for audit. Pure (non-mixture) spectra preferred for authoritative ranking.
    """
    stems = SPM_GROUP_SEED_FAMILIES.get(spm_group) or ()
    global_topk = match_library_topk(
        mean_spec, lib_cr, lib_names, good_mask,
        k=k, prior_only=True, sam_mask=sam_mask, exclude_mixtures=True,
    )
    if not stems:
        return global_topk, global_topk, "diagnostic_window_global"
    fam_topk = match_library_topk(
        mean_spec, lib_cr, lib_names, good_mask,
        k=k, prior_only=True, sam_mask=sam_mask,
        family_stems=stems, require_family=True, exclude_mixtures=True,
    )
    if fam_topk and fam_topk[0][1] <= FAMILY_SAM_HIT_MAX:
        return fam_topk, global_topk, "diagnostic_window_family"
    if fam_topk:
        return fam_topk, global_topk, "diagnostic_window_family_weak"
    return global_topk, global_topk, "diagnostic_window_global"


def olindex3_threepoint_map(
    cube: np.ndarray,
    waves: np.ndarray,
    spm_bad_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Stage4-compatible simplified olivine index (center/left/right band depth)
    used only when multipoint Viviano-Beck OLINDEX3 is unusable.
    Wavelengths follow Stage4 validation diagnostic (~1.21 / 1.08 / 1.69 µm).
    """
    targets = (1.21, 1.08, 1.69)
    bis = [nearest_good_band(waves, wl, spm_bad_mask) for wl in targets]
    H, W, _ = cube.shape
    out = np.full((H, W), np.nan, dtype=np.float32)
    if any(b is None for b in bis):
        return out
    ci, li, ri = bis
    rc, rl, rr = cube[:, :, ci], cube[:, :, li], cube[:, :, ri]
    frac = (waves[ci] - waves[li]) / (waves[ri] - waves[li] + 1e-9)
    cont = rl + frac * (rr - rl)
    with np.errstate(divide="ignore", invalid="ignore"):
        bd = 1.0 - rc / cont
    bd = bd.astype(np.float32)
    bad = (~valid) | (~np.isfinite(rc)) | (~np.isfinite(rl)) | (~np.isfinite(rr)) | (cont < 1e-6)
    bd[bad] = np.nan
    return bd


def band_depth(
    spectrum: np.ndarray,
    waves: np.ndarray,
    center_um: float,
    left_um: float,
    right_um: float,
    bad_mask: np.ndarray,
) -> float:
    """3-point continuum-interpolated band depth; NaN if bands invalid."""
    ci = nearest_good_band(waves, center_um, bad_mask)
    li = nearest_good_band(waves, left_um, bad_mask)
    ri = nearest_good_band(waves, right_um, bad_mask)
    if any(x is None for x in (ci, li, ri)):
        return np.nan

    rc = float(spectrum[ci])
    rl = float(spectrum[li])
    rr = float(spectrum[ri])
    if not all(np.isfinite(x) for x in (rc, rl, rr)):
        return np.nan

    frac = (waves[ci] - waves[li]) / (waves[ri] - waves[li] + 1e-9)
    cont = rl + float(frac) * (rr - rl)
    if cont < 1e-6:
        return np.nan
    return float(1.0 - rc / cont)


def precompute_spm_bands(
    waves: np.ndarray, bad_mask: np.ndarray
) -> Dict[str, Optional[Tuple[int, int, int]]]:
    """Cache (center, left, right) band indices for each SPM."""
    out = {}
    for name, (c, l, r) in SPM_INDICES.items():
        ci = nearest_good_band(waves, c, bad_mask)
        li = nearest_good_band(waves, l, bad_mask)
        ri = nearest_good_band(waves, r, bad_mask)
        out[name] = None if any(x is None for x in (ci, li, ri)) else (ci, li, ri)
    return out


def band_depth_cached(
    spectrum: np.ndarray,
    waves: np.ndarray,
    bands: Optional[Tuple[int, int, int]],
) -> float:
    if bands is None:
        return np.nan
    ci, li, ri = bands
    rc, rl, rr = float(spectrum[ci]), float(spectrum[li]), float(spectrum[ri])
    if not all(np.isfinite(x) for x in (rc, rl, rr)):
        return np.nan
    frac = (waves[ci] - waves[li]) / (waves[ri] - waves[li] + 1e-9)
    cont = rl + float(frac) * (rr - rl)
    if cont < 1e-6:
        return np.nan
    return float(1.0 - rc / cont)


def olindex3_multipoint(
    spectrum: np.ndarray,
    waves: np.ndarray,
    bad_mask: np.ndarray,
) -> float:
    """
    Viviano-Beck OLINDEX3 (multi-point) on I/F reflectance.
    OLINDEX3 = ((0.1*RB1054 + 0.1*RB1211 + 0.4*RB1329 + 0.4*RB1474) / RB1750) - 1
    Equivalent weighted formulation from the guide, normalized by RB1750.
    """
    targets = {
        "RB1054": 1.054,
        "RB1211": 1.211,
        "RB1329": 1.329,
        "RB1474": 1.474,
        "RB1750": 1.750,
    }
    vals = {}
    for k, wl in targets.items():
        bi = nearest_good_band(waves, wl, bad_mask)
        if bi is None or not np.isfinite(spectrum[bi]):
            return np.nan
        vals[k] = float(spectrum[bi])
    rb1750 = vals["RB1750"]
    if abs(rb1750) < 1e-6:
        return np.nan
    weighted = (
        vals["RB1054"] * 0.1
        + vals["RB1211"] * 0.1
        + vals["RB1329"] * 0.4
        + vals["RB1474"] * 0.4
    )
    return float(weighted / rb1750 - 1.0)


def compute_spm_vector(
    spectrum: np.ndarray,
    waves: np.ndarray,
    spm_bands: Dict[str, Optional[Tuple[int, int, int]]],
    bad_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute 12 L-sensor SPM indices on I/F (or library reflectance).
    Uses relaxed SPM band indices (precomputed with build_spm_bad_mask).
    """
    vals = np.empty(N_SPM, dtype=np.float32)
    for j, name in enumerate(SPM_NAMES):
        if name == "OLINDEX3" and bad_mask is not None:
            vals[j] = olindex3_multipoint(spectrum, waves, bad_mask)
        else:
            vals[j] = band_depth_cached(spectrum, waves, spm_bands[name])
    return vals


def compute_spm_map_vectorized(
    cube: np.ndarray,
    waves: np.ndarray,
    spm_bands: Dict[str, Optional[Tuple[int, int, int]]],
    spm_bad_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Fast full-cube SPM map (I/F domain)."""
    H, W, _ = cube.shape
    out = np.full((H, W, N_SPM), np.nan, dtype=np.float32)
    for j, name in enumerate(SPM_NAMES):
        if name == "OLINDEX3":
            targets = [1.054, 1.211, 1.329, 1.474, 1.750]
            bis = [nearest_good_band(waves, wl, spm_bad_mask) for wl in targets]
            if any(b is None for b in bis):
                continue
            rb1054 = cube[:, :, bis[0]]
            rb1211 = cube[:, :, bis[1]]
            rb1329 = cube[:, :, bis[2]]
            rb1474 = cube[:, :, bis[3]]
            rb1750 = cube[:, :, bis[4]]
            with np.errstate(divide="ignore", invalid="ignore"):
                weighted = 0.1 * rb1054 + 0.1 * rb1211 + 0.4 * rb1329 + 0.4 * rb1474
                vals = weighted / rb1750 - 1.0
            vals = vals.astype(np.float32)
            vals[~valid] = np.nan
            vals[~np.isfinite(rb1750) | (np.abs(rb1750) < 1e-6)] = np.nan
            out[:, :, j] = vals
            continue

        bands = spm_bands[name]
        if bands is None:
            continue
        ci, li, ri = bands
        rc, rl, rr = cube[:, :, ci], cube[:, :, li], cube[:, :, ri]
        frac = (waves[ci] - waves[li]) / (waves[ri] - waves[li] + 1e-9)
        cont = rl + frac * (rr - rl)
        with np.errstate(divide="ignore", invalid="ignore"):
            bd = 1.0 - rc / cont
        bd = bd.astype(np.float32)
        bad = (~valid) | (~np.isfinite(rc)) | (~np.isfinite(rl)) | (~np.isfinite(rr)) | (cont < 1e-6)
        bd[bad] = np.nan
        out[:, :, j] = bd
    return out


def sanitize_spm_map(spm_map: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Flag / neutralize artifact SPM channels for naming & training.
    Returns sanitized map + per-index status notes.
    """
    notes: Dict[str, str] = {}
    out = spm_map.copy()
    for j, name in enumerate(SPM_NAMES):
        col = out[:, :, j]
        finite = np.isfinite(col) & valid
        if finite.sum() < 20:
            notes[name] = "unavailable"
            continue
        vals = col[finite]
        med = float(np.median(vals))
        p90 = float(np.percentile(vals, 90))
        std = float(np.std(vals))

        if name in ICE_SPM_NAMES:
            # Ice should be sparse hotspots, not a scene-wide elevated continuum slope.
            hotspot_frac = float(np.mean(vals >= SPM_DETECT_THRESH[name]))
            if med > 0.05 and hotspot_frac > 0.50:
                out[:, :, j] = np.nan
                notes[name] = "disabled_scene_wide_artifact"
                continue
            if std < 0.005:
                out[:, :, j] = np.nan
                notes[name] = "disabled_low_contrast"
                continue
            notes[name] = "ok_gated"

        elif name == "OLINDEX3":
            # Multipoint OLINDEX3 should not be strongly negative everywhere.
            if p90 < 0.0:
                out[:, :, j] = np.nan
                notes[name] = "disabled_all_negative"
                continue
            # Also reject near-zero / no contrast (cannot drive mafic naming)
            if p90 < SPM_DETECT_THRESH[name] and std < 0.01:
                out[:, :, j] = np.nan
                notes[name] = "disabled_low_contrast"
                continue
            out[:, :, j] = np.clip(col, -0.5, 2.0)
            notes[name] = "ok_clipped"

        elif name == "LCPINDEX2":
            if med < -0.5 and p90 < 0.0:
                out[:, :, j] = np.nan
                notes[name] = "disabled_artifact_negative"
                continue
            if p90 < SPM_DETECT_THRESH[name] and std < 0.01:
                out[:, :, j] = np.nan
                notes[name] = "disabled_low_contrast"
                continue
            out[:, :, j] = np.clip(col, -1.0, 2.0)
            notes[name] = "ok_clipped"

        elif name in FLAT_HYDRATION_SPM:
            # Scene-flat elevated BD1900/BD2100 is usually atm residual on TRR3 I/F,
            # not spatially coherent hydration — do not let it name clusters.
            mean = float(np.mean(vals))
            cv = std / max(abs(mean), 1e-6)
            hotspot_frac = float(np.mean(vals >= max(SPM_DETECT_THRESH[name], 0.05)))
            if mean > 0.10 and cv < 0.30 and hotspot_frac > 0.70:
                out[:, :, j] = np.nan
                notes[name] = "disabled_scene_wide_artifact"
                continue
            out[:, :, j] = np.clip(col, -1.0, 2.0)
            notes[name] = "ok"

        else:
            # Soft clip absurd band depths
            out[:, :, j] = np.clip(col, -1.0, 2.0)
            notes[name] = "ok"

    return out, notes


def spectral_angle(s1: np.ndarray, s2: np.ndarray, band_mask: Optional[np.ndarray] = None) -> float:
    """Spectral angle in radians between two spectra."""
    finite = np.isfinite(s1) & np.isfinite(s2)
    if band_mask is not None:
        finite &= band_mask
    if finite.sum() < 10:
        return float(np.pi)
    a, b = s1[finite], s2[finite]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float(np.pi)
    cos_sim = float(np.dot(a, b) / (na * nb))
    return float(np.arccos(np.clip(cos_sim, -1.0, 1.0)))


def match_library_best(
    mean_spec: np.ndarray,
    lib_cr: np.ndarray,
    lib_names: List[str],
    good_mask: np.ndarray,
    prior_only: bool = True,
    sam_mask: Optional[np.ndarray] = None,
) -> Tuple[str, float, int]:
    """Best USGS match; optionally restrict to Mars-prior minerals / SAM windows."""
    best_angle = float(np.pi)
    best_name = "Unknown"
    best_idx = -1
    lib_mat = np.nan_to_num(lib_cr, nan=0.0)
    pix = np.nan_to_num(mean_spec, nan=0.0)
    finite_pix = np.isfinite(mean_spec) & good_mask
    if sam_mask is not None:
        finite_pix = finite_pix & sam_mask.astype(bool)
    for li, name in enumerate(lib_names):
        if prior_only and not _name_in_list(name, MARS_PRIOR_MINERALS):
            continue
        m = finite_pix & np.isfinite(lib_cr[li])
        if m.sum() < 10:
            continue
        a, b = lib_mat[li, m], pix[m]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            continue
        ang = float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
        if ang < best_angle:
            best_angle = ang
            best_name = name
            best_idx = li
    # Fallback to full library if prior set failed
    if best_idx < 0 and prior_only:
        return match_library_best(
            mean_spec, lib_cr, lib_names, good_mask,
            prior_only=False, sam_mask=sam_mask,
        )
    return best_name, best_angle, best_idx


def _lib_spm_family_agree(lib_name: str, spm_name: str) -> bool:
    """True when USGS specimen family is compatible with Viviano-Beck SPM group."""
    if not lib_name or lib_name == "Unknown" or not spm_name or spm_name == "Unknown":
        return True
    lf = mineral_family(lib_name)
    sn = spm_name.lower()
    # Broad compatible groups (methodology: SPM = group, library = specimen)
    if lf in ("olivine", "forsterite", "fayalite") and "olivine" in sn:
        return True
    if lf in ("pyroxene", "enstatite", "bronzite", "hypersthene",
              "diopside", "augite", "pigeonite"):
        if "pyroxene" in sn or "mafic" in sn:
            return True
        # Generic "Pyroxene" library on hydrated SPM → disagree
        if "hydrat" in sn or "clay" in sn or "sulfate" in sn or "carbonate" in sn:
            return False
        return "pyroxene" in sn
    if any(x in lf for x in ("nontronite", "saponite", "montmorillonite", "kaolinite",
                             "illite", "chlorite", "serpentine", "antigorite")):
        return any(x in sn for x in ("phyllosilicate", "clay", "hydrat", "serpentine"))
    if any(x in lf for x in ("gypsum", "kieserite", "jarosite", "alunite", "bassanite")):
        return "sulfate" in sn or "hydrat" in sn
    if any(x in lf for x in ("calcite", "dolomite", "magnesite", "siderite")):
        return "carbonate" in sn
    if any(x in lf for x in ("ice", "frost", "water", "co2")):
        return "ice" in sn
    # Unknown pairing: do not claim agreement (prevents bland-library traps)
    return False


def hybrid_cluster_label(
    spm_name: str,
    lib_name: str,
    sam: float,
    detect_score: float,
    sam_margin: float = 1.0,
    spm_gate_passed: bool = True,
) -> dict:
    """
    Authoritative labeling: discovery-first USGS naming with SPM as context.

    Strong library SAM (margin ≥ SAM_MARGIN_STRONG) can name a unit even when
    the specimen is outside the Viviano SPM family (library_open_usgs) — SPM
    does not gate discovery. Continuum-trap families are blocked. Tied SAM
    never yields a specimen name. Weak library → SPM enrichment or Unresolved.
    """
    agree = _lib_spm_family_agree(lib_name, spm_name)
    mix_lib = is_mixture_library_name(lib_name)
    lib_fam = mineral_family(lib_name)
    if not spm_gate_passed and spm_name not in ("Unknown", ""):
        return {
            "authoritative_label": "Unresolved",
            "display_label": "Unresolved",
            "label_basis": "spm_gate_failed",
            "spm_name": "Unknown",
            "library_match": lib_name,
            "spectral_angle_rad": float(sam),
            "sam_margin_rad": float(sam_margin),
            "lib_spm_agree": False,
        }
    # Tied / non-discriminative library → Unresolved (keep SPM group for context)
    if sam_margin < SAM_MARGIN_STRONG and lib_name not in ("Unknown", ""):
        _spm = spm_name if spm_name not in ("Unknown", "") else "Unknown"
        return {
            "authoritative_label": "Unresolved",
            "display_label": _spm if _spm != "Unknown" else "Unresolved",
            "label_basis": "unresolved_tied_sam",
            "spm_name": _spm,
            "library_match": lib_name,
            "spectral_angle_rad": float(sam),
            "sam_margin_rad": float(sam_margin),
            "lib_spm_agree": bool(agree),
        }
    # Mixture library entries without strong margin → do not name the unit
    if mix_lib and sam_margin < SAM_MARGIN_STRONG and detect_score >= 0.20:
        _spm = spm_name if spm_name not in ("Unknown", "") else "Unknown"
        return {
            "authoritative_label": "Unresolved",
            "display_label": _spm if _spm != "Unknown" else "Unresolved",
            "label_basis": "unresolved_mixture_library",
            "spm_name": _spm,
            "library_match": lib_name,
            "spectral_angle_rad": float(sam),
            "sam_margin_rad": float(sam_margin),
            "lib_spm_agree": bool(agree),
        }
    # Open-set / strong USGS: discriminative library wins (SPM is context only)
    if (
        lib_name != "Unknown"
        and not mix_lib
        and sam < SAM_STRONG_MAX
        and sam_margin >= SAM_MARGIN_STRONG
        and lib_fam not in OPEN_USGS_BLOCKLIST
    ):
        if agree:
            authoritative = lib_name
            basis = "library_strong"
        elif spm_name in ("Unknown", "") or detect_score < 0.20:
            authoritative = lib_name
            basis = "library_open_usgs"
        elif sam_margin >= 1.5 * SAM_MARGIN_STRONG:
            # Strong out-of-family USGS hit — allow despite SPM enrichment
            authoritative = lib_name
            basis = "library_open_usgs"
        elif (
            sam < SAM_SUSPICIOUS_TIGHT
            and not agree
            and detect_score >= 0.20
        ):
            # Tiny angle + SPM disagree without extra margin → continuum risk
            authoritative = "Unresolved"
            basis = "unresolved_suspicious_sam"
        else:
            authoritative = lib_name
            basis = "library_strong" if agree else "library_open_usgs"
    elif (
        lib_name != "Unknown"
        and spm_name not in ("Unknown", "")
        and sam < SAM_SUSPICIOUS_TIGHT
        and not agree
        and detect_score >= 0.20
    ):
        authoritative = "Unresolved"
        basis = "unresolved_suspicious_sam"
    elif detect_score >= 0.25 and spm_name not in ("Unknown", ""):
        if sam >= 0.15 or lib_name == "Unknown" or not agree:
            if not agree and lib_name != "Unknown" and sam_margin < SAM_MARGIN_STRONG:
                authoritative = "Unresolved"
                basis = "unresolved_lib_spm_disagree"
            else:
                authoritative = spm_name
                basis = "spm_enrichment"
        else:
            authoritative = f"{spm_name} / {lib_name}"
            basis = "hybrid_strong_lib"
    elif sam < 0.30 and lib_name != "Unknown" and spm_name not in ("Unknown", ""):
        if agree and sam_margin >= SAM_MARGIN_STRONG:
            authoritative = f"{spm_name} / {lib_name}"
            basis = "hybrid_moderate"
        else:
            authoritative = "Unresolved"
            basis = "unresolved_lib_spm_disagree"
    elif detect_score >= 0.15 and spm_name not in ("Unknown", ""):
        authoritative = spm_name
        basis = "spm_weak"
    elif sam < 0.30 and lib_name != "Unknown":
        authoritative = lib_name if sam_margin >= SAM_MARGIN_STRONG else "Unresolved"
        basis = "library_only" if sam_margin >= SAM_MARGIN_STRONG else "unresolved_tied_sam"
    else:
        authoritative = "Unresolved"
        basis = "insufficient_evidence"
    # Map/legend readability: SPM group when specimen abstains (proof still authority)
    if (
        authoritative == "Unresolved"
        and spm_name not in ("Unknown", "", None)
        and "tied" in basis
    ):
        display_label = spm_name
    elif authoritative == "Unresolved" and spm_name not in ("Unknown", "", None):
        display_label = spm_name
    else:
        display_label = authoritative
    return {
        "authoritative_label": authoritative,
        "display_label": display_label,
        "label_basis": basis,
        "spm_name": spm_name,
        "library_match": lib_name,
        "spectral_angle_rad": float(sam),
        "sam_margin_rad": float(sam_margin),
        "lib_spm_agree": bool(agree),
    }


def split_mega_clusters_by_competing_spm(
    labels: np.ndarray,
    probabilities: np.ndarray,
    spm_flat: np.ndarray,
    gate_info: dict,
    active_idx: np.ndarray,
    n_domain: int,
    mega_frac: float = MEGA_CLUSTER_FRAC,
    max_rounds: int = MEGA_SPLIT_MAX_ROUNDS,
    compete_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    label: str = "Mega-split",
    allow_relative_hotspot_split: bool = False,
    diagnostics: Optional[List[dict]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    If a cluster covers > mega_frac of the SPM-active domain and two competing
    SPM channels both have substantial hotspots inside it, split the mega-unit
    by which SPM is stronger per pixel (BD1900 vs SINDEX, clay pairs, etc.).

    Runs up to max_rounds so hydrate/sulfate units can later split by Al-OH vs
    Fe/Mg-OH. Keeps HDBSCAN methodology: only post-processes oversized mixed units.

    allow_relative_hotspot_split: when True (clay-secondary), if exclusive sides
    are thin but both indices have large hotspots (typical overlap), assign every
    hot pixel by relative SPM strength and split if both sides ≥ min_side.

    Returns updated labels, probabilities, and number of splits performed.
    """
    labels = labels.copy()
    probabilities = probabilities.copy()
    n_valid = len(labels)
    total_splits = 0
    thr_domain = max(80, int(mega_frac * max(n_domain, 1)))
    pairs = tuple(compete_pairs) if compete_pairs is not None else SPM_COMPETE_PAIRS
    active_mask_full = np.zeros(n_valid, dtype=bool)
    if len(active_idx):
        active_mask_full[np.asarray(active_idx, dtype=int)] = True
    if diagnostics is None:
        diagnostics = []

    # Build per-index hotspot on valid pixels from adaptive gates
    hot = np.zeros_like(spm_flat, dtype=bool)
    for j, name in enumerate(SPM_NAMES):
        if name in ICE_SPM_NAMES:
            continue
        g = gate_info.get(name) or {}
        if not g.get("passed"):
            continue
        col = spm_flat[:, j]
        fin = np.isfinite(col)
        thr = float(g.get("thr_adaptive", SPM_DETECT_THRESH.get(name, 0.01)))
        hot[:, j] = fin & (col >= thr)

    print(
        f"  {label} check: thr_domain={thr_domain} "
        f"(mega_frac={mega_frac:.2f} * n_domain={n_domain}"
        f"{'; relative_hotspot' if allow_relative_hotspot_split else ''})"
    )
    for round_i in range(max(1, int(max_rounds))):
        next_id = int(labels.max()) + 1 if labels.max() >= 0 else 0
        n_splits_round = 0
        cluster_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
        for cl in cluster_ids:
            cl_mask = labels == cl
            n_cl = int(cl_mask.sum())
            n_cl_active = int((cl_mask & active_mask_full).sum())
            frac_active = n_cl_active / max(n_domain, 1)
            if n_cl < thr_domain and n_cl_active < thr_domain:
                note = (
                    f"size n={n_cl} active={n_cl_active} "
                    f"({100 * frac_active:.1f}% domain) < thr={thr_domain}"
                )
                print(f"  {label} skip cl{cl}: {note}")
                diagnostics.append({
                    "pass": label, "cluster": cl, "action": "skip_size",
                    "note": note, "n_cl": n_cl, "n_cl_active": n_cl_active,
                    "thr_domain": thr_domain,
                })
                continue

            split_done = False
            best_pair_note = None
            for name_a, name_b in pairs:
                if name_a not in SPM_NAMES or name_b not in SPM_NAMES:
                    continue
                ga = gate_info.get(name_a) or {}
                gb = gate_info.get(name_b) or {}
                if not (ga.get("passed") and gb.get("passed")):
                    continue
                ia = SPM_NAMES.index(name_a)
                ib = SPM_NAMES.index(name_b)
                a_hot = cl_mask & hot[:, ia]
                b_hot = cl_mask & hot[:, ib]
                only_a = a_hot & ~b_hot
                only_b = b_hot & ~a_hot
                both = a_hot & b_hot
                n_a = int(only_a.sum())
                n_b = int(only_b.sum())
                n_both = int(both.sum())
                n_a_hot = int(a_hot.sum())
                n_b_hot = int(b_hot.sum())
                min_side = max(40, SPM_MIN_COMPONENT)
                exclusive_ok = n_a >= min_side and n_b >= min_side
                relative_ok = (
                    allow_relative_hotspot_split
                    and n_a_hot >= min_side
                    and n_b_hot >= min_side
                )
                if not exclusive_ok and not relative_ok:
                    note = (
                        f"{name_a}vs{name_b}: only_a={n_a} only_b={n_b} "
                        f"both={n_both} a_hot={n_a_hot} b_hot={n_b_hot} "
                        f"(need exclusive ≥{min_side}"
                        + (
                            f" or relative hotspots ≥{min_side}"
                            if allow_relative_hotspot_split else ""
                        )
                        + ")"
                    )
                    if best_pair_note is None or (n_a + n_b + n_a_hot + n_b_hot) > 0:
                        best_pair_note = note
                    continue

                # Relative assign: exclusive path uses only_a/only_b + both;
                # relative path assigns all a|b hot pixels by priority×SPM.
                if exclusive_ok:
                    if n_both > 0:
                        va = spm_flat[both, ia].astype(np.float64)
                        vb = spm_flat[both, ib].astype(np.float64)
                        pri_a = float(SPM_LABEL_PRIORITY.get(name_a, 1.0))
                        pri_b = float(SPM_LABEL_PRIORITY.get(name_b, 1.0))
                        prefer_a = (pri_a * np.nan_to_num(va, nan=0.0)) >= (
                            pri_b * np.nan_to_num(vb, nan=0.0)
                        )
                        both_idx = np.where(both)[0]
                        only_a_extra = both_idx[prefer_a]
                        only_b_extra = both_idx[~prefer_a]
                    else:
                        only_a_extra = np.array([], dtype=int)
                        only_b_extra = np.array([], dtype=int)
                    a_idx = np.unique(np.concatenate([
                        np.where(only_a)[0], only_a_extra
                    ]))
                    b_idx = np.unique(np.concatenate([
                        np.where(only_b)[0], only_b_extra
                    ]))
                    split_mode = "exclusive"
                else:
                    contest = a_hot | b_hot
                    c_idx = np.where(contest)[0]
                    va = spm_flat[c_idx, ia].astype(np.float64)
                    vb = spm_flat[c_idx, ib].astype(np.float64)
                    pri_a = float(SPM_LABEL_PRIORITY.get(name_a, 1.0))
                    pri_b = float(SPM_LABEL_PRIORITY.get(name_b, 1.0))
                    prefer_a = (pri_a * np.nan_to_num(va, nan=0.0)) >= (
                        pri_b * np.nan_to_num(vb, nan=0.0)
                    )
                    a_idx = c_idx[prefer_a]
                    b_idx = c_idx[~prefer_a]
                    split_mode = "relative_hotspot"

                if len(a_idx) < min_side or len(b_idx) < min_side:
                    best_pair_note = (
                        f"{name_a}vs{name_b}: after {split_mode} assign "
                        f"side_a={len(a_idx)} side_b={len(b_idx)} "
                        f"< {min_side} (only_a={n_a} only_b={n_b} both={n_both} "
                        f"a_hot={n_a_hot} b_hot={n_b_hot})"
                    )
                    continue
                # Peel the smaller side into a new cluster id (keeps larger as cl)
                if len(b_idx) <= len(a_idx):
                    move_idx, keep_n, new_n = b_idx, len(a_idx), len(b_idx)
                else:
                    move_idx, keep_n, new_n = a_idx, len(b_idx), len(a_idx)
                labels[move_idx] = next_id
                probabilities[move_idx] = np.maximum(probabilities[move_idx], 0.40)
                print(
                    f"  {label} cl{cl} split by {name_a} vs {name_b} "
                    f"(round {round_i + 1}, {split_mode}): "
                    f"keep={keep_n} -> cl{cl}, "
                    f"new={new_n} -> cl{next_id} "
                    f"(frac_active={100 * frac_active:.1f}%)"
                )
                diagnostics.append({
                    "pass": label, "cluster": cl, "action": "split",
                    "pair": [name_a, name_b], "mode": split_mode,
                    "keep": int(keep_n), "new": int(new_n),
                    "new_id": int(next_id),
                    "only_a": n_a, "only_b": n_b, "both": n_both,
                    "a_hot": n_a_hot, "b_hot": n_b_hot,
                })
                next_id += 1
                n_splits_round += 1
                split_done = True
                break  # one split pair per mega-cluster per round
            if not split_done:
                print(
                    f"  {label} skip cl{cl}: large enough "
                    f"(n={n_cl}, active={n_cl_active}, "
                    f"{100 * frac_active:.1f}% domain) but no competing "
                    f"hotspot pair - {best_pair_note or 'no passed compete pairs'}"
                )
                diagnostics.append({
                    "pass": label, "cluster": cl, "action": "skip_no_pair",
                    "note": best_pair_note or "no passed compete pairs",
                    "n_cl": n_cl, "n_cl_active": n_cl_active,
                })

        total_splits += n_splits_round
        if n_splits_round == 0:
            if round_i == 0:
                print(f"  {label}: 0 splits this round (see skip reasons above)")
            break
        # Compact ids before next round so new units can be re-evaluated
        old_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
        remap = {old: new for new, old in enumerate(old_ids)}
        new_labels = np.full_like(labels, -1)
        for old, new in remap.items():
            new_labels[labels == old] = new
        labels = new_labels

    return labels, probabilities, total_splits


def is_quality_spectrum(spectrum_438: np.ndarray, bad_mask_438: np.ndarray) -> bool:
    """Library quality filters from the guide."""
    good_bi = np.where(~bad_mask_438)[0]
    good_vals = spectrum_438[good_bi]
    nan_frac = float(np.sum(np.isnan(good_vals)) / max(1, len(good_vals)))
    if nan_frac > 0.30:
        return False
    finite_vals = good_vals[np.isfinite(good_vals)]
    if len(finite_vals) > 1 and float(np.max(np.abs(np.diff(finite_vals)))) > 0.5:
        return False
    finite_all = spectrum_438[np.isfinite(spectrum_438)]
    if len(finite_all) > 0:
        if float(np.min(finite_all)) < -0.05 or float(np.max(finite_all)) > 1.5:
            return False
    return True


# -------------------------------------------------------------
# MODELS
# -------------------------------------------------------------
class SpectralEncoder(nn.Module):
    """1D CNN encoder: (B, 1, 438) -> (B, latent) + SPM head + recon head.

    Reconstruction head limits information loss from 438→latent so clustering
    fingerprints still carry diagnostic absorption structure.
    """

    def __init__(self, n_bands: int = N_BANDS, latent_dim: int = LATENT_DIM, n_spm: int = N_SPM):
        super().__init__()
        self.n_bands = n_bands
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
        )
        self.spm_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_spm),
            nn.Sigmoid(),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_bands),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.gap(h).squeeze(2)
        return self.fc(h)

    def predict_spm(self, latent: torch.Tensor) -> torch.Tensor:
        return self.spm_head(latent)

    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        return self.recon_head(latent)


class AbundanceNetwork(nn.Module):
    """Predicts mineral abundance fractions; softmax enforces ≥0 and sum=1."""

    def __init__(self, latent_dim: int = LATENT_DIM, n_endmembers: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_endmembers),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(z), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent contrastive loss on L2-normalized latent pairs."""
    # Force float32: AMP half cannot represent large negative mask fill values.
    z1 = z1.float()
    z2 = z2.float()
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def augment_spectrum(
    spectrum_1d: np.ndarray,
    protected_bands: set,
) -> np.ndarray:
    """Mineralogically-informed augmentation (scale / smooth / protected dropout)."""
    aug = spectrum_1d.astype(np.float32, copy=True)
    aug *= np.random.uniform(0.85, 1.15)
    if np.random.random() < 0.5:
        aug = gaussian_filter1d(aug, np.random.uniform(0.3, 0.8)).astype(np.float32)
    all_bands = set(range(len(aug)))
    unprotected = list(all_bands - protected_bands)
    if unprotected:
        n_drop = max(1, int(0.05 * len(unprotected)))
        drop_bi = np.random.choice(unprotected, min(n_drop, len(unprotected)), replace=False)
        aug[drop_bi] = 0.0
    return aug


# -------------------------------------------------------------
# STAGE 0 — Setup and Loading
# -------------------------------------------------------------
def stage0_setup(data_dir: str, output_dir: str) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 0 — Setup and Loading")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)
    ml_dir = os.path.join(data_dir, "ml_output")
    phys_dir = os.path.join(data_dir, "physics_output")

    cube = None
    cube_source = None
    sname = None
    meta: dict = {}

    s4_files = sorted(glob.glob(os.path.join(ml_dir, "**", "*_S4.npy"), recursive=True))
    s3_files = sorted(glob.glob(os.path.join(phys_dir, "**", "*_S3.npy"), recursive=True))
    img_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*l_trr3.img"), recursive=True)
        + glob.glob(os.path.join(data_dir, "*l_trr3.img"))
        + glob.glob(os.path.join(data_dir, "**", "*L_TRR3.IMG"), recursive=True)
    )
    # Deduplicate
    seen = set()
    img_files = [f for f in img_files if not (os.path.basename(f).lower() in seen or seen.add(os.path.basename(f).lower()))]

    if s4_files:
        l_files = [f for f in s4_files if "l_trr3" in os.path.basename(f).lower()]
        path = l_files[0] if l_files else s4_files[0]
        cube = np.load(path).astype(np.float32)
        sname = os.path.basename(path).replace("_S4.npy", "")
        cube_source = f"S4:{path}"
    elif s3_files:
        l_files = [f for f in s3_files if "l_trr3" in os.path.basename(f).lower()]
        path = l_files[0] if l_files else s3_files[0]
        cube = np.load(path).astype(np.float32)
        sname = os.path.basename(path).replace("_S3.npy", "")
        cube_source = f"S3:{path}"
        print("  [WARN] Using physics-corrected S3 (no ML denoising S4 found).")
    elif img_files:
        path = img_files[0]
        sname = os.path.splitext(os.path.basename(path))[0]
        lbl = find_label(path)
        if not lbl:
            raise FileNotFoundError(f"No .lbl found beside {path}")
        meta = parse_pds3_label(lbl)
        cube = load_crism_cube(path, meta).astype(np.float32)
        # Physical I/F clamp
        cube = np.where(np.isfinite(cube), np.clip(cube, -0.05, 1.30), np.nan).astype(np.float32)
        cube_source = f"IMG:{path}"
        print("  [WARN] No *_S4.npy — loading raw/TRR3 .img directly.")
        print("         For best results run Stages 1-4 first, then Stage 5.")
    else:
        raise FileNotFoundError(
            f"No *_S4.npy, *_S3.npy, or L-sensor .img found under {data_dir}"
        )

    H, W, B = cube.shape
    if B != N_BANDS:
        print(f"  [WARN] Expected {N_BANDS} bands, got {B}. Proceeding with B={B}.")
    if H < FULL_HEIGHT_MIN_LINES:
        print(f"  [WARN] Spatial height H={H} < {FULL_HEIGHT_MIN_LINES}: "
              f"subset-like footprint; mineral diversity may be weak.")
    else:
        print(f"  Full-height FRT detected: H={H}, W={W}")

    # Wavelengths: prefer label, else USGS CRISM L descending axis (cached).
    if not meta:
        img_candidates = (
            glob.glob(os.path.join(data_dir, "**", f"{sname}.img"), recursive=True)
            + glob.glob(os.path.join(data_dir, f"{sname}.img"))
        )
        if img_candidates:
            lbl = find_label(img_candidates[0])
            if lbl:
                meta = parse_pds3_label(lbl)
    waves = get_sensor_waves(
        B, meta.get("wavelengths"), data_dir=data_dir, cube=cube
    )
    wave_info = assert_crism_L_wavelengths(waves, cube=cube)
    bad_mask = build_bad_mask(waves)
    spm_bad_mask = build_spm_bad_mask(waves)
    mineral_bad_mask = build_mineral_bad_mask(waves)
    good_mask = ~bad_mask
    mineral_good_mask = ~mineral_bad_mask

    dead_bands: List[int] = []
    eda_json = os.path.join(data_dir, "eda_output", "eda_summary.json")
    if os.path.exists(eda_json):
        with open(eda_json, "r") as f:
            eda = json.load(f)
        if sname in eda:
            dead_bands = eda[sname].get("dead_bands", []) or []
        else:
            for key, val in eda.items():
                if isinstance(val, dict) and "dead_bands" in val:
                    if sname in key or key in sname:
                        dead_bands = val.get("dead_bands", []) or []
                        break

    # Encoding / CR: mark EDA dead bands as bad (pipeline + mineral paths)
    for d in dead_bands:
        if 0 <= int(d) < B:
            bad_mask[int(d)] = True
            mineral_bad_mask[int(d)] = True
    good_mask = ~bad_mask
    mineral_good_mask = ~mineral_bad_mask

    good_bi = np.where(good_mask)[0]
    if len(good_bi) == 0:
        raise RuntimeError("No good bands available after masking.")
    finite_count = np.sum(np.isfinite(cube[:, :, good_bi]), axis=2)
    valid_pixel_mask = finite_count >= (len(good_bi) * 0.5)

    # SPM source: prefer original TRR3 .img so atmospheric caution bands
    # blanked by Stage 3/4 are still available for Viviano-Beck indices.
    cube_spm = None
    spm_source = None
    if cube_source.startswith("IMG:"):
        cube_spm = cube
        spm_source = cube_source
    else:
        cube_spm, img_path = load_matching_img_cube(data_dir, sname, (H, W, B))
        if cube_spm is not None:
            spm_source = f"IMG:{img_path}"
            print("  SPM source: original TRR3 .img (keeps caution-zone bands)")
        else:
            cube_spm = cube
            spm_source = cube_source
            print("  [WARN] No matching .img for SPM — using denoised cube "
                  "(some SPM indices may be unavailable).")

    # Exclude only hard SPM excludes + bands empty in the SPM source cube
    data_dead = build_data_dead_mask(cube_spm, valid_pixel_mask, min_frac=0.05)
    spm_select_mask = spm_bad_mask | data_dead
    # Keep EDA edge dead bands out of SPM selection too
    for d in dead_bands:
        if 0 <= int(d) < B:
            spm_select_mask[int(d)] = True

    print(f"  Cube source: {cube_source}")
    print(f"  SPM source : {spm_source}")
    print(f"  Cube shape: {cube.shape}")
    print(f"  Wavelength range: {waves[0]:.3f}-{waves[-1]:.3f} um")
    print(f"  Good bands (encoding/S4): {int(good_mask.sum())} / {B}")
    print(f"  Mineral CR/SAM bands: {int(mineral_good_mask.sum())} / {B} "
          f"(diagnostic wavelengths restored)")
    print(f"  SPM-usable bands: {int((~spm_select_mask).sum())} / {B} "
          f"(data-dead in SPM cube: {int(data_dead.sum())})")
    print(f"  Dead bands (EDA): {len(dead_bands)}")
    print(f"  Valid pixels: {int(valid_pixel_mask.sum())} / {H * W}")
    print(f"  Device: {_device()}")

    return {
        "sname": sname,
        "cube_denoised": cube,
        "cube_spm": cube_spm,
        "waves": waves.astype(np.float32),
        "bad_mask": bad_mask,
        "mineral_bad_mask": mineral_bad_mask,
        "spm_bad_mask": spm_select_mask,
        "good_mask": good_mask,
        "mineral_good_mask": mineral_good_mask,
        "valid_pixel_mask": valid_pixel_mask,
        "dead_bands": [int(d) for d in dead_bands],
        "H": H, "W": W, "B": B,
        "output_dir": output_dir,
        "data_dir": data_dir,
        "cube_source": cube_source,
        "spm_source": spm_source,
    }


# -------------------------------------------------------------
# STAGE 1 — Spectral Library Preprocessing
# -------------------------------------------------------------
def stage1_library(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 1 — USGS Spectral Library Preprocessing")
    print("-" * 60)

    waves = ctx["waves"]
    bad_mask = ctx.get("mineral_bad_mask", ctx["bad_mask"])
    spm_bad_mask = ctx.get("spm_bad_mask", build_spm_bad_mask(waves))
    dead_bands = ctx["dead_bands"]
    B = ctx["B"]
    output_dir = ctx["output_dir"]

    waves_file = os.path.join(USGS_DIR, "s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt")
    if not os.path.exists(waves_file):
        raise FileNotFoundError(f"USGS wavelength file not found: {waves_file}")

    lib_waves_489 = parse_usgs_ascii_values(waves_file)
    print(f"  Library wavelengths: {len(lib_waves_489)} ch, "
          f"{lib_waves_489[0]:.3f}-{np.nanmax(lib_waves_489):.3f} um")
    print("  NOTE: Library is already CRISM-convolved — NO SRF applied.")

    # 489 -> scene band mapping
    lib_to_scene_map: List[Optional[int]] = []
    for sw in waves:
        closest = int(np.argmin(np.abs(lib_waves_489 - sw)))
        if abs(float(lib_waves_489[closest]) - float(sw)) < 0.010:
            lib_to_scene_map.append(closest)
        else:
            lib_to_scene_map.append(None)
    n_mapped = sum(1 for x in lib_to_scene_map if x is not None)
    print(f"  Mapped library->scene channels: {n_mapped}/{B}")

    mineral_dir = os.path.join(USGS_DIR, "ChapterM_Minerals")
    soil_dir = os.path.join(USGS_DIR, "ChapterS_SoilsAndMixtures")
    files = sorted(glob.glob(os.path.join(mineral_dir, "*.txt")))
    if os.path.isdir(soil_dir):
        files += sorted(glob.glob(os.path.join(soil_dir, "*.txt")))
    print(f"  Spectrum files found: {len(files)}")

    lib_cr_list, lib_l2_list, lib_spm_list, lib_names = [], [], [], []
    spm_bands = precompute_spm_bands(waves, spm_bad_mask)
    skipped = 0
    n_prior = 0

    for fp in files:
        try:
            _header, values_489 = parse_usgs_spectrum(fp)
        except Exception:
            skipped += 1
            continue
        if values_489.size < 489:
            tmp = np.full(489, np.nan, dtype=np.float32)
            n = min(489, values_489.size)
            tmp[:n] = values_489[:n]
            values_489 = tmp

        spec_438 = np.full(B, np.nan, dtype=np.float32)
        for si, li in enumerate(lib_to_scene_map):
            if li is not None and li < values_489.size:
                spec_438[si] = values_489[li]

        if not is_quality_spectrum(spec_438, bad_mask):
            skipped += 1
            continue

        spec_438 = spec_438.copy()
        spec_438[bad_mask] = np.nan
        for d in dead_bands:
            if 0 <= d < B:
                spec_438[d] = np.nan

        filled = fill_nans_interp(spec_438, waves)
        if filled is None:
            skipped += 1
            continue

        spm = compute_spm_vector(filled, waves, spm_bands, bad_mask=spm_bad_mask)

        # Matching-consistent CR (same algorithm as scene cube_cr)
        cr = continuum_removal_match(filled, waves)
        cr[bad_mask] = np.nan
        for d in dead_bands:
            if 0 <= d < B:
                cr[d] = np.nan

        if np.sum(np.isfinite(cr)) < 20:
            skipped += 1
            continue

        l2 = l2_normalize(cr)
        mname = extract_mineral_name(fp)
        if _name_in_list(mname, MARS_PRIOR_MINERALS):
            n_prior += 1

        lib_cr_list.append(cr.astype(np.float32))
        lib_l2_list.append(l2.astype(np.float32))
        lib_spm_list.append(spm)
        lib_names.append(mname)

    if not lib_cr_list:
        raise RuntimeError("No valid USGS library spectra after quality filtering.")

    lib_spectra_cr = np.stack(lib_cr_list, axis=0)
    lib_spectra_cr_l2 = np.stack(lib_l2_list, axis=0)
    lib_spm = np.stack(lib_spm_list, axis=0)

    safe_np_save(os.path.join(output_dir, "lib_spectra_438_cr.npy"), lib_spectra_cr)
    safe_np_save(os.path.join(output_dir, "lib_spectra_438_cr_l2.npy"), lib_spectra_cr_l2)
    safe_np_save(os.path.join(output_dir, "lib_spm.npy"), lib_spm)
    with open(os.path.join(output_dir, "lib_names.json"), "w") as f:
        json.dump(lib_names, f)

    avail = {n: (spm_bands[n] is not None or n == "OLINDEX3") for n in SPM_NAMES}
    print(f"  Valid library spectra: {len(lib_names)} (skipped {skipped})")
    print(f"  Unique mineral names : {len(set(lib_names))}")
    print(f"  Mars-prior matches   : {n_prior}")
    print(f"  SPM band availability: {avail}")

    ctx.update({
        "lib_spectra_cr": lib_spectra_cr,
        "lib_spectra_cr_l2": lib_spectra_cr_l2,
        "lib_spm": lib_spm,
        "lib_names": lib_names,
        "spm_bands": spm_bands,
        "spm_bad_mask": spm_bad_mask,
    })
    return ctx


# -------------------------------------------------------------
# STAGE 2 — Scene CR + SPM
# -------------------------------------------------------------
def stage2_scene_spm(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 2 — Continuum Removal & SPM on Full Scene")
    print("-" * 60)

    from scipy.ndimage import maximum_filter1d

    cube = ctx["cube_denoised"]
    # Prefer original TRR3 I/F for SPM (caution bands still present)
    cube_spm_src = ctx.get("cube_spm", cube)
    waves = ctx["waves"]
    bad_mask = ctx["bad_mask"]
    spm_bad_mask = ctx.get("spm_bad_mask", build_spm_bad_mask(waves))
    dead_bands = ctx["dead_bands"]
    valid = ctx["valid_pixel_mask"]
    H, W, B = ctx["H"], ctx["W"], ctx["B"]
    output_dir = ctx["output_dir"]

    # Mask FRT detector edge columns (saturated uncertainty / bad optics)
    n_before_edge = int(valid.sum())
    valid = mask_edge_columns(valid, EDGE_COL_MARGIN)
    ctx["valid_pixel_mask"] = valid
    print(f"  Edge-column mask (±{EDGE_COL_MARGIN} samples): "
          f"{n_before_edge} → {int(valid.sum())} valid pixels")

    # Recompute SPM band indices against the SPM-source data mask
    spm_bands = precompute_spm_bands(waves, spm_bad_mask)
    ctx["spm_bands"] = spm_bands

    # --- Bland-region spectral ratio (classic CRISM) then Viviano-Beck SPM ---
    t0 = time.time()
    print(f"  SPM/backup IMG: {ctx.get('spm_source', 'denoised')}")
    print(f"  Denoised cube : {ctx.get('cube_source')}")

    def _fill_finite_bands(src: np.ndarray, skip_mask: np.ndarray) -> np.ndarray:
        out = src.astype(np.float32, copy=True)
        for bi in range(B):
            if skip_mask[bi]:
                continue
            band = out[:, :, bi]
            m = np.isfinite(band) & valid
            miss = (~np.isfinite(band)) & valid
            if m.any() and miss.any():
                band = band.copy()
                band[miss] = np.nanmedian(band[m])
                out[:, :, bi] = band
        return out

    cube_spm_raw = _fill_finite_bands(cube_spm_src, spm_bad_mask)
    cube_s4_raw = _fill_finite_bands(cube, bad_mask)

    # Fill diagnostic holes in S4 from IMG so mineral ID can use denoised cube
    for w in (1.054, 1.330, 1.810, 1.930, 2.120, 2.210, 2.300):
        bi = int(np.argmin(np.abs(waves - w)))
        s4b = cube_s4_raw[:, :, bi]
        imgb = cube_spm_raw[:, :, bi]
        miss = valid & (~np.isfinite(s4b)) & np.isfinite(imgb)
        if miss.any():
            s4b = s4b.copy()
            s4b[miss] = imgb[miss]
            cube_s4_raw[:, :, bi] = s4b

    # Avoid double destripe on S4 (Stage 4 already destriped stripe bands)
    if str(ctx.get("cube_source", "")).startswith("S4"):
        print("  Column destripe: skipped on S4 (already Stage-4 destriped)")
        cube_spm_raw, resid_spm = column_destripe(cube_spm_raw, valid, spm_bad_mask)
        print(f"  Column destripe on IMG backup: mean|residual|={resid_spm:.5f}")
    else:
        cube_spm_raw, resid_spm = column_destripe(cube_spm_raw, valid, spm_bad_mask)
        cube_s4_raw, resid_s4 = column_destripe(cube_s4_raw, valid, bad_mask)
        print(f"  Column destripe: mean|residual| SPM={resid_spm:.5f}, S4={resid_s4:.5f}")

    # Bland on the same cube being ratioed
    bland_spm, n_bland_spm = estimate_bland_spectrum(
        cube_spm_raw, valid, spm_bad_mask, pct=BLAND_PCT
    )
    bland_s4, n_bland_s4 = estimate_bland_spectrum(
        cube_s4_raw, valid, bad_mask, pct=BLAND_PCT
    )
    miss_s4 = ~np.isfinite(bland_s4) & np.isfinite(bland_spm)
    bland_s4 = bland_s4.copy()
    bland_s4[miss_s4] = bland_spm[miss_s4]
    print(f"  Bland-region (IMG backup): n={n_bland_spm} | (S4): n={n_bland_s4} "
          f"(lowest-contrast {BLAND_PCT}%)")
    print(f"  Bland finite bands: IMG={int(np.isfinite(bland_spm).sum())}/ {B}, "
          f"S4={int(np.isfinite(bland_s4).sum())}/{B}")
    cube_spm = apply_bland_ratio(cube_spm_raw, bland_spm, valid)
    cube_ratio = apply_bland_ratio(cube_s4_raw, bland_s4, valid)
    safe_np_save(os.path.join(output_dir, "bland_spectrum.npy"), bland_spm)
    safe_np_save(os.path.join(output_dir, "bland_spectrum_s4.npy"), bland_s4)

    # Primary SPM on ratioed denoised S4; IMG fills gaps
    spm_map = compute_spm_map_vectorized(cube_ratio, waves, spm_bands, spm_bad_mask, valid)
    spm_map, spm_notes = sanitize_spm_map(spm_map, valid)
    print("  SPM primary source: bland-ratioed denoised S4")

    need_any = [n for n in SPM_NAMES if not spm_notes.get(n, "").startswith("ok")]
    if need_any:
        print(f"  SPM IMG fallback for: {need_any}")
        spm_img = compute_spm_map_vectorized(cube_spm, waves, spm_bands, spm_bad_mask, valid)
        spm_img_san, notes_img = sanitize_spm_map(spm_img, valid)
        for name in need_any:
            if notes_img.get(name, "").startswith("ok"):
                j = SPM_NAMES.index(name)
                spm_map[:, :, j] = spm_img_san[:, :, j]
                spm_notes[name] = notes_img[name] + "_ratio_img"
                print(f"    Recovered {name} from ratioed IMG ({spm_notes[name]})")

    need_mafic = [n for n in MAFIC_SPM_NAMES if not spm_notes.get(n, "").startswith("ok")]
    if "OLINDEX3" in need_mafic:
        ol3 = olindex3_threepoint_map(cube_ratio, waves, spm_bad_mask, valid)
        fin = np.isfinite(ol3) & valid
        if fin.sum() > 100:
            p90 = float(np.percentile(ol3[fin], 90))
            if p90 >= SPM_DETECT_THRESH["OLINDEX3"]:
                j = SPM_NAMES.index("OLINDEX3")
                spm_map[:, :, j] = np.clip(ol3, -0.5, 2.0)
                spm_notes["OLINDEX3"] = "ok_ratio_3pt_diagnostic"
                print(f"    Recovered OLINDEX3 via 3-pt on S4 ratio (p90={p90:.3f})")
            else:
                print(f"    OLINDEX3 3-pt still weak (p90={p90:.3f})")

    n_ok = sum(1 for v in spm_notes.values() if v.startswith("ok"))
    print(f"  SPM vectorized+sanitized in {time.time() - t0:.1f}s  "
          f"({n_ok}/{N_SPM} indices usable)")
    for k, v in spm_notes.items():
        status = "OK" if v.startswith("ok") else v
        print(f"    SPM {k}: {status}")

    # Fix B: scene-adaptive SPM-active domain (union of passing indices)
    # Ice stays out of clustering domain (ice_or_excluded).
    print("  Building adaptive SPM-active domain (Fix B)...")
    spm_active_mask, spm_gate_info, spm_passed = build_adaptive_spm_active(
        spm_map, valid, spm_notes=spm_notes,
        percentile=SPM_ADAPTIVE_PERCENTILE,
        min_component=SPM_MIN_COMPONENT,
    )
    # Polar / explicit: ice PRESENT gates without expanding clustering domain
    ice_on = ice_proof_enabled(ctx)
    ctx["ice_proof_enabled"] = bool(ice_on)
    ice_proof_gates: dict = {}
    ice_proof_candidates: List[str] = []
    if ice_on:
        print("  Ice-proof screening ENABLED (polar/Scene4 or CRISM_ALLOW_ICE_PROOF) "
              "— ice stays out of SPM-active clustering domain")
        ice_proof_gates, ice_proof_candidates = evaluate_ice_spm_gates_for_proof(
            spm_map, valid, spm_notes=spm_notes,
            percentile=SPM_ADAPTIVE_PERCENTILE,
            min_component=SPM_MIN_COMPONENT,
        )
        for iname, ig in ice_proof_gates.items():
            # Keep clustering gate record as excluded; attach proof-gate side channel
            base = spm_gate_info.get(iname) or {
                "passed": False, "reason": "ice_or_excluded",
            }
            base = dict(base)
            base["clustering_excluded"] = True
            base["ice_proof_gate"] = ig
            spm_gate_info[iname] = base
    else:
        print("  Ice-proof screening off (mid-latitude default; ice clustering-excluded)")
    ctx["ice_proof_gates"] = ice_proof_gates
    ctx["ice_proof_candidates"] = list(ice_proof_candidates)
    # Fallback: if nothing passed, use softer floor on max SPM (still not all pixels)
    if not spm_active_mask.any():
        print("  [WARN] No SPM index passed adaptive gate — "
              "fallback to max-SPM >= p90 domain (non-ice channels only)")
        spm_max = nanmax_spm_exclude_ice(spm_map, axis=2)
        fin = valid & np.isfinite(spm_max) & (spm_max > -1e8)
        if fin.any():
            thr = float(np.percentile(spm_max[fin], 90))
            spm_active_mask = fin & (spm_max >= thr)
        print(f"  Fallback SPM-active pixels: {int(spm_active_mask.sum())}")

    # Mineral bad-mask keeps diagnostic bands
    mineral_bad = ctx.get("mineral_bad_mask")
    if mineral_bad is None:
        mineral_bad = build_mineral_bad_mask(waves)
    s4_dead = build_data_dead_mask(cube_ratio, valid, min_frac=0.05)
    img_dead = build_data_dead_mask(cube_spm, valid, min_frac=0.05)
    mineral_bad = mineral_bad | (s4_dead & img_dead)
    mineral_good = ~mineral_bad
    ctx["mineral_bad_mask"] = mineral_bad
    ctx["mineral_good_mask"] = mineral_good

    # SAM on mineral-good bands (atm cores excluded; diagnostics kept)
    sam_mask = build_sam_band_mask(waves, mineral_good)
    diag_sam_mask = build_diagnostic_sam_mask(waves, mineral_good)
    ctx["sam_mask"] = sam_mask
    ctx["diag_sam_mask"] = diag_sam_mask
    print(f"  Mineral CR/SAM bands: {int(mineral_good.sum())} / {B} "
          f"(SAM after atm exclude: {int(sam_mask.sum())}; "
          f"diagnostic-window SAM: {int(diag_sam_mask.sum())})")

    # Continuum removal on bland-ratioed DENOISED S4 (same CR as library)
    cube_for_cr = cube_ratio
    print("  CR source: bland-ratioed denoised S4 (matching-consistent envelope)")
    t0 = time.time()
    cube_cr = continuum_removal_match_cube(
        cube_for_cr, valid, mineral_bad, size=CR_MATCH_FILTER_SIZE
    )
    print(f"  Continuum removal done in {time.time() - t0:.1f}s")

    # True saturated CO2 core only (not the old false 1.55 µm notch)
    t_notch = time.time()
    cube_cr = suppress_cr_artifact_notch(cube_cr, waves, valid, window=(2.650, 2.800))
    print(f"  Saturated-CO2 notch handling (2.65–2.80 µm) in {time.time() - t_notch:.1f}s")

    # Sanitize CR: kill CO2/fill outliers that blow unmixing MSE to ~1e13
    cube_cr, cr_stats = sanitize_cr_array(cube_cr, mineral_bad=mineral_bad)
    print(
        f"  CR sanitize: clipped={cr_stats['n_clipped']} "
        f"max|CR| before={cr_stats['max_abs_before']:.3g} "
        f"p99 after={cr_stats['p99_abs_after']}"
    )
    ctx["cr_sanitize_stats"] = cr_stats

    # Vectorized L2
    flat = cube_cr.reshape(-1, B)
    norms = np.linalg.norm(np.nan_to_num(flat, nan=0.0), axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    flat_l2 = np.nan_to_num(flat, nan=0.0) / norms
    flat_l2[:, mineral_bad] = np.nan
    cube_cr_l2 = flat_l2.reshape(H, W, B).astype(np.float32)
    cube_cr_l2[~valid] = np.nan

    # Training: SPM-active pixels only (Fix D domain for encoder)
    valid_coords = np.argwhere(valid)
    n_valid = len(valid_coords)
    spm_flat = spm_map[valid]
    active_flat = spm_active_mask[valid]
    active_idx = np.where(active_flat)[0]
    selected = set()
    # Stratify within active domain by each passing SPM
    for j, name in enumerate(SPM_NAMES):
        if name not in spm_passed:
            continue
        col = spm_flat[:, j]
        finite = np.isfinite(col) & active_flat
        if finite.sum() == 0:
            continue
        thr = spm_gate_info[name]["thr_adaptive"]
        top = np.where(finite & (col >= thr))[0]
        n_sample = min(800, len(top))
        if n_sample > 0:
            chosen = np.random.choice(top, n_sample, replace=False)
            selected.update(chosen.tolist())
    target_n = min(TARGET_TRAINING_SAMPLES, max(len(active_idx), 1))
    remaining = target_n - len(selected)
    if remaining > 0 and len(active_idx):
        pool = np.setdiff1d(active_idx, np.array(list(selected), dtype=int), assume_unique=False)
        if len(pool):
            selected.update(
                np.random.choice(pool, min(remaining, len(pool)), replace=False).tolist()
            )
    if not selected:
        # Absolute fallback: random valid
        selected = set(
            np.random.choice(n_valid, min(target_n, n_valid), replace=False).tolist()
        )
    selected_indices = sorted(selected)
    training_coords = valid_coords[selected_indices]
    print(f"  Training pixels: {len(training_coords)} "
          f"(SPM-active domain={int(active_flat.sum())}/{n_valid})")

    # Diagnostic: finite fraction at key mineral wavelengths
    for w in (1.054, 1.330, 1.810, 1.930, 2.120, 2.210, 2.300):
        bi = int(np.argmin(np.abs(waves - w)))
        frac = float(np.mean(np.isfinite(cube_cr[:, :, bi][valid])))
        print(f"    CR finite @{w:.3f} µm (band {bi}): {100 * frac:.1f}%")

    # Large arrays: use Windows-safe save (avoids Errno 22 on locked/mmap'd files)
    safe_np_save(os.path.join(output_dir, "cube_cr.npy"), cube_cr)
    safe_np_save(os.path.join(output_dir, "cube_cr_l2.npy"), cube_cr_l2)
    safe_np_save(os.path.join(output_dir, "spm_map.npy"), spm_map)
    safe_np_save(os.path.join(output_dir, "spm_active_mask.npy"),
                 spm_active_mask.astype(np.uint8))
    safe_np_save(os.path.join(output_dir, "training_coords.npy"), training_coords)
    with open(os.path.join(output_dir, "spm_sanity.json"), "w") as f:
        json.dump(spm_notes, f, indent=2)
    with open(os.path.join(output_dir, "spm_gate_info.json"), "w") as f:
        json.dump({"passed": spm_passed, "gates": spm_gate_info}, f, indent=2)

    print(f"  SPM map shape: {spm_map.shape}")
    print(f"  Training pixels: {len(training_coords)} / {n_valid} valid "
          f"(active domain {int(spm_active_mask.sum())})")
    for j, name in enumerate(SPM_NAMES):
        col = spm_map[..., j]
        finite = col[np.isfinite(col)]
        if finite.size:
            print(f"    {name:10s}: mean={finite.mean():.4f}  p90={np.percentile(finite, 90):.4f}")
        else:
            print(f"    {name:10s}: ALL NAN ({spm_notes.get(name, '?')})")

    ctx.update({
        "cube_cr": cube_cr,
        "cube_cr_l2": cube_cr_l2,
        "cube_ratio": cube_ratio,
        "cube_spm_ratio": cube_spm,
        "bland_spectrum": bland_spm,
        "bland_spectrum_s4": bland_s4,
        "spm_map": spm_map,
        "spm_notes": spm_notes,
        "spm_active_mask": spm_active_mask,
        "spm_gate_info": spm_gate_info,
        "spm_passed": spm_passed,
        "training_coords": training_coords,
        "valid_coords_all": valid_coords,
        "used_bland_ratio": True,
        "cr_source": "bland_ratioed_S4",
    })
    return ctx


# -------------------------------------------------------------
# STAGE 3 — Contrastive Encoder
# -------------------------------------------------------------
def stage3_encoder(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 3 — Contrastive Spectral Encoder Training")
    print("-" * 60)

    device = _device()
    waves = ctx["waves"]
    cube_cr = ctx["cube_cr"]
    cube_cr_l2 = ctx["cube_cr_l2"]
    spm_map = ctx["spm_map"]
    training_coords = ctx["training_coords"]
    lib_l2 = ctx["lib_spectra_cr_l2"]
    lib_cr = ctx["lib_spectra_cr"]
    lib_spm = ctx["lib_spm"]
    valid = ctx["valid_pixel_mask"]
    output_dir = ctx["output_dir"]
    B = ctx["B"]

    # Protected diagnostic bands
    protected = set()
    for wl in SPM_DIAGNOSTIC_WLS:
        bi = int(np.argmin(np.abs(waves - wl)))
        for offset in range(-3, 4):
            if 0 <= bi + offset < B:
                protected.add(bi + offset)

    # Build training pool (L2 for encoder; CR for unmixing targets)
    scene_l2 = cube_cr_l2[training_coords[:, 0], training_coords[:, 1], :]
    scene_cr = cube_cr[training_coords[:, 0], training_coords[:, 1], :]
    scene_spm = spm_map[training_coords[:, 0], training_coords[:, 1], :]

    n_lib = min(LIB_SAMPLE_N, max(1, len(lib_l2) * 3))
    lib_idx = np.random.choice(len(lib_l2), n_lib, replace=True)
    lib_sample_l2 = lib_l2[lib_idx]
    lib_sample_cr = lib_cr[lib_idx]
    lib_spm_sample = lib_spm[lib_idx]

    synth_l2, synth_cr, synth_spm = [], [], []
    n_lib_avail = len(lib_l2)
    n_synth = SYNTH_MIXES if n_lib_avail >= 2 else 0
    for _ in range(n_synth):
        i1, i2 = np.random.choice(n_lib_avail, 2, replace=False)
        alpha = np.random.uniform(0.1, 0.9)
        mixed_l2 = l2_normalize(alpha * lib_l2[i1] + (1 - alpha) * lib_l2[i2])
        mixed_cr = alpha * lib_cr[i1] + (1 - alpha) * lib_cr[i2]
        mixed_spm = alpha * lib_spm[i1] + (1 - alpha) * lib_spm[i2]
        synth_l2.append(mixed_l2)
        synth_cr.append(mixed_cr.astype(np.float32))
        synth_spm.append(mixed_spm.astype(np.float32))

    parts_l2 = [scene_l2, lib_sample_l2]
    parts_cr = [scene_cr, lib_sample_cr]
    parts_spm = [scene_spm, lib_spm_sample]
    if synth_l2:
        parts_l2.append(np.asarray(synth_l2, dtype=np.float32))
        parts_cr.append(np.asarray(synth_cr, dtype=np.float32))
        parts_spm.append(np.asarray(synth_spm, dtype=np.float32))

    pool_spectra = np.nan_to_num(np.concatenate(parts_l2, axis=0), nan=0.0).astype(np.float32)
    pool_spectra_cr = np.concatenate(parts_cr, axis=0).astype(np.float32)
    pool_spectra_cr, pool_cr_stats = sanitize_cr_array(pool_spectra_cr)
    pool_spectra_cr = np.nan_to_num(pool_spectra_cr, nan=0.0).astype(np.float32)
    # Clip residual spikes for stable unmix MSE (targets stay in [0, 2])
    pool_spectra_cr = np.clip(pool_spectra_cr, CR_SANITIZE_LO, CR_SANITIZE_HI)
    pool_spm = np.nan_to_num(np.concatenate(parts_spm, axis=0), nan=0.0).astype(np.float32)
    # Clamp SPM targets into [0, 1] for sigmoid head
    pool_spm = np.clip(pool_spm, 0.0, 1.0)

    n_pool = len(pool_spectra)
    print(f"  Training pool: {n_pool} spectra "
          f"(scene={len(scene_l2)}, lib={n_lib}, synth={n_synth})")
    print(f"  Pool CR sanitize: clipped={pool_cr_stats['n_clipped']} "
          f"p99|CR|={pool_cr_stats['p99_abs_after']}")
    print(f"  Device: {device} | batch={ENCODER_BATCH} | max_epochs={ENCODER_EPOCHS} "
          f"| patience={ENCODER_PATIENCE} | mode={TRAIN_MODE} "
          f"| latent={LATENT_DIM} (+recon head)")

    encoder = SpectralEncoder(n_bands=B, latent_dim=LATENT_DIM, n_spm=N_SPM).to(device)
    prior_loaded = False
    prior_encoder_path = None
    if TRAIN_MODE == "adapt" and SHARED_PRIOR_DIR:
        prior_enc = os.path.join(SHARED_PRIOR_DIR, "encoder_B438.pth")
        prior_encoder_path = prior_enc
        if os.path.isfile(prior_enc):
            try:
                encoder.load_state_dict(
                    torch.load(prior_enc, map_location=device, weights_only=True)
                )
                prior_loaded = True
                print(f"  Warm-start encoder from {prior_enc}")
            except Exception as e:
                print(f"  [WARN] encoder prior load failed ({type(e).__name__}: {e})")
        else:
            print(f"  [WARN] adapt mode but missing encoder prior: {prior_enc}")
    ctx["prior_dir"] = SHARED_PRIOR_DIR
    ctx["prior_encoder_path"] = prior_encoder_path
    ctx["prior_loaded"] = bool(prior_loaded)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ENCODER_EPOCHS)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Hold out validation indices for early-stop (same-scene, no leakage across FRTs)
    idx_all = np.arange(n_pool)
    np.random.shuffle(idx_all)
    n_val = max(32, int(0.10 * n_pool)) if n_pool >= 64 else max(1, n_pool // 10)
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:] if n_pool - n_val >= 4 else idx_all
    print(f"  Encoder split: train={len(train_idx)} val={len(val_idx)}")

    def _encoder_batch_loss(batch_idx, train=True):
        specs = pool_spectra[batch_idx]
        spm_b = pool_spm[batch_idx]
        v1 = np.stack([augment_spectrum(s, protected) for s in specs], axis=0)
        v2 = np.stack([augment_spectrum(s, protected) for s in specs], axis=0)
        t1 = torch.from_numpy(v1).float().unsqueeze(1).to(device)
        t2 = torch.from_numpy(v2).float().unsqueeze(1).to(device)
        spm_t = torch.from_numpy(spm_b).float().to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            z1 = encoder(t1)
            z2 = encoder(t2)
            loss_c = nt_xent_loss(z1, z2, temperature=0.07)
            spm_pred = encoder.predict_spm(z1)
            loss_s = F.mse_loss(spm_pred, spm_t)
            recon = encoder.reconstruct(z1)
            target = torch.from_numpy(specs).float().to(device)
            prot = sorted(protected) if protected else list(range(0, B, 8))
            prot_t = torch.tensor(prot, device=device, dtype=torch.long)
            loss_r = F.mse_loss(recon.index_select(1, prot_t),
                                target.index_select(1, prot_t))
            loss = loss_c + 0.1 * loss_s + 0.2 * loss_r
        return loss

    encoder.train()
    history = []
    val_history = []
    best_val = float("inf")
    patience_ctr = 0
    best_path = os.path.join(output_dir, "encoder_B438.pth")
    min_epochs = (
        ENCODER_MIN_EPOCHS_FULL if TRAIN_MODE == "full" else ENCODER_MIN_EPOCHS_ADAPT
    )
    print(f"  Early-stop: patience={ENCODER_PATIENCE}, min_epochs={min_epochs}")
    t0 = time.time()
    last_epoch = -1
    for epoch in range(ENCODER_EPOCHS):
        last_epoch = epoch
        np.random.shuffle(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        encoder.train()
        for start in range(0, len(train_idx), ENCODER_BATCH):
            batch_idx = train_idx[start:start + ENCODER_BATCH]
            if len(batch_idx) < 4:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = _encoder_batch_loss(batch_idx, train=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        scheduler.step()
        mean_loss = epoch_loss / max(1, n_batches)
        history.append(mean_loss)

        # Validation
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(val_idx), ENCODER_BATCH):
                batch_idx = val_idx[start:start + ENCODER_BATCH]
                if len(batch_idx) < 2:
                    continue
                vl = _encoder_batch_loss(batch_idx, train=False)
                val_losses.append(float(vl.detach().cpu()))
        mean_val = float(np.mean(val_losses)) if val_losses else mean_loss
        val_history.append(mean_val)

        if mean_val < best_val - 1e-5:
            best_val = mean_val
            patience_ctr = 0
            torch.save(encoder.state_dict(), best_path)
        elif epoch + 1 >= min_epochs:
            patience_ctr += 1
        # Before min_epochs: do not accumulate early-stop patience

        if epoch % 10 == 0 or epoch == ENCODER_EPOCHS - 1 or patience_ctr == 0:
            print(f"  Epoch {epoch:03d}/{ENCODER_EPOCHS - 1}  "
                  f"train={mean_loss:.4f}  val={mean_val:.4f}  "
                  f"best_val={best_val:.4f}")

        if patience_ctr >= ENCODER_PATIENCE:
            print(f"  Encoder early stop at epoch {epoch} "
                  f"(patience={ENCODER_PATIENCE}, min_epochs={min_epochs}, "
                  f"best_val={best_val:.4f})")
            break

    print(f"  Encoder training finished in {time.time() - t0:.1f}s "
          f"({last_epoch + 1} epochs)")
    if os.path.isfile(best_path):
        encoder.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    else:
        torch.save(encoder.state_dict(), best_path)
    _clear_cuda()
    # Full-scene encoding
    encoder.eval()
    valid_coords_all = ctx["valid_coords_all"]
    n_valid = len(valid_coords_all)
    latent_vectors = np.zeros((n_valid, LATENT_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_valid, BATCH_INF):
            end = min(start + BATCH_INF, n_valid)
            coords = valid_coords_all[start:end]
            specs = np.nan_to_num(
                cube_cr_l2[coords[:, 0], coords[:, 1], :], nan=0.0
            ).astype(np.float32)
            t = torch.from_numpy(specs).float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = encoder(t)
            latent_vectors[start:end] = z.float().cpu().numpy()

    norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    latent_vectors = (latent_vectors / norms).astype(np.float32)

    safe_np_save(os.path.join(output_dir, "latent_vectors.npy"), latent_vectors)
    safe_np_save(os.path.join(output_dir, "latent_coords.npy"), valid_coords_all)

    print(f"  Encoded {n_valid} pixels -> latent ({LATENT_DIM},)")

    ctx.update({
        "encoder": encoder,
        "pool_spectra": pool_spectra,
        "pool_spectra_cr": pool_spectra_cr,
        "pool_spm": pool_spm,
        "latent_vectors": latent_vectors,
        "encoder_history": history,
        "encoder_val_history": val_history,
        "encoder_best_val": best_val,
        "encoder_epochs_ran": last_epoch + 1,
        "device": device,
        "train_mode": TRAIN_MODE,
        "prior_dir": ctx.get("prior_dir", SHARED_PRIOR_DIR),
        "prior_encoder_path": ctx.get("prior_encoder_path"),
        "prior_loaded": bool(ctx.get("prior_loaded", False)),
    })
    return ctx

# -------------------------------------------------------------
# STAGE 4 — HDBSCAN
# -------------------------------------------------------------
def stage4_hdbscan(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 4 — HDBSCAN Density-Based Clustering")
    print("-" * 60)

    # Stabilize HDBSCAN/numba on Windows (parallel parfor can hang / raise
    # AttributeError deep in llvmlite during first JIT compile).
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    try:
        from numba import config as _numba_config  # type: ignore
        _numba_config.NUMBA_NUM_THREADS = 1
        _numba_config.THREADING_LAYER = "workqueue"
    except Exception:
        pass

    import hdbscan

    latent = ctx["latent_vectors"]
    valid_coords_all = ctx["valid_coords_all"]
    spm_map = ctx["spm_map"]
    cube_cr = ctx["cube_cr"]
    lib_cr = ctx["lib_spectra_cr"]
    lib_names = ctx["lib_names"]
    H, W = ctx["H"], ctx["W"]
    output_dir = ctx["output_dir"]

    n_valid = len(latent)
    spm_flat = ctx["spm_map"][ctx["valid_pixel_mask"]].astype(np.float32)
    # Discovery-first fusion: encoder latent is backbone; diagnostic CR aids
    # mineral absorptions; SPM is soft continuous texture only (NOT hard domains).
    spm_feat = _zscore_cols(spm_flat.astype(np.float64))
    spm_w = np.ones(N_SPM, dtype=np.float64)
    for _nm, _w in SPM_CLUSTER_WEIGHT.items():
        if _nm in SPM_NAMES:
            spm_w[SPM_NAMES.index(_nm)] = float(_w)
    spm_feat = spm_feat * spm_w[None, :]
    waves = ctx["waves"]
    mineral_bad = ctx.get("mineral_bad_mask")
    if mineral_bad is None:
        mineral_bad = build_mineral_bad_mask(waves)
    diag_wls = (1.054, 1.330, 1.750, 1.930, 2.120, 2.210, 2.260, 2.290, 2.400, 2.300)
    diag_bis = [int(np.argmin(np.abs(waves - wl))) for wl in diag_wls]
    valid_mask = ctx["valid_pixel_mask"]
    diag_spec = cube_cr[valid_mask][:, diag_bis].astype(np.float64)
    diag_feat = _zscore_cols(np.nan_to_num(diag_spec, nan=0.0))
    # Continuum band depths (same defs as proof; ice excluded) — mineral mode axes
    depth_raw, depth_keys = compute_cluster_depth_features(
        cube_cr[valid_mask], waves, bad_mask=mineral_bad,
    )
    depth_feat = _zscore_cols(np.nan_to_num(depth_raw, nan=0.0))
    depth_feat, muted_depth = mute_flat_depth_features(
        depth_raw, depth_feat, keys=depth_keys,
    )
    depth_w = (
        float(CLUSTER_DEPTH_WEIGHT_ADAPT)
        if TRAIN_MODE == "adapt"
        else float(CLUSTER_DEPTH_WEIGHT)
    )
    ctx["cluster_depth_keys"] = list(depth_keys)
    ctx["cluster_depth_weight_used"] = float(depth_w)
    ctx["cluster_depth_muted"] = list(muted_depth)
    xy = valid_coords_all.astype(np.float64)
    xy = (xy - np.nanmean(xy, axis=0)) / (np.nanstd(xy, axis=0) + 1e-8)
    n_depth_finite = int(np.isfinite(depth_raw).sum()) if depth_raw.size else 0
    use_umap = os.environ.get("CRISM_USE_UMAP", "0").strip().lower() in (
        "1", "true", "yes",
    )

    def _assemble_cluster_features(include_depth: bool) -> np.ndarray:
        """Latent + soft SPM/diag (+ optional depths) + xy, then PCA."""
        blocks = [
            latent.astype(np.float64),
            CLUSTER_SPM_WEIGHT * spm_feat,
            CLUSTER_DIAG_WEIGHT * diag_feat,
        ]
        if include_depth and depth_feat.shape[1] > 0:
            blocks.append(depth_w * depth_feat)
        cf = np.concatenate(blocks, axis=1).astype(np.float64)
        cf = np.concatenate(
            [cf, CLUSTER_XY_WEIGHT * xy], axis=1
        ).astype(np.float64)
        try:
            from sklearn.decomposition import PCA
            n_comp = int(min(16, cf.shape[1], max(8, n_valid // 5000)))
            pca = PCA(n_components=n_comp, random_state=42)
            pca_n = min(30000, n_valid)
            pca_idx = np.linspace(0, n_valid - 1, pca_n, dtype=int)
            pca.fit(cf[pca_idx])
            cf = pca.transform(cf).astype(np.float64)
            print(
                f"  PCA for clustering: {n_comp} comps "
                f"(var={float(pca.explained_variance_ratio_.sum()):.3f}; "
                f"depth={'on' if include_depth else 'off'})"
            )
        except Exception as e:
            print(
                f"  PCA skipped ({type(e).__name__}: {e}); "
                f"using full latent+diag"
                f"{'+depth' if include_depth else ''}"
            )
        return cf

    def _maybe_umap(cf: np.ndarray) -> np.ndarray:
        if not (use_umap and n_valid >= 2000):
            print(
                "  UMAP disabled (set CRISM_USE_UMAP=1 to enable); "
                "clustering on latent+SPM"
            )
            return cf
        try:
            from umap import UMAP  # type: ignore
            print("  UMAP embedding of latent+SPM before HDBSCAN (n_components=8)")
            reducer = UMAP(
                n_components=8, n_neighbors=30, min_dist=0.0,
                metric="euclidean", random_state=42, verbose=False,
            )
            umap_fit_n = min(15000, n_valid)
            umap_idx = np.linspace(0, n_valid - 1, umap_fit_n, dtype=int)
            reducer.fit(cf[umap_idx])
            chunks = []
            for start in range(0, n_valid, 50000):
                end = min(start + 50000, n_valid)
                chunks.append(reducer.transform(cf[start:end]))
            return np.concatenate(chunks, axis=0).astype(np.float64)
        except Exception as e:
            print(f"  UMAP skipped ({type(e).__name__}: {e}); clustering on latent+SPM")
            return cf

    use_depth_features = bool(depth_feat.shape[1] > 0 and depth_w > 0)
    print(
        f"  Discovery features: latent + diag×{CLUSTER_DIAG_WEIGHT} + "
        f"SPM×{CLUSTER_SPM_WEIGHT} (soft) + depth×{depth_w} "
        f"({len(depth_keys)} keys, ice excluded, mode={TRAIN_MODE}) "
        f"+ xy×{CLUSTER_XY_WEIGHT}"
    )
    if depth_keys:
        print(
            f"  Cluster depth keys: {', '.join(depth_keys)} "
            f"(finite cells={n_depth_finite}/{depth_raw.size})"
        )
    if muted_depth:
        print(f"  Muted flat depth channels: {', '.join(muted_depth)}")
    cluster_feat = _assemble_cluster_features(include_depth=use_depth_features)
    feat_for_cluster = _maybe_umap(cluster_feat)
    ctx["cluster_depth_used"] = bool(use_depth_features)

    # Fix D: HDBSCAN only on SPM-active pixels (K still discovered dynamically).
    # Bland continuum stays label -1 by construction — not forced into minerals.
    spm_active_full = ctx.get("spm_active_mask")
    if spm_active_full is not None:
        active_flat = spm_active_full[ctx["valid_pixel_mask"]].astype(bool)
    else:
        active_flat = np.ones(n_valid, dtype=bool)
        print("  [WARN] No spm_active_mask in ctx — clustering on all valid pixels")
    active_idx = np.where(active_flat)[0]
    if len(active_idx) < max(80, SPM_MIN_COMPONENT):
        print(f"  [WARN] SPM-active domain too small ({len(active_idx)}); "
              f"falling back to top-20% max-SPM pixels (non-ice)")
        spm_proxy = np.nan_to_num(
            nanmax_spm_exclude_ice(spm_flat, axis=1, fill=0.0), nan=0.0
        )
        thr = float(np.percentile(spm_proxy, 80))
        active_idx = np.where(spm_proxy >= thr)[0]
    n_domain = int(len(active_idx))
    feat_domain = np.ascontiguousarray(feat_for_cluster[active_idx], dtype=np.float64)
    print(f"  HDBSCAN domain (Fix D): {n_domain}/{n_valid} SPM-active pixels "
          f"({100.0 * n_domain / max(n_valid, 1):.1f}%); K still free")

    MAX_FIT = HDBSCAN_MAX_FIT
    # Stratified fit: latent PC1 quantiles + spatial grid (discovery, not SPM class)
    fit_local = np.arange(n_domain)
    if n_domain > MAX_FIT:
        lat_dom = latent[active_idx].astype(np.float64)
        try:
            from sklearn.decomposition import PCA as _PCAFit
            _pc = _PCAFit(n_components=1, random_state=42)
            lat_proxy = _pc.fit_transform(lat_dom).ravel()
        except Exception:
            lat_proxy = np.linalg.norm(lat_dom, axis=1)
        coords_dom = valid_coords_all[active_idx]
        n_q = 5
        try:
            quantiles = np.quantile(lat_proxy, np.linspace(0, 1, n_q + 1))
        except Exception:
            quantiles = np.linspace(float(lat_proxy.min()), float(lat_proxy.max()), n_q + 1)
        per_bin = max(1, MAX_FIT // (n_q * 4))
        chosen = []
        rmin, rmax = float(coords_dom[:, 0].min()), float(coords_dom[:, 0].max())
        cmin, cmax = float(coords_dom[:, 1].min()), float(coords_dom[:, 1].max())
        rmid = 0.5 * (rmin + rmax)
        cmid = 0.5 * (cmin + cmax)
        for qi in range(n_q):
            lo, hi = quantiles[qi], quantiles[qi + 1]
            if qi == n_q - 1:
                in_q = (lat_proxy >= lo) & (lat_proxy <= hi)
            else:
                in_q = (lat_proxy >= lo) & (lat_proxy < hi)
            q_idx = np.where(in_q)[0]
            if len(q_idx) == 0:
                continue
            for r_lo, r_hi, c_lo, c_hi in (
                (rmin, rmid, cmin, cmid),
                (rmin, rmid, cmid, cmax),
                (rmid, rmax, cmin, cmid),
                (rmid, rmax, cmid, cmax),
            ):
                cell = q_idx[
                    (coords_dom[q_idx, 0] >= r_lo) & (coords_dom[q_idx, 0] <= r_hi)
                    & (coords_dom[q_idx, 1] >= c_lo) & (coords_dom[q_idx, 1] <= c_hi)
                ]
                if len(cell) == 0:
                    continue
                take = min(per_bin, len(cell))
                if take < len(cell):
                    rng = np.random.default_rng(42 + qi)
                    cell = rng.choice(cell, size=take, replace=False)
                chosen.append(cell)
        if chosen:
            fit_local = np.unique(np.concatenate(chosen))
        if len(fit_local) < MAX_FIT:
            extra = np.setdiff1d(np.arange(n_domain), fit_local)
            need = MAX_FIT - len(fit_local)
            if len(extra) and need > 0:
                rng = np.random.default_rng(7)
                take = min(need, len(extra))
                fit_local = np.concatenate([
                    fit_local, rng.choice(extra, size=take, replace=False)
                ])
        fit_local = fit_local[:MAX_FIT]
        print(f"  Fitting HDBSCAN on latent+spatial stratified subsample "
              f"{len(fit_local)}/{n_domain}, "
              f"then approximate_predict within SPM-active domain", flush=True)
    n_fit = int(len(fit_local))
    def _hdbscan_params(scale: float = 1.0, method: str = "eom"):
        # MCS must match FIT-SET density scale (not full 200k domain).
        # Slightly finer default (0.008) for discovery of spectral modes.
        mcs = max(20, int(0.008 * n_fit * scale))
        mcs = min(mcs, max(20, n_fit // 18))
        ms = max(5, mcs // 12)
        return mcs, ms, method

    min_cluster_size, min_samples, sel_method = _hdbscan_params(0.75, "eom")
    print(f"  HDBSCAN params: min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, method={sel_method}, "
          f"feat_dim={feat_domain.shape[1]}, N_domain={n_domain}, N_fit={n_fit}",
          flush=True)

    def _make_clusterer(mcs, ms, method, algorithm: str = "best"):
        """Construct HDBSCAN with only kwargs supported by the installed version."""
        kwargs = dict(
            min_cluster_size=int(mcs),
            min_samples=int(ms),
            metric="euclidean",
            cluster_selection_method=method,
            prediction_data=True,
        )
        try:
            import inspect
            params = inspect.signature(hdbscan.HDBSCAN.__init__).parameters
            if "core_dist_n_jobs" in params:
                kwargs["core_dist_n_jobs"] = 1  # Windows-safe (no process pool)
            if "algorithm" in params and algorithm:
                kwargs["algorithm"] = algorithm
        except Exception:
            pass
        return hdbscan.HDBSCAN(**kwargs)

    def _fit_hdbscan_safe(mcs, ms, method, Xfit):
        """
        Fit HDBSCAN with fallbacks:
        1) generic algorithm (avoids numba boruvka parfor)
        2) best/default algorithm
        3) sklearn.cluster.HDBSCAN if available
        """
        errors = []
        for algo in ("generic", "best"):
            try:
                print(f"  HDBSCAN fit attempt algorithm={algo} on {len(Xfit)} samples...",
                      flush=True)
                clusterer = _make_clusterer(mcs, ms, method, algorithm=algo)
                clusterer.fit(Xfit)
                if getattr(clusterer, "labels_", None) is not None:
                    return clusterer, "hdbscan"
            except Exception as e:
                errors.append(f"{algo}: {type(e).__name__}: {e}")
                print(f"  HDBSCAN algorithm={algo} failed ({type(e).__name__}: {e})",
                      flush=True)

        # sklearn fallback (no hdbscan.approximate_predict — labels on fit set only)
        try:
            from sklearn.cluster import HDBSCAN as SkHDBSCAN
            print("  Falling back to sklearn.cluster.HDBSCAN", flush=True)
            clusterer = SkHDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(ms),
                metric="euclidean",
                cluster_selection_method=method,
                n_jobs=1,
            )
            clusterer.fit(Xfit)
            return clusterer, "sklearn"
        except Exception as e:
            errors.append(f"sklearn: {type(e).__name__}: {e}")
            raise RuntimeError(
                "All HDBSCAN backends failed:\n  - " + "\n  - ".join(errors)
            ) from e

    def _fit_predict(mcs, ms, method, feat):
        """Fit on SPM-active domain features; return labels length = n_domain."""
        Xfit = np.ascontiguousarray(feat[fit_local], dtype=np.float64)
        clusterer, backend = _fit_hdbscan_safe(mcs, ms, method, Xfit)
        labels_attr = getattr(clusterer, "labels_", None)
        probs_attr = getattr(clusterer, "probabilities_", None)
        if labels_attr is None:
            raise RuntimeError("HDBSCAN fit produced no labels_ attribute")
        if n_domain > MAX_FIT:
            labels_ = np.full(n_domain, -1, dtype=np.int32)
            probs_ = np.zeros(n_domain, dtype=np.float32)
            labels_[fit_local] = np.asarray(labels_attr, dtype=np.int32)
            if probs_attr is not None:
                probs_[fit_local] = np.asarray(probs_attr, dtype=np.float32)
            if backend == "hdbscan":
                try:
                    if getattr(clusterer, "prediction_data_", None) is None:
                        if hasattr(clusterer, "generate_prediction_data"):
                            clusterer.generate_prediction_data()
                    rem = np.setdiff1d(np.arange(n_domain), fit_local)
                    for start in range(0, len(rem), 20000):
                        batch = rem[start:start + 20000]
                        Xb = np.ascontiguousarray(feat[batch], dtype=np.float64)
                        lab, prob = hdbscan.approximate_predict(clusterer, Xb)
                        labels_[batch] = np.asarray(lab, dtype=np.int32)
                        probs_[batch] = np.asarray(prob, dtype=np.float32)
                except Exception as e:
                    print(f"  approximate_predict failed ({type(e).__name__}: {e}); "
                          f"keeping subsample labels only", flush=True)
            else:
                print("  sklearn backend: approximate_predict unavailable; "
                      "non-fit domain pixels remain noise (-1)", flush=True)
        else:
            labels_ = np.asarray(labels_attr, dtype=np.int32)
            probs_ = (
                np.asarray(probs_attr, dtype=np.float32)
                if probs_attr is not None
                else np.ones(n_domain, dtype=np.float32)
            )
        return clusterer, labels_, probs_

    def _expand_to_scene(labels_dom, probs_dom):
        """Map domain labels → full valid-pixel arrays (bland stays -1)."""
        labels_full = np.full(n_valid, -1, dtype=np.int32)
        probs_full = np.zeros(n_valid, dtype=np.float32)
        labels_full[active_idx] = labels_dom
        probs_full[active_idx] = probs_dom
        return labels_full, probs_full

    clusterer, labels_dom, probs_dom = _fit_predict(
        min_cluster_size, min_samples, sel_method, feat_domain
    )
    n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
    n_noise_dom = int((labels_dom == -1).sum())
    noise_frac_dom = n_noise_dom / max(n_domain, 1)

    # EOM retry if zero clusters OR excessive noise *within active domain*
    if n_clusters < 1 or noise_frac_dom > HDBSCAN_NOISE_RETRY:
        reason = "0 clusters" if n_clusters < 1 else f"domain_noise={100 * noise_frac_dom:.1f}%"
        print(f"  HDBSCAN retry ({reason}) — smaller min_cluster_size (still EOM)")
        mcs2, ms2, method2 = _hdbscan_params(0.55, "eom")
        clusterer, labels_dom, probs_dom = _fit_predict(
            mcs2, ms2, method2, feat_domain
        )
        n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
        n_noise_dom = int((labels_dom == -1).sum())
        noise_frac_dom = n_noise_dom / max(n_domain, 1)
        min_cluster_size, min_samples, sel_method = mcs2, ms2, method2

    # Leaf fallback when EOM still finds nothing on the fit subsample
    if n_clusters < 1:
        print("  HDBSCAN leaf fallback (EOM found 0 clusters on fit set)")
        mcs3, ms3, method3 = _hdbscan_params(0.40, "leaf")
        clusterer, labels_dom, probs_dom = _fit_predict(
            mcs3, ms3, method3, feat_domain
        )
        n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
        n_noise_dom = int((labels_dom == -1).sum())
        noise_frac_dom = n_noise_dom / max(n_domain, 1)
        min_cluster_size, min_samples, sel_method = mcs3, ms3, method3

    # Discovery: if EOM yields few mega-modes OR high domain noise, try leaf
    cov_eom = float((labels_dom >= 0).sum()) / max(n_domain, 1)
    if (n_clusters <= 2 or noise_frac_dom > 0.50) and n_domain >= 40000:
        print(
            f"  HDBSCAN leaf retry (n_clusters={n_clusters}, "
            f"domain_noise={100 * noise_frac_dom:.1f}%, "
            f"EOM coverage={100 * cov_eom:.1f}%)"
        )
        mcs_l, ms_l, _ = _hdbscan_params(0.55, "leaf")
        clusterer_l, labels_l, probs_l = _fit_predict(
            mcs_l, ms_l, "leaf", feat_domain
        )
        n_cl_l = int(labels_l.max() + 1) if labels_l.max() >= 0 else 0
        cov_l = float((labels_l >= 0).sum()) / max(n_domain, 1)
        # Keep leaf when it finds more modes; lower coverage bar (0.30) when
        # EOM itself is noisier / lower coverage than leaf.
        cov_keep = 0.30 if (noise_frac_dom > 0.50 or cov_l > cov_eom) else 0.40
        if n_cl_l > n_clusters and cov_l >= cov_keep:
            print(f"  Keeping leaf result: {n_cl_l} clusters "
                  f"(domain coverage {100 * cov_l:.1f}% >= {100 * cov_keep:.0f}% bar; "
                  f"EOM was {n_clusters} @ {100 * cov_eom:.1f}%)")
            clusterer, labels_dom, probs_dom = clusterer_l, labels_l, probs_l
            n_clusters = n_cl_l
            n_noise_dom = int((labels_dom == -1).sum())
            noise_frac_dom = n_noise_dom / max(n_domain, 1)
            min_cluster_size, min_samples, sel_method = mcs_l, ms_l, "leaf"
        else:
            print(f"  Leaf not kept ({n_cl_l} clusters, "
                  f"coverage {100 * cov_l:.1f}% < {100 * cov_keep:.0f}% bar "
                  f"or not more clusters than EOM); retaining EOM")

    # Safety: if depth fusion destroyed all density modes, retry once without
    # depth (keeps core methodology; only pays a second fit on collapse).
    if n_clusters < 1 and use_depth_features:
        print(
            "  [WARN] Depth fusion collapsed HDBSCAN to 0 clusters — "
            "fallback: rebuild features without depth and refit"
        )
        use_depth_features = False
        ctx["cluster_depth_used"] = False
        ctx["cluster_depth_fallback"] = "collapsed_zero_clusters"
        cluster_feat = _assemble_cluster_features(include_depth=False)
        feat_for_cluster = _maybe_umap(cluster_feat)
        feat_domain = np.ascontiguousarray(
            feat_for_cluster[active_idx], dtype=np.float64
        )
        min_cluster_size, min_samples, sel_method = _hdbscan_params(0.75, "eom")
        clusterer, labels_dom, probs_dom = _fit_predict(
            min_cluster_size, min_samples, sel_method, feat_domain
        )
        n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
        n_noise_dom = int((labels_dom == -1).sum())
        noise_frac_dom = n_noise_dom / max(n_domain, 1)
        if n_clusters < 1 or noise_frac_dom > HDBSCAN_NOISE_RETRY:
            mcs2, ms2, method2 = _hdbscan_params(0.55, "eom")
            clusterer, labels_dom, probs_dom = _fit_predict(
                mcs2, ms2, method2, feat_domain
            )
            n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
            n_noise_dom = int((labels_dom == -1).sum())
            noise_frac_dom = n_noise_dom / max(n_domain, 1)
            min_cluster_size, min_samples, sel_method = mcs2, ms2, method2
        if n_clusters < 1:
            mcs3, ms3, method3 = _hdbscan_params(0.40, "leaf")
            clusterer, labels_dom, probs_dom = _fit_predict(
                mcs3, ms3, method3, feat_domain
            )
            n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
            n_noise_dom = int((labels_dom == -1).sum())
            noise_frac_dom = n_noise_dom / max(n_domain, 1)
            min_cluster_size, min_samples, sel_method = mcs3, ms3, method3
        print(
            f"  Depth-fallback result: {n_clusters} clusters, "
            f"domain_noise={100 * noise_frac_dom:.1f}%"
        )

    # Adapt quality fallback: high domain noise with depth → trial without depth
    # (keeps depth if no-depth is not clearly quieter; protects S2-like wins).
    if (
        TRAIN_MODE == "adapt"
        and use_depth_features
        and n_clusters >= 1
        and noise_frac_dom >= ADAPT_DEPTH_DOMAIN_NOISE_RETRY
    ):
        print(
            f"  [adapt] domain_noise={100 * noise_frac_dom:.1f}% with depth "
            f">= {100 * ADAPT_DEPTH_DOMAIN_NOISE_RETRY:.0f}% — "
            f"trial without depth"
        )
        saved_dom = (
            labels_dom.copy(),
            probs_dom.copy(),
            int(n_clusters),
            float(noise_frac_dom),
            clusterer,
            int(min_cluster_size),
            int(min_samples),
            sel_method,
        )
        cluster_feat = _assemble_cluster_features(include_depth=False)
        feat_for_cluster = _maybe_umap(cluster_feat)
        feat_domain = np.ascontiguousarray(
            feat_for_cluster[active_idx], dtype=np.float64
        )
        min_cluster_size, min_samples, sel_method = _hdbscan_params(0.75, "eom")
        clusterer, labels_dom, probs_dom = _fit_predict(
            min_cluster_size, min_samples, sel_method, feat_domain
        )
        n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
        n_noise_dom = int((labels_dom == -1).sum())
        noise_frac_dom = n_noise_dom / max(n_domain, 1)
        if n_clusters < 1 or noise_frac_dom > HDBSCAN_NOISE_RETRY:
            mcs2, ms2, method2 = _hdbscan_params(0.55, "eom")
            clusterer, labels_dom, probs_dom = _fit_predict(
                mcs2, ms2, method2, feat_domain
            )
            n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
            n_noise_dom = int((labels_dom == -1).sum())
            noise_frac_dom = n_noise_dom / max(n_domain, 1)
            min_cluster_size, min_samples, sel_method = mcs2, ms2, method2
        if n_clusters < 1:
            mcs3, ms3, method3 = _hdbscan_params(0.40, "leaf")
            clusterer, labels_dom, probs_dom = _fit_predict(
                mcs3, ms3, method3, feat_domain
            )
            n_clusters = int(labels_dom.max() + 1) if labels_dom.max() >= 0 else 0
            n_noise_dom = int((labels_dom == -1).sum())
            noise_frac_dom = n_noise_dom / max(n_domain, 1)
            min_cluster_size, min_samples, sel_method = mcs3, ms3, method3
        depth_noise = float(saved_dom[3])
        trial_noise = float(noise_frac_dom)
        trial_ncl = int(n_clusters)
        if (
            trial_ncl >= 1
            and trial_noise <= depth_noise - ADAPT_DEPTH_NOISE_IMPROVE
        ):
            use_depth_features = False
            ctx["cluster_depth_used"] = False
            ctx["cluster_depth_fallback"] = "adapt_noise_regression"
            print(
                f"  Keeping no-depth: domain_noise "
                f"{100 * depth_noise:.1f}% → {100 * trial_noise:.1f}%"
            )
        else:
            (
                labels_dom, probs_dom, n_clusters, noise_frac_dom, clusterer,
                min_cluster_size, min_samples, sel_method,
            ) = saved_dom
            # Restore depth features for rescue / DBCV geometry
            cluster_feat = _assemble_cluster_features(include_depth=True)
            feat_for_cluster = _maybe_umap(cluster_feat)
            feat_domain = np.ascontiguousarray(
                feat_for_cluster[active_idx], dtype=np.float64
            )
            n_noise_dom = int((labels_dom == -1).sum())
            noise_frac_dom = float(depth_noise)
            print(
                f"  Keeping depth: no-depth noise "
                f"{100 * trial_noise:.1f}% (n={trial_ncl}) not clearly better "
                f"(need ≤ {100 * (depth_noise - ADAPT_DEPTH_NOISE_IMPROVE):.1f}%)"
            )

    labels, probabilities = _expand_to_scene(labels_dom, probs_dom)
    n_noise = int((labels == -1).sum())
    noise_frac = n_noise / max(n_valid, 1)
    print(f"  Domain clustered: {int((labels_dom >= 0).sum())}/{n_domain} "
          f"({100 * (1 - noise_frac_dom):.1f}% of SPM-active); "
          f"scene noise incl. bland: {100 * noise_frac:.1f}%")

    # Drop pushbroom stripe micro-units (narrow in sample, tall in line)
    stripe_dropped = 0
    for cl in range(n_clusters):
        m = labels == cl
        if m.sum() < 10:
            continue
        ys = valid_coords_all[m, 0]
        xs = valid_coords_all[m, 1]
        x_std = float(np.std(xs))
        y_range = int(ys.max() - ys.min()) if len(ys) else 0
        if x_std <= STRIPE_X_STD_MAX and y_range >= STRIPE_Y_RANGE_MIN:
            labels[m] = -1
            stripe_dropped += 1
    if stripe_dropped:
        print(f"  Dropped {stripe_dropped} stripe-like clusters "
              f"(x_std≤{STRIPE_X_STD_MAX}, y_range≥{STRIPE_Y_RANGE_MIN})")

    # Relabel contiguous ids after stripe drop
    old_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
    remap = {old: new for new, old in enumerate(old_ids)}
    if remap:
        new_labels = np.full_like(labels, -1)
        for old, new in remap.items():
            new_labels[labels == old] = new
        labels = new_labels
    n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    noise_frac = n_noise / max(n_valid, 1)

    # Mega-cluster split (primary only): BD1900 vs SINDEX etc. before rescue
    gate_info_early = ctx.get("spm_gate_info") or {}
    mega_diag: List[dict] = []
    labels, probabilities, n_mega_primary = split_mega_clusters_by_competing_spm(
        labels, probabilities, spm_flat, gate_info_early, active_idx, n_domain,
        mega_frac=MEGA_CLUSTER_FRAC,
        max_rounds=MEGA_SPLIT_MAX_ROUNDS,
        compete_pairs=SPM_COMPETE_PAIRS,
        label="Mega-split",
        diagnostics=mega_diag,
    )
    n_secondary = 0
    if n_mega_primary:
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        n_noise = int((labels == -1).sum())
        noise_frac = n_noise / max(n_valid, 1)
        print(f"  After primary mega-SPM split ({n_mega_primary}): "
              f"{n_clusters} clusters, {n_noise} noise ({100 * noise_frac:.1f}%)")
    else:
        print("  Mega-SPM split (primary): none applied (see skip logs)")

    # Rescue: only within SPM-active noise → nearest centroid (still no bland clustering)
    if n_clusters >= 1:
        active_noise = (labels < 0).copy()
        active_noise[np.setdiff1d(np.arange(n_valid), active_idx)] = False
        n_active_noise = int(active_noise.sum())
        domain_noise_frac = n_active_noise / max(n_domain, 1)
        if active_noise.any():
            centroids = []
            radii = []
            for cl in range(n_clusters):
                m = labels == cl
                if m.sum() < 5:
                    centroids.append(None)
                    radii.append(np.inf)
                    continue
                cxy = feat_for_cluster[m].mean(axis=0)
                d = np.linalg.norm(feat_for_cluster[m] - cxy, axis=1)
                centroids.append(cxy)
                radii.append(float(np.percentile(d, 90)) if len(d) else np.inf)
            # High domain noise → looser radius + more candidates (v5 had ~77% scene noise)
            if domain_noise_frac > 0.50:
                radius_mult = 1.85
                max_rescue = min(80000, n_active_noise)
                print(f"  Strengthened noise rescue: domain_noise="
                      f"{100 * domain_noise_frac:.1f}% → radius×{radius_mult}, "
                      f"cap={max_rescue}")
            else:
                radius_mult = 1.35
                max_rescue = min(40000, n_active_noise)
            rescued = 0
            noise_idx = np.where(active_noise)[0]
            if len(noise_idx) > max_rescue:
                spm_proxy = np.nan_to_num(
                    nanmax_spm_exclude_ice(spm_flat[noise_idx], axis=1, fill=0.0),
                    nan=0.0,
                )
                order = np.argsort(spm_proxy)[::-1][:max_rescue]
                noise_idx = noise_idx[order]
            for i in noise_idx:
                xi = feat_for_cluster[i]
                best_cl, best_d = -1, np.inf
                for cl, cxy in enumerate(centroids):
                    if cxy is None:
                        continue
                    d = float(np.linalg.norm(xi - cxy))
                    if d < best_d:
                        best_d, best_cl = d, cl
                if best_cl >= 0 and best_d <= radius_mult * max(radii[best_cl], 1e-6):
                    labels[i] = best_cl
                    probabilities[i] = max(
                        float(probabilities[i]),
                        0.30 if domain_noise_frac > 0.50 else 0.35,
                    )
                    rescued += 1
            if rescued:
                n_noise = int((labels == -1).sum())
                noise_frac = n_noise / max(n_valid, 1)
                print(f"  SPM-active noise rescue → {rescued} pixels assigned "
                      f"(scene noise now {100 * noise_frac:.1f}%; "
                      f"SPM-active coverage "
                      f"{100.0 * int((labels >= 0).sum()) / max(n_domain, 1):.1f}%)")
            else:
                print(f"  SPM-active noise rescue: 0 assigned "
                      f"(domain_noise={100 * domain_noise_frac:.1f}%, "
                      f"radius×{radius_mult})")

    # Clay-secondary AFTER rescue so grown hydrate units can clear size gate
    # (Scene2 adapt: pre-rescue cl0~1.6k skipped; post-rescue ~34k clears thr)
    if n_clusters >= 1:
        labels, probabilities, n_secondary = split_mega_clusters_by_competing_spm(
            labels, probabilities, spm_flat, gate_info_early, active_idx, n_domain,
            mega_frac=MEGA_SECONDARY_FRAC,
            max_rounds=1,
            compete_pairs=MEGA_SECONDARY_PAIRS,
            label="Clay-secondary-split",
            allow_relative_hotspot_split=True,
            diagnostics=mega_diag,
        )
        if n_secondary:
            n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
            n_noise = int((labels == -1).sum())
            noise_frac = n_noise / max(n_valid, 1)
            print(f"  After post-rescue clay-secondary ({n_secondary}): "
                  f"{n_clusters} clusters, {n_noise} noise "
                  f"({100 * noise_frac:.1f}%)")
        else:
            print("  Clay-secondary-split (post-rescue): none applied "
                  "(see skip logs / mega_split_diagnostics.json)")

    n_mega_splits = int(n_mega_primary) + int(n_secondary)
    ctx["n_mega_splits"] = n_mega_splits
    ctx["n_mega_splits_secondary"] = int(n_secondary)
    ctx["mega_split_diagnostics"] = mega_diag
    try:
        with open(os.path.join(output_dir, "mega_split_diagnostics.json"), "w") as f:
            json.dump(mega_diag, f, indent=2)
    except Exception as e:
        print(f"  [WARN] mega_split_diagnostics write failed: {type(e).__name__}: {e}")

    print(f"  Found {n_clusters} clusters, {n_noise} noise pixels "
          f"({100 * noise_frac:.1f}% scene noise; "
          f"SPM-active coverage "
          f"{100.0 * int((labels >= 0).sum()) / max(n_domain, 1):.1f}%)")
    if n_clusters < 1:
        print(
            "  [MAP UNAVAILABLE] HDBSCAN found 0 clusters — all pixels label -1. "
            "Mineral PRESENT proof (Stage 7b) can still pass from SPM+CR+library; "
            "there is no spatial unit map to name."
        )
        ctx["map_available"] = False
    else:
        ctx["map_available"] = True

    dbcv = None
    dbcv_status = "not_computed"
    try:
        # DBCV on a SMALL subsample of clustered SPM-active points (memory-safe)
        fit_global = active_idx[fit_local]
        fit_labels = labels[fit_global]
        clustered_local = np.where(fit_labels >= 0)[0]
        n_fit_cl = int(fit_labels.max() + 1) if fit_labels.max() >= 0 else 0
        if n_fit_cl >= 2 and len(clustered_local) >= max(50, 2 * min_cluster_size):
            validity_mod = getattr(hdbscan, "validity", None)
            validity_index = getattr(validity_mod, "validity_index", None) if validity_mod else None
            if validity_index is None:
                dbcv_status = "unavailable_no_validity_index"
                print("  DBCV skipped (hdbscan.validity.validity_index unavailable)")
            else:
                if len(clustered_local) > DBCV_MAX_SAMPLES:
                    rng_dbcv = np.random.default_rng(42)
                    clustered_local = rng_dbcv.choice(
                        clustered_local, DBCV_MAX_SAMPLES, replace=False
                    )
                sub_idx = fit_global[clustered_local]
                sub_lab = labels[sub_idx]
                # Need ≥2 labels in subsample
                if len(np.unique(sub_lab[sub_lab >= 0])) >= 2:
                    dbcv = float(validity_index(
                        np.ascontiguousarray(feat_for_cluster[sub_idx], dtype=np.float64),
                        sub_lab.astype(np.int32),
                    ))
                    dbcv_status = "ok"
                    print(f"  DBCV score (n={len(sub_idx)} subsample): {dbcv:.4f}")
                    if dbcv < 0.1:
                        print("  WARNING: DBCV < 0.1 — clustering structure is weak")
                else:
                    dbcv_status = "insufficient_labels_in_subsample"
                    print("  DBCV skipped (subsample lacked ≥2 clusters)")
        else:
            dbcv_status = "insufficient_clustered_points"
            print("  DBCV skipped (insufficient clustered points on fit subset)")
    except Exception as e:
        print(f"  DBCV computation failed ({type(e).__name__}: {e})")
        dbcv = None
        dbcv_status = f"failed:{type(e).__name__}"

    good_mask = ctx.get("mineral_good_mask", ctx["good_mask"])
    sam_mask = ctx.get("sam_mask", good_mask)
    cluster_names = {}
    cluster_mineral_matches = {}
    used_lib_families: set = set()
    gate_info = ctx.get("spm_gate_info") or {}

    # Scene-level SPM hotspot masks — use adaptive gates when available (Fix B)
    hotspot = np.zeros_like(spm_flat, dtype=bool)
    for j, name in enumerate(SPM_NAMES):
        if name in ICE_SPM_NAMES:
            continue
        col = spm_flat[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 20:
            continue
        if name in gate_info and gate_info[name].get("passed"):
            thr = float(gate_info[name]["thr_adaptive"])
        else:
            thr = max(SPM_DETECT_THRESH[name], float(np.percentile(col[fin], 90)))
        hotspot[:, j] = fin & (col >= thr)

    for cl in range(n_clusters):
        cl_mask = labels == cl
        cl_spm = spm_flat[cl_mask]
        detect_scores = np.full(N_SPM, -1.0, dtype=np.float64)
        hotspot_frac = np.full(N_SPM, 0.0, dtype=np.float64)
        for j, name in enumerate(SPM_NAMES):
            if name in ICE_SPM_NAMES:
                continue
            # Do not score SPM indices that failed the adaptive scene gate
            if name in gate_info and not gate_info[name].get("passed", False):
                continue
            scene_col = spm_flat[:, j]
            finite_scene = np.isfinite(scene_col)
            if finite_scene.sum() < 10:
                continue
            thr_abs = SPM_DETECT_THRESH[name]
            if name in gate_info and "thr_adaptive" in gate_info[name]:
                thr = float(gate_info[name]["thr_adaptive"])
            else:
                thr_rel = np.percentile(scene_col[finite_scene], 90)
                thr = max(thr_abs, thr_rel)
            col = cl_spm[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                continue
            detect_scores[j] = float(np.mean(col[finite] >= thr))
            hotspot_frac[j] = float(np.mean(hotspot[cl_mask, j]))
        # Prefer SPM indices where the cluster overlaps scene hotspots
        # Soft priority: BD1900 hydratation over SINDEX continuum proxy
        score = detect_scores.copy()
        score[hotspot_frac < 0.05] = -1.0
        for j, name in enumerate(SPM_NAMES):
            if score[j] < 0:
                continue
            score[j] *= float(SPM_LABEL_PRIORITY.get(name, 1.0))
        if np.all(score < 0):
            score = detect_scores  # fallback
        if np.all(score < 0):
            spm_name = "Unknown"
            detect_best = 0.0
            best_spm_idx = -1
            spm_gate_ok = True
        else:
            best_spm_idx = int(np.argmax(score))
            detect_best = float(max(0.0, detect_scores[best_spm_idx]))
            # Require some hotspot support for authoritative SPM group
            if hotspot_frac[best_spm_idx] < 0.05 and detect_best < 0.50:
                spm_name = "Unknown"
                detect_best = float(hotspot_frac[best_spm_idx])
                spm_gate_ok = True
            else:
                spm_idx_name = SPM_NAMES[best_spm_idx]
                spm_name = SPM_MINERAL_MAP[spm_idx_name]
                spm_gate_ok = bool(
                    gate_info.get(spm_idx_name, {}).get("passed", True)
                )
        cluster_names[cl] = spm_name

        cl_coords = valid_coords_all[cl_mask]
        mean_spec = np.nanmean(cube_cr[cl_coords[:, 0], cl_coords[:, 1], :], axis=0)
        diag_m = ctx.get("diag_sam_mask", sam_mask)
        topk, topk_global, sam_mode = match_library_family_then_global(
            mean_spec, lib_cr, lib_names, good_mask,
            spm_group=spm_name if spm_name not in ("Unknown", "") else "",
            k=5, sam_mask=diag_m,
        )
        if topk:
            best_name, best_angle, best_lib_idx = topk[0]
            sam_margin = (
                float(topk[1][1] - topk[0][1]) if len(topk) >= 2 else 1.0
            )
        else:
            best_name, best_angle, best_lib_idx = "Unknown", float(np.pi), -1
            sam_margin = 0.0
        # Open-set: prefer strong global USGS over tied/weak family pool
        if topk_global and len(topk_global) >= 1:
            g_name, g_ang, g_idx = topk_global[0]
            g_margin = (
                float(topk_global[1][1] - topk_global[0][1])
                if len(topk_global) >= 2 else 1.0
            )
            g_fam = mineral_family(g_name)
            if (
                g_margin >= SAM_MARGIN_STRONG
                and g_ang < SAM_STRONG_MAX
                and not is_mixture_library_name(g_name)
                and g_fam not in OPEN_USGS_BLOCKLIST
                and (
                    sam_margin < SAM_MARGIN_STRONG
                    or not _lib_spm_family_agree(g_name, spm_name)
                    or mineral_family(best_name) != g_fam
                )
            ):
                best_name, best_angle, best_lib_idx = g_name, g_ang, g_idx
                sam_margin = g_margin
                sam_mode = "diagnostic_window_open_usgs"
        fam = mineral_family(best_name)
        if fam in used_lib_families and best_angle >= 0.12:
            alt_name, alt_angle, alt_idx = best_name, best_angle, best_lib_idx
            for name, ang, li in topk[1:]:
                if mineral_family(name) == fam:
                    continue
                alt_name, alt_angle, alt_idx = name, ang, li
                break
            else:
                # Prefer alternate within family pool, else global
                scan = topk_global if topk_global else topk
                alt_name, alt_angle, alt_idx = best_name, best_angle, best_lib_idx
                for name, ang, li in scan:
                    if mineral_family(name) == fam:
                        continue
                    alt_name, alt_angle, alt_idx = name, ang, li
                    break
            if alt_idx >= 0 and alt_angle < float(np.pi):
                best_name, best_angle, best_lib_idx = alt_name, alt_angle, alt_idx
                fam = mineral_family(best_name)
                others = [t for t in topk if mineral_family(t[0]) != fam]
                sam_margin = (
                    float(others[0][1] - best_angle) if others else sam_margin
                )
        if fam != "unknown":
            used_lib_families.add(fam)

        hybrid = hybrid_cluster_label(
            spm_name, best_name, best_angle, detect_best,
            sam_margin=sam_margin, spm_gate_passed=spm_gate_ok,
        )
        # Map UX: Unresolved hydrate units with a clear clay hotspot → clay display
        # (authoritative_label stays Unresolved; PRESENT proof unchanged).
        # BD2300 excluded — often collapsed into BD2290 and not PRESENT.
        clay_display_note = None
        if (
            hybrid.get("authoritative_label") == "Unresolved"
            and spm_name == "Hydrated Minerals"
        ):
            clay_cands = []
            for cname in CLAY_DISPLAY_SPM:
                if cname not in SPM_NAMES:
                    continue
                if gate_info.get(cname) and not gate_info[cname].get("passed", False):
                    continue
                cj = SPM_NAMES.index(cname)
                hf = float(hotspot_frac[cj])
                if hf >= CLAY_DISPLAY_HOTSPOT_FRAC:
                    clay_cands.append((hf, cname))
            if clay_cands:
                clay_cands.sort(reverse=True)
                best_hf, best_cname = clay_cands[0]
                second_hf = clay_cands[1][0] if len(clay_cands) > 1 else 0.0
                if best_hf >= second_hf + CLAY_DISPLAY_MARGIN:
                    hybrid["display_label"] = SPM_MINERAL_MAP[best_cname]
                    clay_display_note = (
                        f"clay_hotspot_context {best_cname} "
                        f"frac={best_hf:.2f}"
                    )
                    hybrid["display_label_basis"] = clay_display_note
        # STRONG: discriminative library (family-agree or open USGS)
        if (
            best_angle < 0.15
            and sam_margin >= SAM_MARGIN_STRONG
            and hybrid["label_basis"] in ("library_strong", "library_open_usgs")
        ):
            mq = "STRONG"
        elif best_angle < 0.30:
            mq = "MODERATE"
        else:
            mq = "WEAK"
        cluster_mineral_matches[str(cl)] = {
            **hybrid,
            "library_index": int(best_lib_idx),
            "match_quality": mq,
            "sam_mode": sam_mode,
            "sam_top5": [
                {"name": n, "angle": float(a), "index": int(i)}
                for n, a, i in topk
            ],
            "sam_top5_global": [
                {"name": n, "angle": float(a), "index": int(i)}
                for n, a, i in (topk_global or [])[:5]
            ],
            "spm_detect_score": detect_best,
            "spm_hotspot_frac": float(hotspot_frac[best_spm_idx]) if best_spm_idx >= 0 else 0.0,
            "n_pixels": int(cl_mask.sum()),
        }
        disp = hybrid.get("display_label") or hybrid["authoritative_label"]
        print(f"  Cluster {cl}: {hybrid['authoritative_label']} "
              f"(display={disp}) "
              f"[basis={hybrid['label_basis']}; {sam_mode}] "
              f"(SAM={best_angle:.3f}, margin={sam_margin:.3f}, {mq})")
        if clay_display_note:
            print(f"    display enrichment: {clay_display_note}")
        if topk:
            top_str = ", ".join(f"{n}:{a:.3f}" for n, a, _ in topk[:3])
            print(f"    SAM top3: {top_str}")

    # Merge clusters that share the same SPM mineral group (reduce over-split)
    family_to_ids: Dict[str, List[int]] = {}
    for cl_s, match in cluster_mineral_matches.items():
        fam_key = match.get("spm_name") or match.get("authoritative_label") or "Unknown"
        if fam_key in ("Unknown", "", "Unresolved") or str(
            match.get("authoritative_label", "")
        ).startswith("Unresolved"):
            continue
        family_to_ids.setdefault(fam_key, []).append(int(cl_s))
    merge_map = {}
    for fam_key, ids in family_to_ids.items():
        if len(ids) < 2:
            continue
        # Keep the largest cluster id as canonical
        ids_sorted = sorted(ids, key=lambda i: cluster_mineral_matches[str(i)]["n_pixels"], reverse=True)
        keep = ids_sorted[0]
        for other in ids_sorted[1:]:
            merge_map[other] = keep
    if merge_map:
        print(f"  Merging {len(merge_map)} same-SPM-family cluster(s)")
        for src, dst in merge_map.items():
            labels[labels == src] = dst
            cluster_mineral_matches[str(dst)]["n_pixels"] = int((labels == dst).sum())
            del cluster_mineral_matches[str(src)]
        # Compact ids
        old_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
        remap = {old: new for new, old in enumerate(old_ids)}
        new_labels = np.full_like(labels, -1)
        new_matches = {}
        new_names = {}
        for old, new in remap.items():
            new_labels[labels == old] = new
            if str(old) in cluster_mineral_matches:
                new_matches[str(new)] = cluster_mineral_matches[str(old)]
                new_matches[str(new)]["n_pixels"] = int((new_labels == new).sum())
                new_names[new] = cluster_names.get(old, new_matches[str(new)].get("spm_name", "Unknown"))
        labels = new_labels
        cluster_mineral_matches = new_matches
        cluster_names = new_names
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        n_noise = int((labels == -1).sum())
        print(f"  After merge: {n_clusters} clusters, {n_noise} noise")

    soft_membership = None
    try:
        mem_fn = getattr(hdbscan, "all_points_membership_vectors", None)
        if n_valid <= MAX_FIT and mem_fn is not None and int(labels.max()) >= 0:
            soft_membership = mem_fn(clusterer)
            soft_membership = np.asarray(soft_membership, dtype=np.float32)
            unc_hdb = 1.0 - np.max(soft_membership, axis=1)
        else:
            unc_hdb = 1.0 - probabilities
    except Exception as e:
        print(f"  soft membership skipped ({type(e).__name__}: {e})")
        soft_membership = None
        unc_hdb = 1.0 - probabilities
    unc_hdb = np.asarray(unc_hdb, dtype=np.float32)
    # Soft noise floor — leave room for recon/abundance terms later
    unc_hdb[labels == -1] = np.maximum(unc_hdb[labels == -1], NOISE_UNC_FLOOR)

    # Composite uncertainty placeholder (refined after unmixing with recon error)
    uncertainty = unc_hdb.copy()

    label_map = np.full((H, W), -1, dtype=np.int32)
    uncertainty_map = np.ones((H, W), dtype=np.float32)
    for i, (r, c) in enumerate(valid_coords_all):
        label_map[r, c] = labels[i]
        uncertainty_map[r, c] = uncertainty[i]

    safe_np_save(os.path.join(output_dir, "cluster_labels.npy"), labels)
    safe_np_save(os.path.join(output_dir, "cluster_uncertainty.npy"), uncertainty)
    safe_np_save(os.path.join(output_dir, "label_map.npy"), label_map)
    safe_np_save(os.path.join(output_dir, "uncertainty_map.npy"), uncertainty_map)
    if soft_membership is not None:
        safe_np_save(os.path.join(output_dir, "soft_membership.npy"),
                     soft_membership.astype(np.float32))
    with open(os.path.join(output_dir, "cluster_names.json"), "w") as f:
        json.dump(cluster_mineral_matches, f, indent=2)

    # Cluster purity metrics (separate from DBCV — mixed pixels inflate "noise")
    cluster_purity = {}
    for cl_s, match in cluster_mineral_matches.items():
        cl = int(cl_s)
        cl_mask = labels == cl
        n_pix = int(cl_mask.sum())
        if n_pix < 5:
            continue
        cl_spm = spm_flat[cl_mask]
        # SPM entropy across finite channels (high = mixed mineral groups)
        means = np.nanmean(np.clip(cl_spm, 0, None), axis=0)
        means = np.nan_to_num(means, nan=0.0)
        if means.sum() > 1e-8:
            p = means / means.sum()
            p = p[p > 1e-12]
            spm_entropy = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            spm_entropy = float("nan")
        cluster_purity[cl_s] = {
            "n_pixels": n_pix,
            "spectral_angle_rad": match.get("spectral_angle_rad"),
            "sam_margin_rad": match.get("sam_margin_rad"),
            "lib_spm_agree": match.get("lib_spm_agree"),
            "spm_entropy": spm_entropy,
            "spm_name": match.get("spm_name"),
            "library_match": match.get("library_match"),
            "authoritative_label": match.get("authoritative_label"),
            "display_label": match.get("display_label") or match.get(
                "authoritative_label"
            ),
        }
    # Specimen CR-depth audit: library_strong sulfate/carbonate/olivine names
    # require >=2 positive diagnostic depths above floor (PRESENT unchanged).
    SPECIMEN_DEPTH_KEYS = {
        "sulfate": ("d1.75", "d2.26", "d2.40", "d2.10"),
        "carbonate": ("d2.50", "d2.30"),
        "olivine": ("d1.05", "d1.33"),
    }
    SPECIMEN_MIN_POS_DEPTHS = 2
    sulfate_depth_audit = {}
    specimen_depth_demotions = 0
    for cl_s, match in cluster_mineral_matches.items():
        lib = str(match.get("library_match") or match.get("authoritative_label") or "")
        spm_g = str(match.get("spm_name") or "")
        basis = str(match.get("label_basis") or "")
        auth = str(match.get("authoritative_label") or "")
        lf = lib.lower()
        family = None
        if (
            spm_g in ("Sulfates", "Monohydrated Sulfates")
            or any(x in lf for x in (
                "jarosite", "gypsum", "alunite", "kieserite",
                "szomolnokite", "epsomite", "bassanite",
            ))
        ):
            family = "sulfate"
        elif spm_g == "Carbonates" or any(
            x in lf for x in ("calcite", "dolomite", "magnesite", "siderite")
        ):
            family = "carbonate"
        elif spm_g == "Olivine" or any(
            x in lf for x in ("olivine", "forsterite", "fayalite")
        ):
            family = "olivine"
        if family is None:
            continue
        cl = int(cl_s)
        cl_mask = labels == cl
        if cl_mask.sum() < 10:
            continue
        coords = valid_coords_all[cl_mask]
        mean_spec = np.nanmean(cube_cr[coords[:, 0], coords[:, 1], :], axis=0)
        depths = compute_proof_depths(mean_spec, waves, ~good_mask)
        keys = SPECIMEN_DEPTH_KEYS[family]
        floor = float(PROOF_DEPTH_BD)
        audit = {k: float(depths.get(k, float("nan"))) for k in keys}
        n_pos = sum(
            1 for k in keys
            if np.isfinite(audit[k]) and float(audit[k]) >= floor
        )
        audit["n_pixels"] = int(cl_mask.sum())
        audit["library_match"] = lib
        audit["family"] = family
        audit["n_positive_above_floor"] = int(n_pos)
        audit["depth_floor"] = floor
        # Demote specimen claims only (not SPM group display enrichment)
        needs_depth = basis in (
            "library_strong", "library_open_usgs", "library_only",
            "hybrid_strong_lib", "hybrid_moderate",
        ) and auth not in ("Unresolved", "", "Unknown")
        if needs_depth and n_pos < SPECIMEN_MIN_POS_DEPTHS:
            match["map_proof_prior_label"] = auth
            match["map_proof_prior_basis"] = basis
            match["authoritative_label"] = "Unresolved"
            match["display_label"] = (
                spm_g if spm_g not in ("Unknown", "", None) else "Unresolved"
            )
            match["label_basis"] = "specimen_depth_demoted"
            match["match_quality"] = "WEAK"
            match["specimen_depth_note"] = (
                f"demoted {auth}: {n_pos}/{SPECIMEN_MIN_POS_DEPTHS} "
                f"{family} depths >= {floor}"
            )
            audit["note"] = "demoted_insufficient_positive_depths"
            specimen_depth_demotions += 1
            print(
                f"  Specimen-depth: cluster {cl_s} demoted {auth} -> Unresolved "
                f"({n_pos}/{SPECIMEN_MIN_POS_DEPTHS} {family} depths >= {floor})"
            )
        else:
            audit["note"] = "log_only_depths_ok_or_not_specimen_claim"
        sulfate_depth_audit[cl_s] = audit
        if cl_s in cluster_purity:
            cluster_purity[cl_s]["sulfate_depth_audit"] = audit
            cluster_purity[cl_s]["authoritative_label"] = match.get(
                "authoritative_label"
            )
            cluster_purity[cl_s]["display_label"] = match.get("display_label")
            cluster_purity[cl_s]["label_basis"] = match.get("label_basis")
        print(
            f"  {family.capitalize()} depth audit cl{cl_s} ({lib}): "
            + ", ".join(
                f"{k}={audit[k]:.4f}" if np.isfinite(audit[k]) else f"{k}=nan"
                for k in keys
            )
            + f" | n_pos={n_pos}"
        )
    ctx["specimen_depth_demotions"] = int(specimen_depth_demotions)
    if sulfate_depth_audit:
        with open(os.path.join(output_dir, "sulfate_depth_audit.json"), "w") as f:
            json.dump(sulfate_depth_audit, f, indent=2)
    # Refresh cluster_names if specimen depths demoted any labels
    if specimen_depth_demotions:
        with open(os.path.join(output_dir, "cluster_names.json"), "w") as f:
            json.dump(cluster_mineral_matches, f, indent=2)

    with open(os.path.join(output_dir, "cluster_purity.json"), "w") as f:
        json.dump(cluster_purity, f, indent=2)
    if cluster_purity:
        mean_ent = float(np.nanmean([v["spm_entropy"] for v in cluster_purity.values()]))
        print(f"  Cluster purity: {len(cluster_purity)} units, "
              f"mean SPM entropy={mean_ent:.3f} "
              f"(higher → more mixed within cluster)")

    # Spectral separation among cluster mean CR spectra (discovery quality)
    cluster_spectral_gap = None
    try:
        means_cr = []
        cl_ids_sorted = sorted(int(x) for x in np.unique(labels) if x >= 0)
        diag_m = ctx.get("diag_sam_mask", good_mask)
        for cl in cl_ids_sorted:
            m = labels == cl
            if m.sum() < 10:
                continue
            coords = valid_coords_all[m]
            means_cr.append(
                np.nanmean(cube_cr[coords[:, 0], coords[:, 1], :], axis=0)
            )
        gaps = []
        for i in range(len(means_cr)):
            for j in range(i + 1, len(means_cr)):
                a = np.nan_to_num(means_cr[i], nan=0.0)
                b = np.nan_to_num(means_cr[j], nan=0.0)
                msk = (
                    np.isfinite(means_cr[i]) & np.isfinite(means_cr[j])
                    & good_mask.astype(bool)
                )
                if diag_m is not None:
                    msk = msk & np.asarray(diag_m, dtype=bool)
                if msk.sum() < 10:
                    continue
                aa, bb = a[msk], b[msk]
                na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
                if na < 1e-8 or nb < 1e-8:
                    continue
                gaps.append(float(np.arccos(np.clip(np.dot(aa, bb) / (na * nb), -1, 1))))
        if gaps:
            cluster_spectral_gap = float(np.mean(gaps))
            print(f"  Mean inter-cluster SAM gap: {cluster_spectral_gap:.4f} rad "
                  f"(n_pairs={len(gaps)})")
    except Exception as e:
        print(f"  Spectral gap skipped ({type(e).__name__}: {e})")

    ctx.update({
        "labels": labels,
        "probabilities": probabilities,
        "uncertainty": uncertainty,
        "unc_hdb": unc_hdb,
        "label_map": label_map,
        "uncertainty_map": uncertainty_map,
        "n_clusters": n_clusters,
        "dbcv": dbcv,
        "dbcv_status": dbcv_status,
        "cluster_mineral_matches": cluster_mineral_matches,
        "cluster_purity": cluster_purity,
        "cluster_spectral_gap": cluster_spectral_gap,
        "sulfate_depth_audit": sulfate_depth_audit,
        "soft_membership": soft_membership,
        "spm_flat": spm_flat,
    })
    return ctx

# -------------------------------------------------------------
# STAGE 5 — Neural Unmixing
# -------------------------------------------------------------
def stage5_unmixing(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 5 — Physics-Constrained Neural Unmixing")
    print("-" * 60)

    device = ctx.get("device", _device())
    encoder = ctx["encoder"]
    lib_l2 = ctx["lib_spectra_cr_l2"]
    lib_cr = ctx["lib_spectra_cr"]
    lib_spm = ctx["lib_spm"]
    lib_names = ctx["lib_names"]
    n_clusters = max(1, ctx["n_clusters"])
    pool_spectra = ctx["pool_spectra"]
    pool_spectra_cr = ctx["pool_spectra_cr"]
    latent_vectors = ctx["latent_vectors"]
    valid_coords_all = ctx["valid_coords_all"]
    cube_cr = ctx["cube_cr"]
    valid = ctx["valid_pixel_mask"]
    labels = ctx["labels"]
    H, W, B = ctx["H"], ctx["W"], ctx["B"]
    output_dir = ctx["output_dir"]

    sam_mask = ctx.get("sam_mask", ctx["good_mask"])
    spm_flat = ctx.get("spm_flat")
    if spm_flat is None:
        spm_flat = ctx["spm_map"][valid]

    # Active SPM mineral groups drive endmember budget (scientifically motivated)
    # Prefer Fix-B passing indices only (no ice / no weak scene-wide hits)
    active_groups = []
    spm_passed = ctx.get("spm_passed") or []
    gate_info = ctx.get("spm_gate_info") or {}
    if spm_passed:
        for name in spm_passed:
            active_groups.append(SPM_MINERAL_MAP[name])
    else:
        for j, name in enumerate(SPM_NAMES):
            if name in ICE_SPM_NAMES:
                continue
            notes = ctx.get("spm_notes", {})
            if not str(notes.get(name, "")).startswith("ok"):
                continue
            col = spm_flat[:, j] if spm_flat.ndim == 2 else ctx["spm_map"][:, :, j][valid]
            fin = np.isfinite(col)
            if fin.sum() < 20:
                continue
            thr = max(SPM_DETECT_THRESH[name], float(np.percentile(col[fin], 90)))
            if float(np.percentile(col[fin], 90)) >= thr:
                active_groups.append(SPM_MINERAL_MAP[name])
    # One mafic background only if no mafic already passed
    if not any("Pyroxene" in g or g == "Olivine" for g in active_groups):
        active_groups.append("High-Ca Pyroxene")
    active_groups = list(dict.fromkeys(active_groups))
    print(f"  SPM-active mineral groups (Fix B/D seeds): {active_groups}")

    n_end = min(6, max(n_clusters + 1, len(active_groups), 3), len(lib_l2))
    n_end = max(2, n_end)

    def _family_has_spm_support(fam: str) -> bool:
        """Allow EM only if its family is compatible with an active SPM group."""
        if not fam or fam == "unknown":
            return False
        fl = fam.lower()
        mafic = {"olivine", "forsterite", "fayalite", "pyroxene", "enstatite",
                 "bronzite", "hypersthene", "diopside", "augite", "pigeonite"}
        clay = {"nontronite", "saponite", "montmorillonite", "kaolinite", "illite",
                "beidellite", "halloysite", "hectorite", "vermiculite", "talc",
                "chlorite", "clinochlore", "chamosite", "serpentine", "antigorite",
                "lizardite", "chrysotile"}
        sulfate = {"gypsum", "bassanite", "kieserite", "jarosite", "alunite",
                   "anhydrite", "epsomite"}
        carb = {"calcite", "dolomite", "magnesite", "siderite"}
        for g in active_groups:
            gl = g.lower()
            if fl in mafic and ("pyroxene" in gl or "olivine" in gl):
                return True
            if fl in clay and any(x in gl for x in ("clay", "phyllosilicate", "hydrat", "serpentine")):
                return True
            if fl in sulfate and ("sulfate" in gl or "hydrat" in gl):
                return True
            if fl in carb and "carbonate" in gl:
                return True
            if fl in gl or any(tok in fl for tok in gl.replace("/", " ").split() if len(tok) > 3):
                return True
        return False

    prior_idx = [i for i, n in enumerate(lib_names) if _name_in_list(n, MARS_PRIOR_MINERALS)]
    # Restrict FPS pool to SPM-supported families (stops Alunite/Siderite without SPM)
    supported_prior = [
        i for i in prior_idx if _family_has_spm_support(mineral_family(lib_names[i]))
    ]
    if len(supported_prior) >= n_end:
        pool_idx = supported_prior
    elif len(prior_idx) >= n_end:
        print(f"  [WARN] SPM-supported prior pool small ({len(supported_prior)}); "
              f"using full Mars-prior pool.")
        pool_idx = prior_idx
    else:
        print(f"  [WARN] Only {len(prior_idx)} Mars-prior library spectra; expanding pool.")
        pool_idx = list(range(len(lib_l2)))

    def _angle_ok(cand: int, chosen: List[int]) -> bool:
        if not chosen:
            return True
        for s in chosen:
            if spectral_angle(lib_l2[cand], lib_l2[s], band_mask=sam_mask) < MIN_ENDMEMBER_ANGLE:
                return False
        return True

    selected: List[int] = []
    selected_families: set = set()
    matches = ctx.get("cluster_mineral_matches", {})

    # Seed 1: cluster library matches (always allowed — they are observed units)
    for cl_key, match in matches.items():
        li = match.get("library_index", -1)
        if not (isinstance(li, int) and li >= 0 and li not in selected):
            continue
        fam = mineral_family(lib_names[li])
        if fam in selected_families or not _angle_ok(li, selected):
            continue
        selected.append(li)
        selected_families.add(fam)
        if len(selected) >= n_end:
            break

    # Seed 1b: force ≥1 library specimen per active SPM mineral group
    def _pick_family_seed(fam_stems: Sequence[str]) -> int:
        for stem in fam_stems:
            for i in pool_idx:
                if i in selected:
                    continue
                if is_mixture_library_name(lib_names[i]):
                    continue
                if stem not in mineral_family(lib_names[i]) and stem not in lib_names[i].lower():
                    continue
                if not _angle_ok(i, selected):
                    continue
                return i
        return -1

    for group in active_groups:
        if len(selected) >= n_end:
            break
        stems = SPM_GROUP_SEED_FAMILIES.get(group)
        if not stems:
            continue
        # Skip if we already have a compatible family
        if any(
            any(s in mineral_family(lib_names[i]) or s in lib_names[i].lower() for s in stems)
            for i in selected
        ):
            continue
        li = _pick_family_seed(stems)
        if li >= 0:
            selected.append(li)
            selected_families.add(mineral_family(lib_names[li]))
            print(f"  Seeded {lib_names[li]} for SPM group '{group}'")

    # Seed 2: SPM-hot scene pixels → nearest Mars-prior library spectrum
    # Prefer diagnostic-window SAM so clay/phyllosilicate hotspots don't map to hematite.
    if len(selected) < n_end and spm_flat is not None:
        max_spm = np.nan_to_num(
            nanmax_spm_exclude_ice(spm_flat, axis=1, fill=0.0), nan=0.0
        )
        hot_n = min(40, max(10, int(0.002 * len(max_spm))))
        hot_idx = np.argsort(max_spm)[-hot_n:]
        for hi in hot_idx:
            if len(selected) >= n_end:
                break
            r, c = valid_coords_all[hi]
            spec = cube_cr[r, c, :]
            diag_m = ctx.get("diag_sam_mask", sam_mask)
            name, ang, li = match_library_best(
                spec, lib_cr, lib_names, ctx["good_mask"],
                prior_only=True, sam_mask=diag_m,
            )
            if li < 0 or ang > 0.35:
                continue
            fam = mineral_family(name)
            if fam in selected_families or not _angle_ok(li, selected):
                continue
            if not _family_has_spm_support(fam):
                continue
            selected.append(li)
            selected_families.add(fam)
        print(f"  SPM-hot library seeds: {len(selected)} endmembers so far")

    if not selected:
        pool_spm = lib_spm[pool_idx]
        selected = [pool_idx[int(np.nanargmax(np.nanmean(pool_spm, axis=1)))]]
        selected_families.add(mineral_family(lib_names[selected[0]]))

    lib_l2_num = np.nan_to_num(lib_l2, nan=0.0)
    stuck = 0
    while len(selected) < n_end and stuck < len(pool_idx) + 5:
        stuck += 1
        best_i, best_d = -1, -1.0
        for i in pool_idx:
            if i in selected:
                continue
            fam = mineral_family(lib_names[i])
            if fam in selected_families or not _angle_ok(i, selected):
                continue
            if not _family_has_spm_support(fam):
                continue
            dmin = min(float(np.linalg.norm(lib_l2_num[i] - lib_l2_num[s])) for s in selected)
            if dmin > best_d:
                best_d, best_i = dmin, i
        if best_i < 0:
            break
        selected.append(best_i)
        selected_families.add(mineral_family(lib_names[best_i]))

    n_end = len(selected)
    E_init = np.nan_to_num(lib_cr[selected], nan=0.0).astype(np.float32)
    E_init = np.clip(E_init, CR_SANITIZE_LO, CR_SANITIZE_HI)
    endmember_names = [lib_names[i] for i in selected]
    print(f"  Endmembers ({n_end}, SPM-co-supported + unique-family FPS): {endmember_names}")

    # Freeze encoder — Phase A
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    abundance_net = AbundanceNetwork(latent_dim=LATENT_DIM, n_endmembers=n_end).to(device)
    print(f"  AbundanceNetwork: always fresh (never transferred; mode={TRAIN_MODE})")
    E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32, device=device))
    E_init_t = torch.tensor(E_init, dtype=torch.float32, device=device)
    if UNMIX_FREEZE_E_PHASE_A:
        E.requires_grad_(False)
        print("  Phase A: library endmembers FROZEN (abundances learn under ASC)")

    opt_params = list(abundance_net.parameters())
    if E.requires_grad:
        opt_params.append(E)
    optimizer = torch.optim.AdamW(opt_params, lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNMIX_EPOCHS_A)
    use_amp = device.type == "cuda"

    # Encode pool for unmixing
    with torch.no_grad():
        pool_latent_list = []
        for start in range(0, len(pool_spectra), BATCH_INF):
            end = min(start + BATCH_INF, len(pool_spectra))
            t = torch.from_numpy(pool_spectra[start:end]).float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = encoder(t)
            pool_latent_list.append(z.float().cpu().numpy())
        pool_latent = np.concatenate(pool_latent_list, axis=0).astype(np.float32)

    n_pool = len(pool_latent)
    idx_all = np.arange(n_pool)

    def _unmix_loss(abundances, reconstructed, target):
        loss_recon = F.mse_loss(reconstructed, target)
        # Mild entropy min → sparse pixels; keep weak so one EM cannot monopolize
        loss_sparse = -torch.mean(
            torch.sum(abundances * torch.log(abundances + 1e-8), dim=1)
        )
        loss_lib = F.mse_loss(E, E_init_t)
        mean_a = abundances.mean(dim=0)
        n_e = abundances.shape[1]
        # Stronger use regularizer → discourage single-EM collapse (ASC preserved)
        loss_use = torch.sum((mean_a - 1.0 / n_e) ** 2)
        # Batch-max diversity: penalize if max mean abundance dominates
        loss_dom = torch.relu(mean_a.max() - 0.70) ** 2
        if E.requires_grad:
            e_n = F.normalize(E, dim=1)
            sim = torch.mm(e_n, e_n.t())
            loss_div = (sim.sum() - torch.diag(sim).sum()) / max(1, n_e * (n_e - 1))
        else:
            loss_div = torch.zeros((), device=target.device)
        return (
            loss_recon
            + 0.002 * loss_sparse
            + UNMIX_LIB_WEIGHT * loss_lib
            + 0.05 * loss_div
            + 0.12 * loss_use
            + 0.08 * loss_dom
        )

    print(f"  Phase A: {UNMIX_EPOCHS_A} epochs (encoder frozen, E "
          f"{'frozen' if not E.requires_grad else 'learnable'})")
    abundance_net.train()
    for epoch in range(UNMIX_EPOCHS_A):
        np.random.shuffle(idx_all)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pool, ENCODER_BATCH):
            batch_idx = idx_all[start:start + ENCODER_BATCH]
            if len(batch_idx) < 4:
                continue
            z = torch.from_numpy(pool_latent[batch_idx]).float().to(device)
            target = torch.from_numpy(pool_spectra_cr[batch_idx]).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            abundances = abundance_net(z)
            reconstructed = torch.mm(abundances, E)
            loss = _unmix_loss(abundances, reconstructed, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(abundance_net.parameters()), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == UNMIX_EPOCHS_A - 1:
            print(f"    Epoch {epoch:03d}  loss={epoch_loss / max(1, n_batches):.5f}")

    # Prune unused endmembers (mean abund < 2%) then rebuild net — keep ASC core
    with torch.no_grad():
        mean_ab = []
        for start in range(0, n_pool, BATCH_INF):
            z = torch.from_numpy(pool_latent[start:start + BATCH_INF]).float().to(device)
            mean_ab.append(abundance_net(z).mean(dim=0).cpu().numpy())
        mean_ab = np.mean(np.stack(mean_ab, axis=0), axis=0)
    keep = [i for i, m in enumerate(mean_ab) if m >= 0.02]
    if len(keep) < 2:
        keep = list(np.argsort(mean_ab)[-2:])
    if len(keep) < n_end:
        print(f"  Pruning unused endmembers: keep {[endmember_names[i] for i in keep]} "
              f"(mean abund={[float(mean_ab[i]) for i in keep]})")
        E_init = E_init[keep]
        endmember_names = [endmember_names[i] for i in keep]
        n_end = len(keep)
        abundance_net = AbundanceNetwork(latent_dim=LATENT_DIM, n_endmembers=n_end).to(device)
        E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32, device=device))
        E_init_t = torch.tensor(E_init, dtype=torch.float32, device=device)
        E.requires_grad_(False)
        # Short retrain abundances on pruned set
        optimizer = torch.optim.AdamW(abundance_net.parameters(), lr=5e-4, weight_decay=1e-4)
        abundance_net.train()
        for epoch in range(min(15, UNMIX_EPOCHS_A)):
            np.random.shuffle(idx_all)
            for start in range(0, n_pool, ENCODER_BATCH):
                batch_idx = idx_all[start:start + ENCODER_BATCH]
                if len(batch_idx) < 4:
                    continue
                z = torch.from_numpy(pool_latent[batch_idx]).float().to(device)
                target = torch.from_numpy(pool_spectra_cr[batch_idx]).float().to(device)
                optimizer.zero_grad(set_to_none=True)
                abundances = abundance_net(z)
                loss = _unmix_loss(abundances, torch.mm(abundances, E), target)
                loss.backward()
                optimizer.step()

    # Phase B — fine-tune encoder + abundances; keep E library-anchored (frozen)
    print(f"  Phase B: {UNMIX_EPOCHS_B} epochs (encoder+abundances; E frozen to library)")
    for p in encoder.parameters():
        p.requires_grad = True
    E.requires_grad_(False)
    optimizer_j = torch.optim.AdamW(
        list(encoder.parameters()) + list(abundance_net.parameters()),
        lr=5e-5,
    )
    encoder.train()
    abundance_net.train()
    for epoch in range(UNMIX_EPOCHS_B):
        np.random.shuffle(idx_all)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pool, ENCODER_BATCH):
            batch_idx = idx_all[start:start + ENCODER_BATCH]
            if len(batch_idx) < 4:
                continue
            specs = torch.from_numpy(pool_spectra[batch_idx]).float().unsqueeze(1).to(device)
            target = torch.from_numpy(pool_spectra_cr[batch_idx]).float().to(device)
            optimizer_j.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = encoder(specs)
            z = z.float()
            abundances = abundance_net(z)
            reconstructed = torch.mm(abundances, E)
            loss = _unmix_loss(abundances, reconstructed, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(abundance_net.parameters()),
                max_norm=1.0,
            )
            optimizer_j.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        if epoch % 5 == 0 or epoch == UNMIX_EPOCHS_B - 1:
            print(f"    Epoch {epoch:03d}  loss={epoch_loss / max(1, n_batches):.5f}")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    abundance_net.eval()
    torch.save(
        {"abundance_net": abundance_net.state_dict(),
         "E": E.detach().cpu(),
         "endmember_names": endmember_names},
        os.path.join(output_dir, "unmixer_B438.pth"),
    )

    # Re-encode scene after fine-tune
    n_valid = len(valid_coords_all)
    with torch.no_grad():
        for start in range(0, n_valid, BATCH_INF):
            end = min(start + BATCH_INF, n_valid)
            coords = valid_coords_all[start:end]
            specs = np.nan_to_num(
                ctx["cube_cr_l2"][coords[:, 0], coords[:, 1], :], nan=0.0
            ).astype(np.float32)
            t = torch.from_numpy(specs).float().unsqueeze(1).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = encoder(t)
            latent_vectors[start:end] = z.float().cpu().numpy()
    norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    latent_vectors = (latent_vectors / norms).astype(np.float32)
    safe_np_save(os.path.join(output_dir, "latent_vectors.npy"), latent_vectors)
    ctx["latent_vectors"] = latent_vectors

    abundance_map = np.full((H, W, n_end), np.nan, dtype=np.float32)
    recon_error_map = np.full((H, W), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_valid, BATCH_INF):
            end = min(start + BATCH_INF, n_valid)
            z = torch.from_numpy(latent_vectors[start:end]).float().to(device)
            abundances = abundance_net(z)
            reconstructed = torch.mm(abundances, E)
            ab_np = abundances.cpu().numpy()
            re_np = reconstructed.cpu().numpy()
            coords = valid_coords_all[start:end]
            for i, (r, c) in enumerate(coords):
                abundance_map[r, c, :] = ab_np[i]
                actual = cube_cr[r, c, :]
                mask = np.isfinite(actual)
                if mask.any():
                    # Compare in sanitized CR space (targets clipped to [0, 2])
                    tgt = np.clip(actual[mask], CR_SANITIZE_LO, CR_SANITIZE_HI)
                    pred = np.clip(re_np[i, mask], CR_SANITIZE_LO, CR_SANITIZE_HI)
                    recon_error_map[r, c] = float(np.mean((pred - tgt) ** 2))
                else:
                    recon_error_map[r, c] = np.nan

    dominant_map = np.full((H, W), -1, dtype=np.int32)
    dominant_map[valid] = np.nanargmax(abundance_map[valid], axis=1)

    # Physical validity
    ab_valid = abundance_map[valid]
    assert np.all(ab_valid >= -1e-6), "Negative abundances found!"
    row_sums = np.nansum(ab_valid, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3), "Row sums not ~1.0!"
    max_ab = np.nanmax(ab_valid, axis=1)
    mean_max = float(np.mean(max_ab))
    print(f"  Mean max abundance: {mean_max:.3f} (expected 0.3–0.9)")
    mean_per_em = np.nanmean(ab_valid, axis=0)
    dead_ems = [endmember_names[i] for i, m in enumerate(mean_per_em) if m < 0.01]
    print(f"  Mean abundance / EM: "
          + ", ".join(f"{n}={m:.3f}" for n, m in zip(endmember_names, mean_per_em)))
    if dead_ems:
        print(f"  [WARN] Near-dead endmembers (mean abund < 1%): {dead_ems}")

    # Agreement proxy: map cluster id vs dominant endmember is not 1:1;
    # report fraction of clustered pixels with a confident dominant abundance.
    clustered = labels >= 0
    if clustered.any():
        conf = max_ab[clustered]
        print(f"  Confident dominant (max abund>0.35) among clustered: "
              f"{100.0 * np.mean(conf > 0.35):.1f}%")

    safe_np_save(os.path.join(output_dir, "abundance_map.npy"), abundance_map)
    safe_np_save(os.path.join(output_dir, "recon_error_map.npy"), recon_error_map)
    safe_np_save(os.path.join(output_dir, "dominant_map.npy"), dominant_map)
    with open(os.path.join(output_dir, "endmember_names.json"), "w") as f:
        json.dump(endmember_names, f, indent=2)

    # Composite uncertainty: HDBSCAN + (1 - max abundance) + scaled recon error
    unc_hdb = ctx.get("unc_hdb", ctx["uncertainty"]).astype(np.float32)
    unc_ab = np.zeros_like(unc_hdb)
    unc_re = np.zeros_like(unc_hdb)
    re_valid = recon_error_map[valid]
    re_finite = re_valid[np.isfinite(re_valid)]
    re_p95 = float(np.percentile(re_finite, 95)) if re_finite.size else 1.0
    re_p95 = max(re_p95, 1e-6)
    for i, (r, c) in enumerate(valid_coords_all):
        unc_ab[i] = 1.0 - float(np.nanmax(abundance_map[r, c, :]))
        re = recon_error_map[r, c]
        unc_re[i] = float(np.clip(re / re_p95, 0.0, 1.0)) if np.isfinite(re) else 1.0
    uncertainty = (0.45 * unc_hdb + 0.35 * unc_ab + 0.20 * unc_re).astype(np.float32)
    uncertainty[labels == -1] = np.maximum(uncertainty[labels == -1], NOISE_UNC_FLOOR)
    uncertainty_map = np.ones((H, W), dtype=np.float32)
    for i, (r, c) in enumerate(valid_coords_all):
        uncertainty_map[r, c] = uncertainty[i]
    safe_np_save(os.path.join(output_dir, "cluster_uncertainty.npy"), uncertainty)
    safe_np_save(os.path.join(output_dir, "uncertainty_map.npy"), uncertainty_map)

    _clear_cuda()
    ctx.update({
        "abundance_net": abundance_net,
        "E": E,
        "E_init": E_init,
        "endmember_names": endmember_names,
        "n_endmembers": n_end,
        "abundance_map": abundance_map,
        "recon_error_map": recon_error_map,
        "dominant_map": dominant_map,
        "mean_max_abundance": mean_max,
        "uncertainty": uncertainty,
        "uncertainty_map": uncertainty_map,
    })
    return ctx


# -------------------------------------------------------------
# STAGE 6 — Spatial Smoothing
# -------------------------------------------------------------
def smooth_labels(
    label_map: np.ndarray,
    segments: np.ndarray,
    uncertainty_map: np.ndarray,
    cube_cr: np.ndarray,
    n_iterations: int = 5,
    protect_labels: Optional[set] = None,
) -> np.ndarray:
    """
    Weighted majority-vote smoothing inside already-clustered pixels only.

    Core idea preserved (SLIC + spectral-neighbor votes + minority protect).
    Critical fix: never promote HDBSCAN noise (-1) into a mineral class —
    noise stays unclassified so the map does not flood with the majority label.
    """
    smoothed = label_map.copy()
    protect_labels = protect_labels or set()
    for _ in range(n_iterations):
        new_labels = smoothed.copy()
        for seg_id in np.unique(segments):
            if seg_id < 0:
                continue
            seg_mask = segments == seg_id
            if not seg_mask.any():
                continue

            # Operate only on clustered pixels; leave noise as -1
            clustered = seg_mask & (smoothed >= 0)
            if not clustered.any():
                continue

            # Freeze protected minority labels inside this superpixel
            active = clustered.copy()
            for lbl in list(protect_labels):
                keep = active & (smoothed == lbl)
                if keep.any() and int(keep.sum()) >= 3:
                    new_labels[keep] = lbl
                    active = active & ~keep
            if not active.any():
                continue

            seg_labels = smoothed[active]
            valid_labels = seg_labels[seg_labels >= 0]
            if len(valid_labels) == 0:
                continue

            # Require a real majority inside the clustered portion (avoid 1-pixel votes)
            if len(valid_labels) < 3:
                continue

            dilated = binary_dilation(seg_mask, iterations=1)
            neighbor_mask = dilated & ~seg_mask
            votes: Counter = Counter(valid_labels.tolist())
            own_unc = float(np.mean(uncertainty_map[active]))
            own_weight = 1.0 / (own_unc + 0.01)
            for lbl, count in list(votes.items()):
                votes[lbl] = count * own_weight

            seg_spec = np.nanmean(cube_cr[active], axis=0)
            for nseg in np.unique(segments[neighbor_mask]):
                if nseg < 0:
                    continue
                nseg_mask = segments == nseg
                nseg_valid = smoothed[nseg_mask]
                nseg_valid = nseg_valid[nseg_valid >= 0]
                if len(nseg_valid) == 0:
                    continue
                nseg_spec = np.nanmean(cube_cr[nseg_mask & (smoothed >= 0)], axis=0)
                if not np.any(np.isfinite(nseg_spec)):
                    continue
                spec_dist = spectral_angle(seg_spec, nseg_spec)
                if spec_dist > 0.3:
                    continue
                weight = 1.0 / (spec_dist + 0.01)
                for lbl in nseg_valid:
                    votes[lbl] = votes.get(lbl, 0) + weight

            if votes:
                winner = max(votes, key=votes.get)
                new_labels[active] = winner
                # Explicitly keep noise pixels in this superpixel as -1
                noise_in_seg = seg_mask & (smoothed < 0)
                new_labels[noise_in_seg] = -1
        smoothed = new_labels
    return smoothed


def stage6_smoothing(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 6 — CRF / Superpixel Spatial Smoothing")
    print("-" * 60)

    from skimage.segmentation import slic

    waves = ctx["waves"]
    cube_cr = ctx["cube_cr"]
    valid = ctx["valid_pixel_mask"]
    label_map = ctx["label_map"]
    uncertainty_map = ctx["uncertainty_map"]
    abundance_map = ctx["abundance_map"]
    H, W = ctx["H"], ctx["W"]
    output_dir = ctx["output_dir"]

    ch1 = int(np.argmin(np.abs(waves - 1.30)))
    ch2 = int(np.argmin(np.abs(waves - 1.80)))
    ch3 = int(np.argmin(np.abs(waves - 2.30)))
    rgb = np.stack([
        np.nan_to_num(cube_cr[:, :, ch1]),
        np.nan_to_num(cube_cr[:, :, ch2]),
        np.nan_to_num(cube_cr[:, :, ch3]),
    ], axis=2).astype(np.float64)
    for ch in range(3):
        vals = rgb[:, :, ch][valid]
        if vals.size == 0:
            continue
        vmin, vmax = np.percentile(vals, [2, 98])
        rgb[:, :, ch] = np.clip((rgb[:, :, ch] - vmin) / (vmax - vmin + 1e-8), 0, 1)

    n_segments = max(50, int(valid.sum() // 50))
    n_segments = min(n_segments, max(10, int(valid.sum())))
    segments = slic(
        rgb, n_segments=n_segments, compactness=10, sigma=1.0,
        start_label=0, mask=valid,
    )
    print(f"  SLIC superpixels: ~{n_segments} requested, "
          f"{len(np.unique(segments[valid]))} formed")

    # Protect minority clusters (small but real units)
    labels = ctx["labels"]
    protect = set()
    for cl in range(ctx["n_clusters"]):
        n_pix = int((labels == cl).sum())
        if MIN_CLUSTER_KEEP <= n_pix < max(MIN_CLUSTER_KEEP * 5, int(0.05 * labels.size)):
            protect.add(cl)
    print(f"  Protected minority clusters: {sorted(protect)}")

    smoothed_label_map = smooth_labels(
        label_map, segments, uncertainty_map, cube_cr,
        n_iterations=5, protect_labels=protect,
    )

    smoothed_abundance = abundance_map.copy()
    for seg_id in np.unique(segments):
        if seg_id < 0:
            continue
        seg_mask = segments == seg_id
        if not seg_mask.any():
            continue
        seg_abund = np.nanmean(abundance_map[seg_mask], axis=0)
        smoothed_abundance[seg_mask] = seg_abund

    # Renormalize
    totals = np.nansum(smoothed_abundance, axis=2, keepdims=True)
    totals[totals < 1e-6] = 1.0
    smoothed_abundance = smoothed_abundance / totals
    smoothed_abundance[~valid] = np.nan

    safe_np_save(os.path.join(output_dir, "smoothed_label_map.npy"), smoothed_label_map)
    safe_np_save(os.path.join(output_dir, "smoothed_abundance_map.npy"),
                 smoothed_abundance.astype(np.float32))
    safe_np_save(os.path.join(output_dir, "slic_segments.npy"), segments.astype(np.int32))

    ctx.update({
        "segments": segments,
        "smoothed_label_map": smoothed_label_map,
        "smoothed_abundance": smoothed_abundance,
    })
    return ctx


# -------------------------------------------------------------
# RARE MINERAL DETECTION
# -------------------------------------------------------------
def detect_rare_minerals(ctx: dict) -> dict:
    """
    Strategy 1: HDBSCAN noise-pixel library matching (Mars-prior / rare list).
    Strategy 2: Secondary abundance of rare endmembers above strict threshold.
    """
    print("\n" + "-" * 60)
    print(" RARE MINERAL DETECTION (Strategies 1 + 2)")
    print("-" * 60)

    labels = ctx["labels"]
    valid_coords_all = ctx["valid_coords_all"]
    cube_cr = ctx["cube_cr"]
    lib_cr = ctx["lib_spectra_cr"]
    lib_names = ctx["lib_names"]
    abundance_map = ctx["abundance_map"]
    endmember_names = ctx["endmember_names"]
    valid = ctx["valid_pixel_mask"]
    good_mask = ctx["good_mask"]
    H, W = ctx["H"], ctx["W"]
    output_dir = ctx["output_dir"]
    n_end = ctx["n_endmembers"]

    rare_detections = []
    noise_idx = np.where(labels == -1)[0]
    max_noise = 2000 if H >= FULL_HEIGHT_MIN_LINES else 500
    if len(noise_idx) > max_noise:
        noise_idx = np.random.choice(noise_idx, max_noise, replace=False)

    for i in noise_idx:
        r, c = valid_coords_all[i]
        spec = cube_cr[r, c, :]
        best_name, best_angle, _ = match_library_best(
            spec, lib_cr, lib_names, good_mask, prior_only=True,
            sam_mask=ctx.get("sam_mask"),
        )
        if best_angle < RARE_ANGLE_THRESH and (
            _name_in_list(best_name, MARS_RARE_MINERALS)
            or _name_in_list(best_name, MARS_PRIOR_MINERALS)
        ):
            # Prefer rare list for reporting confidence boost
            is_rare = _name_in_list(best_name, MARS_RARE_MINERALS)
            if is_rare or best_angle < 0.15:
                rare_detections.append({
                    "pixel": [int(r), int(c)],
                    "mineral": best_name,
                    "spectral_angle": float(best_angle),
                    "confidence": "HIGH" if best_angle < 0.12 else "MODERATE",
                    "is_rare_prior": bool(is_rare),
                    "source": "hdbscan_noise",
                })

    print(f"  Strategy 1: {len(rare_detections)} noise-pixel rare/prior matches "
          f"(of {len(noise_idx)} noise pixels scanned)")

    # Strategy 2 — secondary abundances of rare endmembers only (vectorized).
    # Gate: secondary abund high, dominant not near-pure, optional SPM co-support.
    minor_detections = []
    spm_map = ctx.get("spm_map")
    spm_idx = {n: i for i, n in enumerate(SPM_NAMES)}
    max_ab_map = np.nanmax(abundance_map, axis=2)
    dom_map = np.full((H, W), -1, dtype=np.int32)
    finite_ab = np.all(np.isfinite(abundance_map), axis=2) & valid
    if finite_ab.any():
        dom_map[finite_ab] = np.nanargmax(abundance_map[finite_ab], axis=1)

    mean_ab_per_em = np.full(n_end, np.nan, dtype=np.float64)
    if finite_ab.any():
        mean_ab_per_em = np.nanmean(abundance_map[finite_ab], axis=0)

    for em_idx in range(n_end):
        name = endmember_names[em_idx]
        if not _name_in_list(name, MARS_RARE_MINERALS):
            continue
        # Dominant / primary mixture endmembers are not "rare" secondaries
        mean_ab = float(mean_ab_per_em[em_idx]) if np.isfinite(mean_ab_per_em[em_idx]) else 0.0
        if mean_ab >= RARE_PRIMARY_MEAN_ABUND_EXCLUDE:
            print(f"  Strategy 2 skip {name}: mean_abund={mean_ab:.3f} "
                  f">= {RARE_PRIMARY_MEAN_ABUND_EXCLUDE} (primary EM, not rare)")
            continue
        cand = (
            finite_ab
            & (dom_map != em_idx)
            & (abundance_map[:, :, em_idx] > MINOR_ABUND_THRESH)
            & (max_ab_map < 0.85)
        )
        if spm_map is not None:
            # Require a related SPM hotspot so residual unmixing noise is not reported
            fam = mineral_family(name)
            spm_ok = np.zeros((H, W), dtype=bool)
            if fam in ("kieserite", "gypsum", "epsomite", "bassanite", "alunite", "jarosite"):
                for sn in ("SINDEX2", "BD2100_2", "BD1900_2"):
                    if sn in spm_idx:
                        col = spm_map[:, :, spm_idx[sn]]
                        thr = SPM_DETECT_THRESH.get(sn, 0.01)
                        spm_ok |= np.isfinite(col) & (col >= thr)
            elif fam in ("calcite", "dolomite", "magnesite", "siderite"):
                if "BD2500_2" in spm_idx:
                    col = spm_map[:, :, spm_idx["BD2500_2"]]
                    spm_ok |= np.isfinite(col) & (col >= SPM_DETECT_THRESH["BD2500_2"])
            elif fam in ("prehnite", "epidote", "serpentine", "antigorite",
                         "lizardite", "chrysotile", "opal", "analcime"):
                for sn in ("BD2210_2", "BD2290", "BD2300"):
                    if sn in spm_idx:
                        col = spm_map[:, :, spm_idx[sn]]
                        spm_ok |= np.isfinite(col) & (col >= SPM_DETECT_THRESH[sn])
            else:
                spm_ok[:] = True
            # If the related SPM channels were sanitized to all-NaN, skip family
            if not spm_ok.any() and fam in (
                "kieserite", "gypsum", "epsomite", "calcite", "dolomite"
            ):
                continue
            if spm_ok.any():
                cand &= spm_ok

        ys, xs = np.where(cand)
        for r, c in zip(ys.tolist(), xs.tolist()):
            minor_detections.append({
                "pixel": [int(r), int(c)],
                "mineral": name,
                "abundance": float(abundance_map[r, c, em_idx]),
                "dominant": endmember_names[int(dom_map[r, c])],
                "source": "abundance_secondary_rare",
            })

    minor_by_mineral: Dict[str, int] = {}
    for d in minor_detections:
        minor_by_mineral[d["mineral"]] = minor_by_mineral.get(d["mineral"], 0) + 1
    print(f"  Strategy 2: {len(minor_detections)} rare secondary-abundance detections "
          f"(thresh>{MINOR_ABUND_THRESH}, max_dom<0.85, SPM-gated) "
          f"across {len(minor_by_mineral)} minerals")

    rare_report = {
        "strategy1_noise_matches": rare_detections[:5000],
        "strategy2_secondary_abundance": minor_detections[:8000],
        "strategy1_count": len(rare_detections),
        "strategy2_count": len(minor_detections),
        "strategy2_by_mineral": minor_by_mineral,
        "thresholds": {
            "rare_angle": RARE_ANGLE_THRESH,
            "minor_abundance": MINOR_ABUND_THRESH,
            "max_dominant_for_secondary": 0.85,
            "spm_cosupport": True,
            "primary_mean_abund_exclude": RARE_PRIMARY_MEAN_ABUND_EXCLUDE,
        },
    }
    with open(os.path.join(output_dir, "rare_mineral_detections.json"), "w") as f:
        json.dump(rare_report, f, indent=2)

    ctx["rare_report"] = rare_report
    return ctx


# -------------------------------------------------------------
# Fix F — Mineral presence proof (SPM + diagnostic CR + library)
# -------------------------------------------------------------
def _library_to_proof_groups(lib_name: str, spm_name: str) -> List[str]:
    """Mineral groups a cluster label claims (for map↔proof gating)."""
    groups: List[str] = []
    if spm_name and spm_name not in ("Unknown", "Unresolved", ""):
        groups.append(str(spm_name))
    lf = (lib_name or "").lower()
    if any(x in lf for x in ("jarosite", "gypsum", "alunite", "kieserite",
                             "szomolnokite", "epsomite", "bassanite", "copiapite")):
        groups.append("Sulfates")
        if any(x in lf for x in ("kieserite", "szomolnokite")):
            groups.append("Monohydrated Sulfates")
    if any(x in lf for x in ("calcite", "dolomite", "magnesite", "siderite")):
        groups.append("Carbonates")
    if any(x in lf for x in ("olivine", "forsterite", "fayalite")):
        groups.append("Olivine")
    if any(x in lf for x in ("diopside", "augite")):
        groups.append("High-Ca Pyroxene")
    if any(x in lf for x in ("enstatite", "hypersthene", "bronzite", "pigeonite")):
        groups.append("Low-Ca Pyroxene")
    # unique preserve order
    out: List[str] = []
    for g in groups:
        if g not in out:
            out.append(g)
    return out


def apply_map_proof_consistency(ctx: dict) -> dict:
    """
    Demote cluster library_strong / open-USGS names when that mineral group
    is not PRESENT in mineral_proof. Display falls back to SPM group or
    Unresolved — proof remains the scientific authority.
    """
    proof = ctx.get("mineral_proof") or {}
    present = {
        g for g, e in (proof.get("minerals") or {}).items() if e.get("present")
    }
    matches = ctx.get("cluster_mineral_matches") or {}
    if not matches:
        return ctx
    n_demoted = 0
    for cl_s, m in matches.items():
        basis = str(m.get("label_basis") or "")
        auth = str(m.get("authoritative_label") or "")
        if auth in ("Unresolved", "", "Unknown"):
            continue
        if basis not in (
            "library_strong", "library_open_usgs", "library_only",
            "hybrid_strong_lib", "hybrid_moderate",
        ):
            continue
        claimed = _library_to_proof_groups(
            m.get("library_match") or auth, m.get("spm_name") or ""
        )
        if not claimed:
            continue
        if any(g in present for g in claimed):
            continue
        spm_g = m.get("spm_name") or "Unknown"
        m["map_proof_prior_label"] = auth
        m["map_proof_prior_basis"] = basis
        m["authoritative_label"] = "Unresolved"
        m["display_label"] = (
            spm_g if spm_g not in ("Unknown", "", None) else "Unresolved"
        )
        m["label_basis"] = "map_proof_demoted_not_present"
        m["match_quality"] = "WEAK"
        m["map_proof_note"] = (
            f"demoted {auth}: claimed {claimed} not in PRESENT "
            f"{sorted(present)}"
        )
        n_demoted += 1
        print(
            f"  Map-proof: cluster {cl_s} demoted {auth} -> Unresolved "
            f"(not PRESENT; spm={spm_g})"
        )
    ctx["cluster_mineral_matches"] = matches
    ctx["map_proof_demotions"] = int(n_demoted)
    if n_demoted:
        out_dir = ctx.get("output_dir")
        if out_dir:
            try:
                with open(os.path.join(out_dir, "cluster_names.json"), "w") as f:
                    json.dump(matches, f, indent=2)
                purity = {}
                for cl_s, match in matches.items():
                    purity[cl_s] = {
                        "n_pixels": match.get("n_pixels"),
                        "spectral_angle_rad": match.get("spectral_angle_rad"),
                        "sam_margin_rad": match.get("sam_margin_rad"),
                        "lib_spm_agree": match.get("lib_spm_agree"),
                        "spm_name": match.get("spm_name"),
                        "library_match": match.get("library_match"),
                        "authoritative_label": match.get("authoritative_label"),
                        "display_label": match.get("display_label"),
                        "label_basis": match.get("label_basis"),
                        "map_proof_note": match.get("map_proof_note"),
                    }
                with open(os.path.join(out_dir, "cluster_purity.json"), "w") as f:
                    json.dump(purity, f, indent=2)
            except Exception as e:
                print(f"  [WARN] map_proof rewrite failed: {type(e).__name__}: {e}")
        # Keep report cluster_purity in sync if already built
        if ctx.get("cluster_purity") is not None:
            ctx["cluster_purity"] = {
                cl_s: {
                    "n_pixels": m.get("n_pixels"),
                    "authoritative_label": m.get("authoritative_label"),
                    "display_label": m.get("display_label"),
                    "label_basis": m.get("label_basis"),
                }
                for cl_s, m in matches.items()
            }
    return ctx


def _demote_proof_mineral(proof: dict, group: str, reason: str) -> None:
    """Flip PRESENT → CANDIDATE and refresh summary line (ice competition)."""
    minerals = proof.get("minerals") or {}
    ent = minerals.get(group) or {}
    if not ent:
        return
    ent["present"] = False
    ent.setdefault("evidence", []).append(reason)
    spm = ent.get("spm_index") or "?"
    proof["summary"] = [
        s for s in proof.get("summary") or []
        if not s.startswith(f"{group} ")
    ]
    proof["summary"].append(
        f"{group} ({spm}): CANDIDATE/WEAK — "
        f"{'; '.join(ent.get('evidence') or [])}"
    )


def _apply_ice_hydrate_competition(proof: dict) -> dict:
    """
    Compete ice vs hydrate/olivine/mono-sulfate after PRESENT decisions.

    Global ice-proof path (polar or CRISM_ALLOW_ICE_PROOF):
    - Hydrate: demote if ice depth > d1.93 + 0.01
    - Olivine: demote if ice depth > d1.05 + 0.01 AND any of
      thin hot−bland, soft_proof_flag, or tied SAM (margin < SAM_MARGIN_STRONG)
    - Mono-sulfate: demote if family SAM > 0.10 and ice PRESENT
    """
    notes: List[str] = []
    minerals = proof.get("minerals") or {}
    ice_groups = ("Water Ice", "CO2 Ice")
    ice_present = []
    best_ice_d = -1.0
    best_ice_key = None
    best_ice_group = None
    for g in ice_groups:
        ent = minerals.get(g) or {}
        if not ent.get("present"):
            continue
        ice_present.append(g)
        dk = ent.get("proof_depth_key") or SPM_PROOF_DEPTH_KEY.get(
            ent.get("spm_index") or "", ""
        )
        d = float((ent.get("band_depths_hot") or {}).get(dk, float("nan")))
        if np.isfinite(d) and d > best_ice_d:
            best_ice_d = float(d)
            best_ice_key = dk
            best_ice_group = g

    audit = {
        "ice_present": ice_present,
        "best_ice_depth": None if best_ice_d < 0 else best_ice_d,
        "best_ice_key": best_ice_key,
        "best_ice_group": best_ice_group,
        "notes": notes,
        "demotions": [],
    }
    if not ice_present or best_ice_d < 0:
        notes.append("no_ice_PRESENT_or_no_ice_depth — no demotion")
        return audit

    hyd = minerals.get("Hydrated Minerals") or {}
    if hyd.get("present"):
        d193 = float((hyd.get("band_depths_hot") or {}).get("d1.93", float("nan")))
        if np.isfinite(d193) and best_ice_d > d193 + 0.010:
            _demote_proof_mineral(
                proof,
                "Hydrated Minerals",
                f"demoted_ice_dominates_hydrate {best_ice_group} "
                f"{best_ice_key}={best_ice_d:.4f} > d1.93={d193:.4f}+0.01",
            )
            notes.append(
                f"Hydrated Minerals demoted — ice depth wins "
                f"({best_ice_d:.4f} vs d1.93={d193:.4f})"
            )
            audit["demotions"].append("Hydrated Minerals")
        else:
            notes.append(
                f"hydrate kept PRESENT alongside ice "
                f"(ice={best_ice_d:.4f}, d1.93="
                f"{'nan' if not np.isfinite(d193) else f'{d193:.4f}'})"
            )

    ol = minerals.get("Olivine") or {}
    if ol.get("present"):
        d105 = float((ol.get("band_depths_hot") or {}).get("d1.05", float("nan")))
        hmb = ol.get("olivine_hot_minus_bland")
        if hmb is None:
            db105 = float((ol.get("band_depths_bland") or {}).get("d1.05", float("nan")))
            if np.isfinite(d105) and np.isfinite(db105):
                hmb = float(d105 - db105)
        thin_hmb = (
            hmb is None
            or (isinstance(hmb, float) and not np.isfinite(hmb))
            or float(hmb) < PROOF_HOT_MINUS_BLAND
        )
        sam_m = float(ol.get("sam_margin") or 0.0)
        tied_sam = sam_m < SAM_MARGIN_STRONG
        soft_flag = bool(ol.get("soft_polar_flag") or ol.get("soft_proof_flag"))
        # Soft evidence is enough once ice beats d1.05 — do not let bland-mask
        # luck alone keep PRESENT (v3 failure mode).
        soft_evidence = thin_hmb or soft_flag or tied_sam
        ice_beats_ol = np.isfinite(d105) and best_ice_d > d105 + 0.010
        if soft_evidence and ice_beats_ol:
            ol["soft_polar_flag"] = True
            ol["soft_proof_flag"] = True
            hmb_s = (
                "nan"
                if hmb is None or not np.isfinite(float(hmb))
                else f"{float(hmb):.4f}"
            )
            why = []
            if thin_hmb:
                why.append(f"thin_hmb={hmb_s}<{PROOF_HOT_MINUS_BLAND}")
            if tied_sam:
                why.append(f"tied_sam_margin={sam_m:.4f}<{SAM_MARGIN_STRONG}")
            if soft_flag and not thin_hmb and not tied_sam:
                why.append("soft_proof_flag")
            why_s = ";".join(why) if why else "soft"
            _demote_proof_mineral(
                proof,
                "Olivine",
                (
                    f"demoted_soft_olivine_ice_competes {best_ice_key}="
                    f"{best_ice_d:.4f} > d1.05={d105:.4f}+0.01 ({why_s})"
                ),
            )
            notes.append(
                f"Olivine demoted — soft evidence + ice depth > d1.05 ({why_s})"
            )
            audit["demotions"].append("Olivine")
        elif ice_beats_ol:
            ol.setdefault("evidence", []).append(
                f"soft_olivine_ice_depth_competes {best_ice_key}="
                f"{best_ice_d:.4f} vs d1.05={d105:.4f}"
            )
            ol["soft_polar_flag"] = True
            ol["soft_proof_flag"] = True
            notes.append(
                "Olivine soft-flagged — ice depth competes with d1.05 "
                "(no soft/tied/thin evidence; PRESENT kept)"
            )
        elif soft_flag or tied_sam:
            notes.append(
                "Olivine soft evidence set but ice depth does not beat "
                "d1.05+0.01 — PRESENT kept"
            )

    mono = minerals.get("Monohydrated Sulfates") or {}
    if mono.get("present"):
        sam = float(mono.get("best_sam") or 0.0)
        weak_sam = sam > 0.10
        if weak_sam:
            mono["soft_polar_flag"] = True
            mono["soft_proof_flag"] = True
            _demote_proof_mineral(
                proof,
                "Monohydrated Sulfates",
                f"demoted_soft_mono_sulfate_ice_context "
                f"family_sam={sam:.4f}>0.10 ice_PRESENT",
            )
            notes.append(
                f"Monohydrated Sulfates demoted — family SAM={sam:.4f}>0.10 "
                "with ice PRESENT"
            )
            audit["demotions"].append("Monohydrated Sulfates")
        elif mono.get("soft_polar_flag") or mono.get("soft_proof_flag"):
            notes.append(
                "Monohydrated Sulfates soft_proof_flag but SAM<=0.10 — PRESENT kept"
            )
    return audit


def _build_present_evidence_and_clay_sensitivity(proof: dict) -> Tuple[dict, dict]:
    """
    Compact PRESENT evidence for reports + log-only clay hard-floor (0.008) check.
    Does not change PRESENT decisions.
    """
    evidence: Dict[str, dict] = {}
    for group, ent in (proof.get("minerals") or {}).items():
        if not ent.get("present"):
            continue
        depth_key = ent.get("proof_depth_key")
        depths = ent.get("band_depths_hot") or {}
        depths_b = ent.get("band_depths_bland") or {}
        d_val = float("nan")
        db_val = float("nan")
        if depth_key and depth_key != "sulfate_multi" and depth_key in depths:
            d_val = float(depths.get(depth_key, float("nan")))
            db_val = float(depths_b.get(depth_key, float("nan")))
        elif depth_key == "sulfate_multi":
            d_val = float(ent.get("sulfate_best_depth", float("nan")))
            depth_key = ent.get("sulfate_best_depth_key") or "sulfate_multi"
        hot_minus_bland = (
            float(d_val - db_val)
            if np.isfinite(d_val) and np.isfinite(db_val)
            else float("nan")
        )
        evidence[group] = {
            "spm_index": ent.get("spm_index"),
            "n_hot": int(ent.get("n_hot_pixels") or 0),
            "depth_key": depth_key,
            "depth": None if not np.isfinite(d_val) else float(d_val),
            "depth_floor": ent.get("proof_depth_floor"),
            "hot_minus_bland": (
                None if not np.isfinite(hot_minus_bland) else float(hot_minus_bland)
            ),
            "library": ent.get("best_library"),
            "sam": ent.get("best_sam"),
            "present": True,
        }

    clay_rows = {}
    for spm_name in sorted(CLAY_BD_SPM):
        group = SPM_MINERAL_MAP.get(spm_name, spm_name)
        ent = (proof.get("minerals") or {}).get(group) or {}
        depth_key = SPM_PROOF_DEPTH_KEY.get(spm_name)
        d = float((ent.get("band_depths_hot") or {}).get(depth_key, float("nan")))
        present_now = bool(ent.get("present"))
        survives_hard = bool(
            present_now and np.isfinite(d) and d >= float(PROOF_DEPTH_BD)
        )
        clay_rows[group] = {
            "spm_index": spm_name,
            "depth_key": depth_key,
            "depth": None if not np.isfinite(d) else float(d),
            "present_at_adaptive_floor": present_now,
            "survives_hard_floor_008": survives_hard,
            "would_drop_at_008": bool(present_now and not survives_hard),
            "adaptive_floor": float(PROOF_DEPTH_BD_MIN),
            "hard_floor": float(PROOF_DEPTH_BD),
        }
    clay_sens = {
        "note": (
            "Log-only sensitivity: which clay PRESENT would drop if floor were "
            f"{PROOF_DEPTH_BD:.3f} instead of adaptive {PROOF_DEPTH_BD_MIN:.3f}. "
            "Does not change PRESENT."
        ),
        "hard_floor": float(PROOF_DEPTH_BD),
        "adaptive_floor": float(PROOF_DEPTH_BD_MIN),
        "minerals": clay_rows,
        "n_would_drop": int(sum(1 for v in clay_rows.values() if v["would_drop_at_008"])),
    }
    return evidence, clay_sens


def write_mineral_proof(ctx: dict) -> dict:
    """
    Evidence package proving which minerals are present in the scene.

    For each Fix-B passing SPM index: adaptive hotspot map stats, continuum
    band depths on restored diagnostic bands, library SAM top-5 on mean CR,
    overlap with HDBSCAN clusters. Writes mineral_proof.json + fig7.
    """
    print("\n" + "-" * 60)
    print(" STAGE 7b — Mineral Presence Proof")
    print("-" * 60)

    output_dir = ctx["output_dir"]
    waves = ctx["waves"]
    spm_map = ctx["spm_map"]
    valid = ctx["valid_pixel_mask"]
    cube_cr = ctx["cube_cr"]
    lib_cr = ctx["lib_spectra_cr"]
    lib_names = ctx["lib_names"]
    labels = ctx["labels"]
    valid_coords = ctx["valid_coords_all"]
    mineral_bad = ctx.get("mineral_bad_mask", build_mineral_bad_mask(waves))
    mineral_good = ~mineral_bad
    sam_mask = ctx.get("sam_mask", mineral_good)
    gate_info = ctx.get("spm_gate_info") or {}
    spm_passed = ctx.get("spm_passed") or []
    ice_proof_on = bool(ctx.get("ice_proof_enabled"))
    ice_candidates = list(ctx.get("ice_proof_candidates") or [])
    ice_gates = ctx.get("ice_proof_gates") or {}
    # Proof loop: clustering-passed SPM + polar ice proof candidates
    proof_spm_names = list(spm_passed)
    for iname in ice_candidates:
        if iname not in proof_spm_names:
            proof_spm_names.append(iname)
    cluster_matches = ctx.get("cluster_mineral_matches") or {}
    H, W = ctx["H"], ctx["W"]

    proof = {
        "scene": ctx.get("sname"),
        "cr_source": ctx.get("cr_source", "unknown"),
        "diagnostic_bands_restored": True,
        "spm_passed": list(spm_passed),
        "ice_proof_enabled": ice_proof_on,
        "ice_proof_candidates": list(ice_candidates),
        "minerals": {},
        "summary": [],
    }

    # Diagnostic wavelength finiteness (proof that Fix A worked)
    diag_finite = {}
    for w in (1.054, 1.330, 1.810, 1.930, 2.120, 2.210, 2.300):
        bi = int(np.argmin(np.abs(waves - w)))
        frac = float(np.mean(np.isfinite(cube_cr[:, :, bi][valid]))) if valid.any() else 0.0
        diag_finite[f"{w:.3f}"] = {"band": bi, "finite_frac": frac}
    proof["cr_diagnostic_finite_frac"] = diag_finite

    rng = np.random.default_rng(42)

    def _mean_cr(mask, nmax=4000):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None, 0
        if len(ys) > nmax:
            ii = rng.choice(len(ys), nmax, replace=False)
            ys, xs = ys[ii], xs[ii]
        return np.nanmean(cube_cr[ys, xs, :], axis=0), int(len(ys))

    # Bland reference: lowest 10% of max non-ice SPM among valid
    spm_max = nanmax_spm_exclude_ice(spm_map, axis=2)
    fin_max = valid & np.isfinite(spm_max) & (spm_max > -1e8)
    bland_mask = np.zeros_like(valid)
    if fin_max.any():
        thr_b = float(np.percentile(spm_max[fin_max], 10))
        bland_mask = fin_max & (spm_max <= thr_b)
    spec_bland, n_bland = _mean_cr(bland_mask)

    # Ice PRESENT sparsity cap from non-ice gate stats (universal, not Scene-4 knobs)
    n_valid_pix = int(np.sum(valid))
    non_ice_kept_fracs: List[float] = []
    for pname in spm_passed:
        if pname in ICE_SPM_NAMES:
            continue
        gi = gate_info.get(pname) or {}
        nk = gi.get("n_kept")
        if nk is None:
            continue
        non_ice_kept_fracs.append(float(nk) / max(n_valid_pix, 1))
    if non_ice_kept_fracs:
        ice_hot_frac_cap = max(
            ICE_PROOF_HOT_FRAC_FLOOR,
            ICE_PROOF_HOT_FRAC_MULT * float(np.median(non_ice_kept_fracs)),
        )
    else:
        ice_hot_frac_cap = ICE_PROOF_HOT_FRAC_FLOOR
    proof["ice_proof_sparsity"] = {
        "n_valid": n_valid_pix,
        "non_ice_kept_frac_median": (
            None if not non_ice_kept_fracs
            else float(np.median(non_ice_kept_fracs))
        ),
        "hot_frac_cap": float(ice_hot_frac_cap),
        "component_frac_cap": float(ice_hot_frac_cap),
    }
    if ice_proof_on:
        print(
            f"  Ice PRESENT sparsity cap: hot/max-comp ≤ {ice_hot_frac_cap:.3f} "
            f"of valid (floor={ICE_PROOF_HOT_FRAC_FLOOR}, "
            f"2×median non-ice gate)"
        )

    for name in proof_spm_names:
        j = SPM_NAMES.index(name)
        group = SPM_MINERAL_MAP[name]
        is_ice = name in ICE_SPM_NAMES
        if is_ice:
            ginfo = ice_gates.get(name) or (gate_info.get(name) or {}).get(
                "ice_proof_gate"
            ) or {}
        else:
            ginfo = gate_info.get(name, {})
        thr = float(ginfo.get("thr_adaptive", SPM_DETECT_THRESH[name]))
        hot = valid & np.isfinite(spm_map[:, :, j]) & (spm_map[:, :, j] >= thr)
        # Prefer gated component mask already in spm_active if single-index
        n_hot = int(hot.sum())
        spec_hot, n_used = _mean_cr(hot)
        entry = {
            "spm_index": name,
            "mineral_group": group,
            "thr_adaptive": thr,
            "n_hot_pixels": n_hot,
            "gate": ginfo,
            "present": False,
            "evidence": [],
            "ice_proof_only": bool(is_ice),
        }
        if spec_hot is None or n_hot < SPM_MIN_COMPONENT:
            entry["evidence"].append("insufficient_hot_pixels")
            proof["minerals"][group] = entry
            continue

        # Band depths on restored CR (explicit keys + fallbacks for dead shoulders)
        depths = compute_proof_depths(spec_hot, waves, mineral_bad)
        if spec_bland is not None:
            depths_bland = compute_proof_depths(spec_bland, waves, mineral_bad)
        else:
            depths_bland = {k: float("nan") for k in PROOF_DEPTH_DEFS}
        entry["band_depths_hot"] = depths
        entry["band_depths_bland"] = depths_bland

        # Family-restricted diagnostic SAM (avoids Quartz/Hematite continuum traps)
        diag_m = ctx.get("diag_sam_mask")
        if diag_m is None:
            diag_m = build_diagnostic_sam_mask(waves, mineral_good)
        group_name = SPM_MINERAL_MAP.get(name, name)
        topk, topk_global, sam_mode = match_library_family_then_global(
            spec_hot, lib_cr, lib_names, mineral_good,
            spm_group=group_name, k=5, sam_mask=diag_m,
        )
        entry["library_top5"] = [
            {"name": n, "angle": float(a), "family": mineral_family(n)}
            for n, a, _i in topk
        ]
        entry["library_top5_global"] = [
            {"name": n, "angle": float(a), "family": mineral_family(n)}
            for n, a, _i in (topk_global or [])[:5]
        ]
        entry["sam_mode"] = sam_mode
        if topk:
            entry["best_library"] = topk[0][0]
            entry["best_sam"] = float(topk[0][1])
            entry["sam_margin"] = (
                float(topk[1][1] - topk[0][1]) if len(topk) > 1 else 1.0
            )

        # Cluster overlap
        cl_overlap = {}
        hot_flat = hot[valid]
        for cl_s, match in cluster_matches.items():
            cl = int(cl_s)
            cl_mask = labels == cl
            inter = int(np.sum(cl_mask & hot_flat))
            if inter > 0:
                cl_overlap[cl_s] = {
                    "n_overlap": inter,
                    "frac_of_cluster": inter / max(int(cl_mask.sum()), 1),
                    "label": match.get("authoritative_label"),
                    "display_label": match.get("display_label") or match.get(
                        "authoritative_label"
                    ),
                }
        entry["cluster_overlap"] = cl_overlap

        # Presence decision: require strong spectral evidence (not cluster overlap alone)
        reasons = []
        if n_hot >= SPM_MIN_COMPONENT:
            reasons.append(f"adaptive_SPM_hotspot n={n_hot} thr={thr:.4f}")
        depth_hits = []
        library_hits = []
        depth_key = SPM_PROOF_DEPTH_KEY.get(name)
        # group_name already set above for family SAM

        if name == "OLINDEX3":
            d105 = depths.get("d1.05", float("nan"))
            db105 = depths_bland.get("d1.05", float("nan"))
            if np.isfinite(d105) and d105 >= PROOF_DEPTH_OLIVINE:
                depth_hits.append(f"CR_depth_1.05={d105:.4f}")
            if (
                np.isfinite(d105) and np.isfinite(db105)
                and d105 >= PROOF_DEPTH_OLIVINE
                and d105 > db105 + PROOF_HOT_MINUS_BLAND
            ):
                depth_hits.append(f"depth_1.05_hot>bland ({d105:.4f}>{db105:.4f})")
            if (
                topk
                and topk[0][1] <= FAMILY_SAM_HIT_MAX
                and mineral_family(topk[0][0]) in ("olivine", "forsterite", "fayalite")
            ):
                library_hits.append(f"library_best={topk[0][0]} SAM={topk[0][1]:.4f}")
            elif (
                topk
                and topk[0][1] <= FAMILY_SAM_HIT_MAX
                and _lib_spm_family_agree(topk[0][0], "Olivine")
            ):
                library_hits.append(f"library_compatible={topk[0][0]}")
        elif name in ("HCPINDEX2", "LCPINDEX2"):
            d212 = depths.get("d2.12", float("nan"))
            db212 = depths_bland.get("d2.12", float("nan"))
            if np.isfinite(d212) and d212 >= PROOF_DEPTH_PYROXENE:
                depth_hits.append(f"CR_depth_2.12={d212:.4f}")
            if (
                np.isfinite(d212) and np.isfinite(db212)
                and d212 >= PROOF_DEPTH_PYROXENE
                and d212 > db212 + PROOF_HOT_MINUS_BLAND
            ):
                depth_hits.append(f"d2.12_hot>bland")
            if topk and topk[0][1] <= FAMILY_SAM_HIT_MAX and (
                "pyroxene" in mineral_family(topk[0][0])
                or mineral_family(topk[0][0]) in (
                    "diopside", "augite", "enstatite", "hypersthene",
                    "bronzite", "pigeonite",
                )
            ):
                library_hits.append(f"library_best={topk[0][0]}")
        elif name == "SINDEX2":
            # SINDEX2 is continuum-convexity — require sulfate-family CR depths,
            # not d2.10 alone (Scene1: huge SINDEX hotspot with d2.10≈0).
            entry["proof_depth_key"] = "sulfate_multi"
            best_sulf_key = None
            best_sulf_d = -1.0
            floor = float(PROOF_DEPTH_BD_MIN)  # sulfate: min floor + hot>bland
            for sk in SULFATE_PROOF_DEPTH_KEYS:
                d = depths.get(sk, float("nan"))
                db = depths_bland.get(sk, float("nan"))
                if np.isfinite(d) and d >= floor:
                    depth_hits.append(f"CR_depth_{sk}={d:.4f}")
                    if d > best_sulf_d:
                        best_sulf_d = float(d)
                        best_sulf_key = sk
                if (
                    np.isfinite(d) and np.isfinite(db)
                    and d >= floor
                    and d > db + PROOF_HOT_MINUS_BLAND
                ):
                    depth_hits.append(f"{sk}_hot>bland")
            if best_sulf_key:
                entry["sulfate_best_depth_key"] = best_sulf_key
                entry["sulfate_best_depth"] = best_sulf_d
            # Pure sulfate-family library only (skip mixtures for corroboration)
            pure_hit = None
            for n, a, _i in (topk or []):
                if is_mixture_library_name(n):
                    continue
                fam = mineral_family(n)
                if (
                    a <= FAMILY_SAM_HIT_MAX
                    and (
                        fam in SULFATE_PURE_FAMILIES
                        or _lib_spm_family_agree(n, "Sulfates")
                    )
                ):
                    pure_hit = (n, a)
                    break
            if pure_hit:
                library_hits.append(
                    f"library_compatible={pure_hit[0]} SAM={pure_hit[1]:.4f}"
                )
            elif topk and is_mixture_library_name(topk[0][0]):
                reasons.append(
                    f"sulfate_mixture_library_skipped={topk[0][0]}"
                )
        elif depth_key is not None:
            d = depths.get(depth_key, float("nan"))
            db = depths_bland.get(depth_key, float("nan"))
            floor = adaptive_bd_depth_floor(name, db)
            entry["proof_depth_floor"] = float(floor)
            if np.isfinite(d) and d >= floor:
                depth_hits.append(f"CR_depth_{depth_key}={d:.4f}")
            if (
                np.isfinite(d) and np.isfinite(db)
                and d >= floor
                and d > db + PROOF_HOT_MINUS_BLAND
            ):
                depth_hits.append(f"{depth_key}_hot>bland")
            # Family-restricted corroboration (preferred SAM already family-scoped)
            if (
                topk
                and topk[0][1] <= FAMILY_SAM_HIT_MAX
                and not is_mixture_library_name(topk[0][0])
                and (
                    _lib_spm_family_agree(topk[0][0], group_name)
                    or "family" in sam_mode
                )
            ):
                if _lib_spm_family_agree(topk[0][0], group_name):
                    library_hits.append(
                        f"library_compatible={topk[0][0]} SAM={topk[0][1]:.4f}"
                    )
                elif "family" in sam_mode:
                    library_hits.append(
                        f"library_family={topk[0][0]} SAM={topk[0][1]:.4f}"
                    )
            # Mono-sulfate: bland continuum already at floor → residual reject
            if name == "BD2100_2" and np.isfinite(db) and db >= floor:
                reasons.append(
                    f"bland_residual_reject {depth_key}_bland={db:.4f}>={floor}"
                )
                entry["bland_residual_reject"] = True
            # Carbonate co-evidence: supporting d2.30 hot>bland
            if name == "BD2500_2":
                d230 = depths.get("d2.30", float("nan"))
                db230 = depths_bland.get("d2.30", float("nan"))
                carb_support = (
                    np.isfinite(d230) and np.isfinite(db230)
                    and d230 >= floor
                    and d230 > db230 + PROOF_HOT_MINUS_BLAND
                )
                entry["carbonate_d230_support"] = bool(carb_support)
                if carb_support:
                    depth_hits.append(
                        f"carbonate_coevidence_d2.30={d230:.4f}>bland"
                    )

        reasons.extend(depth_hits)
        reasons.extend(library_hits)
        if cl_overlap:
            reasons.append(f"overlaps_clusters={list(cl_overlap.keys())}")

        abs_depth_ok = any(h.startswith("CR_depth_") for h in depth_hits)
        hot_bland_ok = any("hot>bland" in h for h in depth_hits)
        lib_ok = len(library_hits) >= 1

        # PRESENT rules (multi-evidence — stop single-band floor clears):
        if name in ICE_SPM_NAMES:
            # Polar ice proof: hotspot + contrast + CR depth + hot>bland
            # + Viviano-like sparsity (reject scene-wide ice index maps).
            # Library optional (no ice standards in current USGS pool).
            contrast_ok = bool(ginfo.get("contrast_ok", True))
            hot_frac = float(n_hot) / max(n_valid_pix, 1)
            max_comp = int(ginfo.get("max_component") or 0)
            max_comp_frac = float(max_comp) / max(n_valid_pix, 1)
            sparse_ok = (
                hot_frac <= ice_hot_frac_cap
                and max_comp_frac <= ice_hot_frac_cap
            )
            entry["ice_hot_frac"] = hot_frac
            entry["ice_max_component_frac"] = max_comp_frac
            entry["ice_sparsity_cap"] = float(ice_hot_frac_cap)
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and hot_bland_ok
                and contrast_ok
                and sparse_ok
            )
            if n_hot >= SPM_MIN_COMPONENT and not abs_depth_ok:
                reasons.append("ice_no_diagnostic_CR_depth")
            if abs_depth_ok and not hot_bland_ok:
                reasons.append("ice_hot_not_gt_bland")
            if not contrast_ok:
                reasons.append("ice_gate_low_contrast")
            if not sparse_ok:
                reasons.append(
                    f"ice_not_sparse hot_frac={hot_frac:.3f} "
                    f"max_comp_frac={max_comp_frac:.3f} "
                    f"cap={ice_hot_frac_cap:.3f}"
                )
            reasons.append("ice_clustering_excluded_proof_only")
        elif name == "OLINDEX3":
            entry["present"] = (
                n_hot >= max(SPM_MIN_COMPONENT * 5, 500)
                and abs_depth_ok
                and (lib_ok or hot_bland_ok)
            )
            # Polar audit: thin hot-bland is soft evidence (log-only; no kill)
            if ice_proof_on and entry["present"]:
                d105 = float(depths.get("d1.05", float("nan")))
                db105 = float(depths_bland.get("d1.05", float("nan")))
                hmb = (
                    d105 - db105
                    if np.isfinite(d105) and np.isfinite(db105)
                    else float("nan")
                )
                entry["olivine_hot_minus_bland"] = (
                    None if not np.isfinite(hmb) else float(hmb)
                )
                if not np.isfinite(hmb) or hmb < PROOF_HOT_MINUS_BLAND:
                    reasons.append(
                        "soft_olivine_thin_hot_bland_ice_context"
                        + (f" hmb={hmb:.4f}" if np.isfinite(hmb) else "")
                    )
                    entry["soft_polar_flag"] = True
                    entry["soft_proof_flag"] = True
                sam_m = float(entry.get("sam_margin") or 0.0)
                if sam_m < SAM_MARGIN_STRONG:
                    reasons.append(
                        f"soft_olivine_tied_sam_margin={sam_m:.4f}"
                    )
                    entry["soft_polar_flag"] = True
                    entry["soft_proof_flag"] = True
        elif name in ("HCPINDEX2", "LCPINDEX2"):
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and (lib_ok or hot_bland_ok)
            )
        elif name == "BD2500_2":
            # Carbonates: depth + co-evidence + carbonate-family library
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and bool(entry.get("carbonate_d230_support"))
                and lib_ok
            )
            if abs_depth_ok and not lib_ok:
                reasons.append("carbonate_needs_library_family")
        elif name == "BD2100_2":
            # Mono-sulfates: reject bland residual; require library family
            # AND a strong-enough family SAM (stops bland-mask churn from
            # clearing PRESENT on weak Kieserite-like hits ~0.16 rad).
            bland_reject = bool(entry.get("bland_residual_reject"))
            sam = float(entry.get("best_sam") or 1.0)
            sam_ok = bool(np.isfinite(sam) and sam <= SAM_STRONG_MAX)
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and hot_bland_ok
                and lib_ok
                and not bland_reject
                and sam_ok
            )
            if abs_depth_ok and not lib_ok and not bland_reject:
                reasons.append("mono_sulfate_needs_library_family")
            if lib_ok and not sam_ok:
                reasons.append(
                    f"mono_sulfate_weak_family_sam={sam:.4f}>{SAM_STRONG_MAX}"
                )
                entry["soft_proof_flag"] = True
            if ice_proof_on and entry.get("present"):
                if sam > 0.10:
                    reasons.append(
                        f"soft_mono_sulfate_weak_family_sam={sam:.4f}_ice_context"
                    )
                    entry["soft_polar_flag"] = True
                    entry["soft_proof_flag"] = True
        elif name == "SINDEX2":
            # Poly/general sulfates: multi-band CR + pure-family library
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and hot_bland_ok
                and lib_ok
            )
            if abs_depth_ok and not lib_ok:
                reasons.append("sulfate_needs_pure_library_family")
            if n_hot >= SPM_MIN_COMPONENT and not abs_depth_ok:
                reasons.append("sulfate_no_diagnostic_CR_depth")
        elif name == "BD1900_2":
            # Hydrate: if d1.93 has abs floor + hot>bland, allow PRESENT even when
            # d1.05 is larger — unless library actively contradicts hydrate.
            d105 = depths.get("d1.05", float("nan"))
            d193 = depths.get("d1.93", float("nan"))
            dominated = (
                np.isfinite(d105) and np.isfinite(d193)
                and d105 > d193 + 0.02
            )
            lib_contradicts = False
            if topk_global and not lib_ok:
                gbest = topk_global[0][0]
                gang = float(topk_global[0][1])
                gf = mineral_family(gbest)
                # Quartz/hematite continuum traps are NOT active hydrate contradictions
                if gf in ("quartz", "hematite"):
                    pass
                elif (
                    gang < SAM_SUSPICIOUS_TIGHT
                    and not _lib_spm_family_agree(gbest, group_name)
                    and gf in (
                        "enstatite", "hypersthene", "diopside", "augite",
                        "olivine", "forsterite", "fayalite", "pigeonite",
                    )
                ):
                    lib_contradicts = True
                    reasons.append(
                        f"hydrate_library_contradicts={gbest} SAM={gang:.4f}"
                    )
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and hot_bland_ok
                and not lib_contradicts
            )
            if dominated and entry["present"]:
                reasons.append("note_d1.05_larger_than_d1.93_but_d1.93_ok")
            elif dominated and not entry["present"]:
                reasons.append("hydrate_d1.05_dominates_without_library")
        else:
            # Other BD (Al-OH, Fe/Mg-OH, Fe/Mg phys): need abs depth + hot>bland
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT
                and abs_depth_ok
                and hot_bland_ok
            )
        entry["evidence"] = reasons
        entry["depth_hits"] = depth_hits
        entry["library_hits"] = library_hits
        if "proof_depth_key" not in entry:
            entry["proof_depth_key"] = depth_key
        proof["minerals"][group] = entry
        status = "PRESENT" if entry["present"] else "CANDIDATE/WEAK"
        proof["summary"].append(f"{group} ({name}): {status} — {'; '.join(reasons)}")
        print(f"  {status}: {group} via {name} | n_hot={n_hot} | {reasons}")

    # Polar ice vs hydrate / olivine competition (after all PRESENT decisions)
    if ice_proof_on:
        ice_audit = _apply_ice_hydrate_competition(proof)
        proof["ice_competition_audit"] = ice_audit
        if ice_audit.get("notes"):
            for note in ice_audit["notes"]:
                print(f"  Ice competition: {note}")
        try:
            with open(os.path.join(output_dir, "ice_competition_audit.json"), "w") as f:
                json.dump(_json_clean(ice_audit), f, indent=2, allow_nan=False,
                          default=_json_default)
        except Exception as e:
            print(f"  [WARN] ice_competition_audit write failed: {type(e).__name__}: {e}")

    # Collapse BD2290 / BD2300 double-count unless both exceed strong floor
    m229 = proof["minerals"].get("Fe/Mg-OH Phyllosilicates")
    m230 = proof["minerals"].get("Fe/Mg Phyllosilicates")
    if (
        m229 and m230
        and m229.get("present")
        and m230.get("present")
    ):
        d229 = float((m229.get("band_depths_hot") or {}).get("d2.29") or 0.0)
        d230 = float((m230.get("band_depths_hot") or {}).get("d2.30") or 0.0)
        both_strong = (
            d229 >= PROOF_DEPTH_BD_STRONG and d230 >= PROOF_DEPTH_BD_STRONG
            and (m229.get("library_hits") or m230.get("library_hits"))
        )
        if not both_strong:
            if d229 >= d230:
                m230["present"] = False
                m230.setdefault("evidence", []).append(
                    "collapsed_into_BD2290_FeMgOH"
                )
                print("  Collapse: Fe/Mg Phyllosilicates → CANDIDATE "
                      "(kept Fe/Mg-OH BD2290)")
            else:
                m229["present"] = False
                m229.setdefault("evidence", []).append(
                    "collapsed_into_BD2300_FeMgPhys"
                )
                print("  Collapse: Fe/Mg-OH → CANDIDATE "
                      "(kept Fe/Mg Phyllosilicates BD2300)")
            # Rebuild summary lines for these two groups
            proof["summary"] = [
                s for s in proof["summary"]
                if "Fe/Mg-OH Phyllosilicates" not in s
                and "Fe/Mg Phyllosilicates (" not in s
                and not s.startswith("Fe/Mg Phyllosilicates (BD2300)")
            ]
            for grp, ent, spm in (
                ("Fe/Mg-OH Phyllosilicates", m229, "BD2290"),
                ("Fe/Mg Phyllosilicates", m230, "BD2300"),
            ):
                st = "PRESENT" if ent.get("present") else "CANDIDATE/WEAK"
                proof["summary"].append(
                    f"{grp} ({spm}): {st} — {'; '.join(ent.get('evidence') or [])}"
                )

    # Also note minerals NOT present (failed gates)
    for name, ginfo in gate_info.items():
        group = SPM_MINERAL_MAP.get(name, name)
        if group in proof["minerals"]:
            continue
        if name in ICE_SPM_NAMES:
            # Mid-lat: keep silent ice_or_excluded. Polar: record honest CANDIDATE.
            if not ice_proof_on:
                continue
            ig = ice_gates.get(name) or ginfo.get("ice_proof_gate") or ginfo
            if not ig.get("passed") and not ig.get("proof_candidate"):
                proof["minerals"][group] = {
                    "spm_index": name,
                    "mineral_group": group,
                    "present": False,
                    "ice_proof_only": True,
                    "evidence": [
                        f"ice_proof_gate_failed: {ig.get('reason', ig)}",
                        "ice_clustering_excluded_proof_only",
                    ],
                }
                proof["summary"].append(
                    f"{group} ({name}): CANDIDATE/WEAK — ice_proof_gate_failed"
                )
            continue
        if not ginfo.get("passed", False):
            proof["minerals"][group] = {
                "spm_index": name,
                "mineral_group": group,
                "present": False,
                "evidence": [f"failed_adaptive_gate: {ginfo.get('reason', ginfo)}"],
            }

    # PRESENT evidence table + log-only clay hard-floor sensitivity (no PRESENT change)
    present_evidence, clay_sens = _build_present_evidence_and_clay_sensitivity(proof)
    proof["minerals_present_evidence"] = present_evidence
    proof["clay_floor_sensitivity"] = clay_sens
    proof["map_available"] = bool(ctx.get("map_available", True))
    if not proof["map_available"]:
        proof["map_note"] = (
            "HDBSCAN found 0 clusters; PRESENT claims are scene-level proof only "
            "(no spatial mineral-unit map)."
        )
    print("  PRESENT evidence (cite these — not names alone):")
    if present_evidence:
        for grp, ev in present_evidence.items():
            d = ev.get("depth")
            hmb = ev.get("hot_minus_bland")
            d_s = f"{d:.4f}" if d is not None else "nan"
            h_s = f"{hmb:.4f}" if hmb is not None else "nan"
            print(
                f"    {grp}: n_hot={ev.get('n_hot')} "
                f"{ev.get('depth_key')}={d_s} floor={ev.get('depth_floor')} "
                f"hot-bland={h_s} lib={ev.get('library')} SAM={ev.get('sam')}"
            )
    else:
        print("    (none)")
    if clay_sens.get("n_would_drop", 0):
        print(
            f"  Clay floor sensitivity (log-only): "
            f"{clay_sens['n_would_drop']} PRESENT clay(s) would drop at "
            f"hard floor {PROOF_DEPTH_BD:.3f}:"
        )
        for grp, row in clay_sens["minerals"].items():
            if row.get("would_drop_at_008"):
                print(
                    f"    would_drop: {grp} depth={row.get('depth')} "
                    f"(adaptive PRESENT, hard 0.008 fail)"
                )
    else:
        print(
            f"  Clay floor sensitivity (log-only): all adaptive clay PRESENT "
            f"also survive hard floor {PROOF_DEPTH_BD:.3f}"
        )
    with open(os.path.join(output_dir, "clay_floor_sensitivity.json"), "w") as f:
        json.dump(_json_clean(clay_sens), f, indent=2, allow_nan=False, default=_json_default)

    # Attach log-only sulfate CR audit (does not change PRESENT)
    audit = ctx.get("sulfate_depth_audit") or {}
    if audit:
        proof["sulfate_depth_audit"] = audit
        proof["sulfate_depth_audit_note"] = (
            "Cluster-mean CR depths for Jarosite/Sulfates units; "
            "log only — does not promote Sulfates to PRESENT"
        )

    with open(os.path.join(output_dir, "mineral_proof.json"), "w") as f:
        json.dump(_json_clean(proof), f, indent=2, allow_nan=False, default=_json_default)

    # Figure: OLINDEX map + hotspot + cluster overlay if olivine passed
    try:
        plt.rcParams.update(_DARK_RC)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        ol = spm_map[:, :, SPM_NAMES.index("OLINDEX3")].copy()
        ol[~valid] = np.nan
        im0 = axes[0].imshow(ol, cmap="viridis", interpolation="nearest")
        axes[0].set_title("OLINDEX3 (ratioed IMG)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        hot_ol = np.zeros((H, W), dtype=np.float32)
        if "OLINDEX3" in spm_passed:
            thr = float(gate_info["OLINDEX3"]["thr_adaptive"])
            hot_ol[valid & np.isfinite(ol) & (ol >= thr)] = 1.0
        axes[1].imshow(hot_ol, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        axes[1].set_title(
            f"Adaptive olivine hotspot\n(thr={gate_info.get('OLINDEX3', {}).get('thr_adaptive', float('nan')):.3f})"
        )
        lab_map = np.full((H, W), np.nan, dtype=np.float32)
        for i, (r, c) in enumerate(valid_coords):
            if labels[i] >= 0:
                lab_map[r, c] = labels[i]
        im2 = axes[2].imshow(lab_map, cmap="tab20", interpolation="nearest")
        title_c = "HDBSCAN clusters (SPM-active domain)"
        if int(np.nanmax(lab_map) if np.isfinite(lab_map).any() else -1) < 0:
            title_c = "HDBSCAN: no clusters (map unavailable)"
        axes[2].set_title(title_c)
        for ax in axes:
            ax.set_xlabel("Sample")
            ax.set_ylabel("Line")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig7_mineral_proof.png"), dpi=160)
        plt.close(fig)
        print(f"  Wrote fig7_mineral_proof.png and mineral_proof.json")
    except Exception as e:
        print(f"  [WARN] fig7 failed: {type(e).__name__}: {e}")

    ctx["mineral_proof"] = proof
    ctx["minerals_present_evidence"] = present_evidence
    ctx["clay_floor_sensitivity"] = clay_sens
    return ctx


# -------------------------------------------------------------
# VERDICT (clustering quality ≠ mineral presence)
# -------------------------------------------------------------
def compute_scene_verdict(ctx: dict) -> dict:
    """
    Build separate clustering_quality and mineral_detection_confidence scores.

    overall_verdict is driven primarily by mineral proof evidence, not by a
    failed DBCV (-1) which previously forced WEAK even when minerals were proven.
    """
    dbcv = ctx.get("dbcv")
    dbcv_status = ctx.get("dbcv_status", "unknown")
    n_clusters = int(ctx.get("n_clusters") or 0)
    labels = ctx.get("labels")
    n_valid = int(len(labels)) if labels is not None else 0
    n_noise = int((labels == -1).sum()) if labels is not None else 0
    noise_frac = n_noise / max(n_valid, 1)

    # --- clustering quality: DBCV primary; spectral gap can lift WEAK→MODERATE ---
    spectral_gap = ctx.get("cluster_spectral_gap")
    if dbcv is None:
        clustering_quality = "UNAVAILABLE"
        clustering_note = f"DBCV {dbcv_status}; not used to force WEAK"
    elif dbcv > 0.3:
        clustering_quality = "STRONG"
        clustering_note = f"DBCV={dbcv:.4f}"
    elif dbcv > 0.1:
        clustering_quality = "MODERATE"
        clustering_note = f"DBCV={dbcv:.4f}"
    else:
        clustering_quality = "WEAK"
        clustering_note = f"DBCV={dbcv:.4f}"

    if n_clusters <= 0:
        clustering_quality = "WEAK"
        clustering_note += "; no clusters"
    elif noise_frac > 0.85 and clustering_quality == "STRONG":
        clustering_quality = "MODERATE"
        clustering_note += f"; high noise_frac={noise_frac:.2f}"

    # Discovery separation: ≥3 modes with mean inter-cluster SAM gap
    if (
        clustering_quality == "WEAK"
        and n_clusters >= 3
        and spectral_gap is not None
        and float(spectral_gap) >= CLUSTER_INTER_SAM_GAP_MODERATE
    ):
        clustering_quality = "MODERATE"
        clustering_note += (
            f"; spectral_gap={float(spectral_gap):.4f}>="
            f"{CLUSTER_INTER_SAM_GAP_MODERATE}"
        )

    # --- mineral detection confidence from proof ---
    proof = ctx.get("mineral_proof") or {}
    minerals = proof.get("minerals") or {}
    present = [k for k, v in minerals.items() if v.get("present")]
    n_present = len(present)
    n_with_depth = 0
    n_with_library = 0
    for v in minerals.values():
        if not v.get("present"):
            continue
        if v.get("depth_hits"):
            n_with_depth += 1
        if v.get("library_hits"):
            n_with_library += 1

    # SAM margins from cluster matches (discriminative labeling)
    matches = ctx.get("cluster_mineral_matches") or {}
    strong_sam = 0
    tied_sam = 0
    tied_claimed = 0  # tied SAM but still printed a specimen name (bad)
    for m in matches.values():
        margin = float(m.get("sam_margin_rad") or m.get("sam_margin") or 0.0)
        basis = str(m.get("label_basis") or "")
        auth = str(m.get("authoritative_label") or "")
        is_tied = ("tied" in basis) or (margin < SAM_MARGIN_STRONG)
        if is_tied:
            tied_sam += 1
            # Honest Unresolved abstention is NOT a false claim — no score penalty
            if auth not in ("Unresolved", "Unknown", ""):
                tied_claimed += 1
        if margin >= SAM_MARGIN_STRONG and float(m.get("spectral_angle_rad") or 1.0) < SAM_STRONG_MAX:
            strong_sam += 1

    score = 0
    score += min(n_present, 3) * 2          # up to +6
    score += min(n_with_depth, 3)           # up to +3
    score += min(n_with_library, 2)        # up to +2
    score += min(strong_sam, 2)            # up to +2
    if tied_claimed and n_present:
        score -= 1
    if clustering_quality == "STRONG":
        score += 1
    elif clustering_quality == "WEAK":
        score -= 0  # do not punish mineral proof for weak DBCV

    if n_present == 0:
        mineral_confidence = "NONE"
        overall = "WEAK" if clustering_quality in ("WEAK", "UNAVAILABLE") else "MODERATE"
        overall_note = "no minerals passed proof gates"
    elif score >= 8 and n_with_depth >= 1 and n_with_library >= 1:
        # STRONG requires at least one family-agreeing library hit (no depth-only STRONG)
        mineral_confidence = "STRONG"
        overall = "STRONG"
        overall_note = (
            f"proof present={present}; depth_support={n_with_depth}; "
            f"library_support={n_with_library}"
        )
    elif score >= 4 and n_with_depth >= 1:
        mineral_confidence = "MODERATE"
        overall = "MODERATE"
        overall_note = (
            f"proof present={present}; depth_support={n_with_depth}; "
            f"library_support={n_with_library}"
            + ("; depth-only capped at MODERATE" if n_with_library < 1 else "")
        )
    else:
        mineral_confidence = "WEAK"
        overall = "WEAK"
        overall_note = (
            f"proof present={present} but evidence thin "
            f"(depth={n_with_depth}, library={n_with_library}, score={score})"
        )

    # Honest mapping: detections without spatial units cannot be overall STRONG
    map_available = bool(ctx.get("map_available", n_clusters >= 1))
    if (n_clusters <= 0 or not map_available) and overall == "STRONG":
        overall = "MODERATE"
        overall_note += (
            "; overall capped at MODERATE (no spatial clusters / map unavailable)"
        )

    out = {
        "overall_verdict": overall,
        "overall_note": overall_note,
        "map_available": map_available,
        "clustering_quality": clustering_quality,
        "clustering_note": clustering_note,
        "mineral_detection_confidence": mineral_confidence,
        "verdict_score": int(score),
        "n_minerals_present": n_present,
        "n_present_with_depth": n_with_depth,
        "n_present_with_library": n_with_library,
        "dbcv": None if dbcv is None else float(dbcv),
        "dbcv_status": dbcv_status,
        "noise_fraction": float(noise_frac),
        "n_clusters": n_clusters,
        "cluster_spectral_gap": (
            None if spectral_gap is None else float(spectral_gap)
        ),
        "n_mega_splits": int(ctx.get("n_mega_splits") or 0),
        "n_mega_splits_secondary": int(ctx.get("n_mega_splits_secondary") or 0),
    }
    ctx["verdict_detail"] = out
    return out


# -------------------------------------------------------------
# STAGE 7 — Validation & Figures
# -------------------------------------------------------------
def stage7_validation(ctx: dict) -> dict:
    print("\n" + "-" * 60)
    print(" STAGE 7 — Validation and Publication Figures")
    print("-" * 60)

    output_dir = ctx["output_dir"]
    labels = ctx["labels"]
    uncertainty = ctx["uncertainty"]
    spm_flat = ctx["spm_flat"]
    valid_coords_all = ctx["valid_coords_all"]
    label_map = ctx["label_map"]
    smoothed_label_map = ctx["smoothed_label_map"]
    uncertainty_map = ctx["uncertainty_map"]
    abundance_map = ctx["smoothed_abundance"]
    recon_error_map = ctx["recon_error_map"]
    cube_cr = ctx["cube_cr"]
    waves = ctx["waves"]
    lib_cr = ctx["lib_spectra_cr"]
    lib_names = ctx["lib_names"]
    cluster_matches = ctx["cluster_mineral_matches"]
    n_clusters = ctx["n_clusters"]
    dbcv = ctx["dbcv"]
    endmember_names = ctx["endmember_names"]
    n_end = ctx["n_endmembers"]
    valid = ctx["valid_pixel_mask"]
    H, W = ctx["H"], ctx["W"]
    sname = ctx["sname"]
    mean_max = ctx["mean_max_abundance"]

    plt.rcParams.update(_DARK_RC)

    # Tier 1 — SPM recall
    validation_results = {}
    for cl in range(n_clusters):
        cl_mask = labels == cl
        match = cluster_matches.get(str(cl), {})
        spm_name = match.get("spm_name", "Unknown")
        spm_idx = None
        for j, (name, mineral_group) in enumerate(SPM_MINERAL_MAP.items()):
            if mineral_group == spm_name:
                spm_idx = j
                break
        if spm_idx is None:
            continue
        spm_vals = spm_flat[:, spm_idx]
        finite = np.isfinite(spm_vals)
        if finite.sum() == 0:
            continue
        threshold = np.percentile(spm_vals[finite], 80)
        top_mask = finite & (spm_vals >= threshold)
        recall = float(np.sum(cl_mask & top_mask) / max(1, top_mask.sum()))
        validation_results[f"cluster_{cl}"] = {
            "spm_name": spm_name,
            "library_match": match.get("library_match"),
            "spectral_angle": match.get("spectral_angle_rad"),
            "spm_recall_top20": recall,
            "n_pixels": int(cl_mask.sum()),
        }

    # Tier 3 — uncertainty quality
    corr_spm_unc = None
    ratio = None
    corr_unc_recon = None
    spm_max = nanmax_spm_exclude_ice(spm_flat, axis=1, fill=np.nan)
    finite = np.isfinite(spm_max) & np.isfinite(uncertainty)
    if finite.sum() > 30:
        corr_spm_unc, _ = spearmanr(spm_max[finite], uncertainty[finite])
        corr_spm_unc = float(corr_spm_unc)
        print(f"  Spearman(uncertainty, SPM): {corr_spm_unc:.3f}")

    interior_mask = np.zeros(len(labels), dtype=bool)
    boundary_mask = np.zeros(len(labels), dtype=bool)
    for i, (r, c) in enumerate(valid_coords_all):
        if labels[i] == -1:
            continue
        is_boundary = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if label_map[nr, nc] != labels[i]:
                    is_boundary = True
                    break
        if is_boundary:
            boundary_mask[i] = True
        else:
            interior_mask[i] = True
    if interior_mask.any() and boundary_mask.any():
        ratio = float(np.mean(uncertainty[boundary_mask]) /
                      max(1e-6, np.mean(uncertainty[interior_mask])))
        print(f"  Boundary/Interior uncertainty ratio: {ratio:.2f}")

    recon_errors = recon_error_map[valid]
    finite = np.isfinite(recon_errors) & np.isfinite(uncertainty)
    if finite.sum() > 30:
        corr_unc_recon = float(np.corrcoef(uncertainty[finite], recon_errors[finite])[0, 1])
        print(f"  Pearson(uncertainty, recon_error): {corr_unc_recon:.3f}")

    # -- Figures ----------------------------------------------
    # Fig 1: classification + uncertainty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    n_colors = max(n_clusters, 1)
    cmap = plt.cm.get_cmap("tab20", max(n_colors, 2))
    show = smoothed_label_map.astype(float)
    show[smoothed_label_map < 0] = np.nan
    im1 = ax1.imshow(show, cmap=cmap, interpolation="nearest")
    ax1.set_title("Smoothed Mineral Classification")
    ax1.set_xlabel("Sample"); ax1.set_ylabel("Line")
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    im2 = ax2.imshow(uncertainty_map, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title("Cluster Uncertainty")
    ax2.set_xlabel("Sample"); ax2.set_ylabel("Line")
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_classification.png"), dpi=160)
    plt.close(fig)

    # Fig 2: abundance panels
    ncols = min(6, n_end)
    nrows = int(np.ceil(n_end / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_end:
            im = ax.imshow(abundance_map[:, :, i], cmap="viridis", vmin=0, vmax=1,
                           interpolation="nearest")
            ax.set_title(endmember_names[i][:18], fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Endmember Abundance Maps", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_abundance_panels.png"), dpi=160)
    plt.close(fig)

    # Fig 3: uncertainty validation scatters
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    m_spm = np.isfinite(spm_max) & np.isfinite(uncertainty)
    if m_spm.sum() > 10:
        axes[0].scatter(spm_max[m_spm], uncertainty[m_spm], s=4, alpha=0.35, c="#58a6ff")
    axes[0].set_xlabel("Max SPM"); axes[0].set_ylabel("Uncertainty")
    axes[0].set_title(f"Spearman={corr_spm_unc if corr_spm_unc is not None else 'n/a'}")
    axes[1].bar(["Interior", "Boundary"],
                [float(np.mean(uncertainty[interior_mask])) if interior_mask.any() else 0,
                 float(np.mean(uncertainty[boundary_mask])) if boundary_mask.any() else 0],
                color=["#3fb950", "#f85149"])
    axes[1].set_title(f"B/I ratio={ratio if ratio is not None else 'n/a'}")
    m_re = np.isfinite(recon_errors) & np.isfinite(uncertainty)
    if m_re.sum() > 10:
        axes[2].scatter(recon_errors[m_re], uncertainty[m_re], s=4, alpha=0.35, c="#d2a8ff")
    axes[2].set_xlabel("Recon MSE"); axes[2].set_ylabel("Uncertainty")
    axes[2].set_title(f"Pearson={corr_unc_recon if corr_unc_recon is not None else 'n/a'}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_uncertainty_validation.png"), dpi=160)
    plt.close(fig)

    # Fig 4: SPM vs cluster bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    if n_clusters > 0:
        means = []
        names = []
        for cl in range(n_clusters):
            cl_mask = labels == cl
            means.append(np.nanmean(spm_flat[cl_mask], axis=0))
            m = cluster_matches.get(str(cl), {})
            disp = m.get("display_label") or m.get("authoritative_label") or "?"
            names.append(f"C{cl}:{disp}"[:22])
        means = np.asarray(means)
        x = np.arange(N_SPM)
        width = 0.8 / max(n_clusters, 1)
        for cl in range(n_clusters):
            ax.bar(x + cl * width, means[cl], width=width, label=names[cl])
        ax.set_xticks(x + width * (n_clusters - 1) / 2)
        ax.set_xticklabels(SPM_NAMES, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=7)
    ax.set_ylabel("Mean SPM")
    ax.set_title("Cluster Mean SPM Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_spm_comparison.png"), dpi=160)
    plt.close(fig)

    # Fig 5: cluster mean spectra vs library match
    n_show = max(n_clusters, 1)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.4 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for cl in range(n_clusters):
        ax = axes[cl]
        cl_mask = labels == cl
        coords = valid_coords_all[cl_mask]
        mean_spec = np.nanmean(cube_cr[coords[:, 0], coords[:, 1], :], axis=0)
        ax.plot(waves, mean_spec, color="#58a6ff", lw=1.5, label="Cluster mean")
        match = cluster_matches.get(str(cl), {})
        lib_idx = match.get("library_index", -1)
        if isinstance(lib_idx, int) and 0 <= lib_idx < len(lib_cr):
            ax.plot(waves, lib_cr[lib_idx], color="#f0883e", lw=1.2, alpha=0.85,
                    label=f"Lib: {match.get('library_match', '?')}")
        ax.set_ylabel("CR refl.")
        disp = match.get("display_label") or match.get("authoritative_label") or "?"
        auth = match.get("authoritative_label") or "?"
        ax.set_title(
            f"Cluster {cl}: {disp}"
            + (f" (auth={auth})" if auth != disp else "")
            + f" | SAM={match.get('spectral_angle_rad', np.nan):.3f}"
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(waves[0], waves[-1])
    if n_clusters == 0:
        axes[0].text(0.5, 0.5, "No clusters found", ha="center", va="center",
                     transform=axes[0].transAxes)
    axes[-1].set_xlabel("Wavelength (µm)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_spectra_comparison.png"), dpi=160)
    plt.close(fig)

    # Fig 6: reconstruction error
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(recon_error_map, cmap="inferno", interpolation="nearest")
    ax.set_title("Unmixing Reconstruction Error (MSE)")
    ax.set_xlabel("Sample"); ax.set_ylabel("Line")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_recon_error.png"), dpi=160)
    plt.close(fig)

    # Provisional verdict before mineral_proof (recomputed after write_mineral_proof)
    vdetail = compute_scene_verdict(ctx)
    verdict = vdetail["overall_verdict"]
    rare = ctx.get("rare_report", {})
    H, W = ctx["H"], ctx["W"]
    report = {
        "scene": sname,
        "cube_shape": [int(H), int(W), int(ctx["B"])],
        "cube_source": ctx.get("cube_source"),
        "full_height": bool(H >= FULL_HEIGHT_MIN_LINES),
        "n_clusters": int(n_clusters),
        "n_noise_pixels": int((labels == -1).sum()),
        "n_mega_splits": int(ctx.get("n_mega_splits") or 0),
        "cluster_spectral_gap": (
            None if ctx.get("cluster_spectral_gap") is None
            else float(ctx["cluster_spectral_gap"])
        ),
        "n_endmembers": int(n_end),
        "dbcv_score": None if dbcv is None else float(dbcv),
        "dbcv_status": ctx.get("dbcv_status", "unknown"),
        "tier1_spm_accuracy": validation_results,
        "tier2_library_match": cluster_matches,
        "tier3_uncertainty": {
            "spearman_spm_unc": corr_spm_unc,
            "boundary_interior_ratio": ratio,
            "pearson_unc_recon": corr_unc_recon,
            "uncertainty_model": "0.45*HDBSCAN + 0.35*(1-max_abund) + 0.20*recon_error",
        },
        "physical_validity": {
            "abundances_positive": True,
            "row_sums_one": True,
            "mean_max_abundance": float(mean_max),
        },
        "rare_minerals": {
            "strategy1_count": rare.get("strategy1_count", 0),
            "strategy2_count": rare.get("strategy2_count", 0),
            "strategy2_by_mineral": rare.get("strategy2_by_mineral", {}),
            "thresholds": rare.get("thresholds", {}),
        },
        "spm_sanity": ctx.get("spm_notes", {}),
        "overall_verdict": verdict,
        "clustering_quality": vdetail["clustering_quality"],
        "mineral_detection_confidence": vdetail["mineral_detection_confidence"],
        "verdict_detail": vdetail,
        "labels_provisional": True,
        "map_available": bool(ctx.get("map_available", n_clusters >= 1)),
        "external_ground_truth": None,
        "disclaimer": (
            "overall_verdict is driven by mineral_proof evidence (SPM + CR depth / "
            "library SAM), not by DBCV alone. clustering_quality is reported separately. "
            "Mineral names remain provisional without external published geologic maps. "
            "Always cite minerals_present_evidence (n_hot, depth, hot-bland), not names alone."
        ),
        "endmember_names": endmember_names,
        "spm_indices": SPM_NAMES,
        "methodology": {
            "library": "USGS splib07 ASCIIdata_splib07b_cvCRISMjMTR3 (pre-convolved)",
            "srf_convolution": False,
            "channel_subset": "489 joined -> 438 L-sensor",
            "spm_reference": "Viviano-Beck et al. (2014)",
            "spm_mask": "relaxed hard-exclude for diagnostic bands",
            "endmember_selection": "SPM-co-supported seeds + unique-family FPS; prune unused",
            "encoder": "1D CNN + NT-Xent + SPM anchor + recon head (latent=128)",
            "clustering": (
                "Discovery-first HDBSCAN on SPM-active domain (K free); "
                "encoder latent + diagnostic CR primary; SPM soft/low-weight only "
                "(not hard mineral domains); leaf retry for finer modes; "
                "last-resort mega-split by competing SPM; family merge"
            ),
            "unmixing": "AbundanceNetwork + softmax ASC; library E frozen; use/dom regularizers",
            "labeling": (
                "Open-set USGS: strong global SAM can name outside SPM families "
                "(library_open_usgs); continuum traps blocked; Unresolved on tied SAM; "
                "SPM is context/proof screening, not discovery gate"
            ),
            "verdict_cap": "STRONG requires library_family corroboration; depth-only max MODERATE",
            "proof_rules": (
                "BD multi-evidence; clay adaptive floor; SINDEX multi-band sulfate CR "
                "(not d2.10 alone); carbonate needs lib+d2.30; mono-sulfate bland reject; "
                "Fe/Mg collapse; tied-SAM penalty only if specimen claimed"
            ),
            "smoothing": "SLIC votes on clustered pixels only (noise stays -1)",
            "spm_sources": "bland-ratioed denoised S4 primary; IMG fallback; absolute SPM gates + contrast check",
            "bland_ratio": True,
            "column_destripe": True,
            "edge_column_mask": EDGE_COL_MARGIN,
            "cr_source": ctx.get("cr_source", "bland_ratioed_S4"),
            "cube_source": ctx.get("cube_source"),
            "latent_dim": LATENT_DIM,
            "spm_active_clustering": True,
            "mega_cluster_split": True,
            "mega_split_max_rounds": MEGA_SPLIT_MAX_ROUNDS,
            "mega_secondary_frac": MEGA_SECONDARY_FRAC,
            "mega_secondary_pairs": [list(p) for p in MEGA_SECONDARY_PAIRS],
            "v6_freeze": (
                "CLUSTER_SPM/DIAG weights, noise-rescue, leaf bars, "
                "PROOF_DEPTH_*, SAM_MARGIN_STRONG, MEGA_CLUSTER_FRAC primary"
            ),
            "ice_proof": (
                "Polar/Scene4 or CRISM_ALLOW_ICE_PROOF: BD1500/BD1435 PRESENT "
                "screening only; ice stays out of SPM-active clustering domain; "
                "nanmax SPM fallbacks exclude ICE_SPM; SPM_CLUSTER_WEIGHT ice=0; "
                "soft olivine/mono demote when ice PRESENT competes; "
                "ice PRESENT requires contrast + sparsity vs non-ice gate median"
            ),
            "spm_cluster_channel_weights": dict(SPM_CLUSTER_WEIGHT),
            "ice_proof_sparsity_floor": ICE_PROOF_HOT_FRAC_FLOOR,
            "ice_proof_sparsity_mult": ICE_PROOF_HOT_FRAC_MULT,
            "ice_proof_enabled": bool(ctx.get("ice_proof_enabled")),
            "encoder_min_epochs_full": ENCODER_MIN_EPOCHS_FULL,
            "mineral_proof": True,
            "verdict": (
                "mineral_proof primary; clustering_quality from DBCV and/or "
                "inter-cluster spectral SAM gap; overall capped at MODERATE "
                "when map_available=false / n_clusters=0"
            ),
            "specimen_depth_gate": (
                "library_strong sulfate/carbonate/olivine names require "
                ">=2 diagnostic CR depths above PROOF_DEPTH_BD"
            ),
            "specimen_depth_demotions": int(
                ctx.get("specimen_depth_demotions") or 0
            ),
            "hdbscan_max_fit": HDBSCAN_MAX_FIT,
            "dbcv_max_samples": DBCV_MAX_SAMPLES,
            "hdbscan_mcs_scale": "fit_set",
            "cluster_feature_weights": {
                "spm": CLUSTER_SPM_WEIGHT,
                "diag": CLUSTER_DIAG_WEIGHT,
                "depth": float(
                    ctx.get("cluster_depth_weight_used", CLUSTER_DEPTH_WEIGHT)
                ),
                "depth_full": CLUSTER_DEPTH_WEIGHT,
                "depth_adapt": CLUSTER_DEPTH_WEIGHT_ADAPT,
                "xy": CLUSTER_XY_WEIGHT,
            },
            "cluster_depth_keys": list(
                ctx.get("cluster_depth_keys") or CLUSTER_DEPTH_KEYS
            ),
            "cluster_depth_used": bool(ctx.get("cluster_depth_used", True)),
            "cluster_depth_fallback": ctx.get("cluster_depth_fallback"),
            "cluster_depth_muted": list(ctx.get("cluster_depth_muted") or []),
            "cluster_depth_note": (
                "Continuum band depths from PROOF_DEPTH_DEFS fused into HDBSCAN; "
                "ice d1.43/d1.50 excluded; flat channels muted; adapt uses milder "
                "depth weight; zero-cluster / adapt-noise fallbacks drop depth only; "
                "CLUSTER_SPM/DIAG freeze unchanged"
            ),
            "prior_dir": ctx.get("prior_dir") or SHARED_PRIOR_DIR,
            "prior_encoder_path": ctx.get("prior_encoder_path"),
            "prior_loaded": bool(ctx.get("prior_loaded", False)),
            "map_proof_consistency": (
                "library_strong cluster names demoted when mineral group "
                "not PRESENT in mineral_proof"
            ),
            "map_proof_demotions": int(ctx.get("map_proof_demotions") or 0),
            "cr_sanitize": [CR_SANITIZE_LO, CR_SANITIZE_HI],
            "proof_depth_keys": SPM_PROOF_DEPTH_KEY,
            "sulfate_proof_depth_keys": list(SULFATE_PROOF_DEPTH_KEYS),
            "proof_depth_floors": {
                "olivine": PROOF_DEPTH_OLIVINE,
                "pyroxene": PROOF_DEPTH_PYROXENE,
                "bd_generic": PROOF_DEPTH_BD,
                "bd_min_clay_adaptive": PROOF_DEPTH_BD_MIN,
            },
            "sam_proof": "diagnostic_window_open_usgs",
        },
    }
    # Attach mineral proof summary if available
    if ctx.get("mineral_proof"):
        report["mineral_proof_summary"] = ctx["mineral_proof"].get("summary", [])
        present = [
            k for k, v in ctx["mineral_proof"].get("minerals", {}).items()
            if v.get("present")
        ]
        report["minerals_present"] = present
        report["external_ground_truth"] = None
        if present:
            report["disclaimer"] = (
                "Minerals listed in minerals_present are supported by Viviano-Beck "
                "SPM hotspots plus diagnostic-band CR depths and/or library SAM on the "
                "bland-ratioed denoised S4 cube (see mineral_proof.json). Abundance "
                "maps are soft mixture estimates, not ground-truth percentages. "
                "Labels remain provisional without external published geologic maps."
            )
    report_path = os.path.join(output_dir, "mineral_identification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Figures saved to {output_dir}")
    print(f"  Report: {report_path}")
    print(f"  Provisional verdict: {verdict} "
          f"(clustering={vdetail['clustering_quality']}; "
          f"mineral={vdetail['mineral_detection_confidence']}; "
          f"dbcv={dbcv if dbcv is not None else 'n/a'})")

    ctx["report"] = report
    return ctx


# -------------------------------------------------------------
# ORCHESTRATOR
# -------------------------------------------------------------
def run_mineral_identification(data_dir=None, output_dir=None,
                               mode=None, prior_dir=None):
    """
    Run Stages 0–7 of CRISM mineral identification (linear pipeline).

    Parameters
    ----------
    data_dir : str, optional
        CRISM scene directory containing ml_output/*_S4.npy
    output_dir : str, optional
        Destination for mineral_output products
    mode : 'full' | 'adapt' | None
        None → CRISM_TRAIN_MODE env (default full).
        adapt: warm-start encoder from multi-scene prior, shorter epoch caps.
        AbundanceNetwork is always trained fresh.
    prior_dir : str, optional
        Directory with encoder_B438.pth (and denoise priors for Stage4).

    Returns
    -------
    dict
        Validation / summary report, or None on failure.
    """
    global DATA_DIR, OUTPUT_DIR, TRAIN_MODE, SHARED_PRIOR_DIR
    global ENCODER_EPOCHS, UNMIX_EPOCHS_A, UNMIX_EPOCHS_B
    global ENCODER_PATIENCE, UNMIX_PATIENCE
    _set_seed(42)

    data_dir = data_dir or os.environ.get("CRISM_DATA_DIR", DATA_DIR)
    output_dir = output_dir or os.path.join(data_dir, "mineral_output")
    DATA_DIR = data_dir
    OUTPUT_DIR = output_dir

    try:
        from crism_transfer_config import (
            resolve_mode, resolve_prior_dir, epoch_budget, patience_budget,
        )
        TRAIN_MODE = resolve_mode(mode)
        SHARED_PRIOR_DIR = resolve_prior_dir(prior_dir)
        ENCODER_EPOCHS = epoch_budget(TRAIN_MODE, "encoder")
        UNMIX_EPOCHS_A = epoch_budget(TRAIN_MODE, "unmix_a")
        UNMIX_EPOCHS_B = epoch_budget(TRAIN_MODE, "unmix_b")
        ENCODER_PATIENCE = patience_budget(TRAIN_MODE, "encoder")
        UNMIX_PATIENCE = patience_budget(TRAIN_MODE, "unmix")
    except Exception as e:
        print(f"  [WARN] transfer config: {e}; using full defaults")
        TRAIN_MODE = (mode or "full").lower()
        SHARED_PRIOR_DIR = prior_dir

    print("=" * 65)
    print("  STAGE 5: CRISM MINERAL IDENTIFICATION")
    print(f"  Mode: {TRAIN_MODE} | encoder_ep≤{ENCODER_EPOCHS} | "
          f"unmix_A≤{UNMIX_EPOCHS_A}")
    if TRAIN_MODE == "adapt":
        print(f"  Prior dir: {SHARED_PRIOR_DIR}")
    print("  Linear pipeline: Setup -> Library -> SPM -> Encoder ->")
    print("                   HDBSCAN -> Unmixing -> Smoothing -> Validate")
    print("=" * 65)
    print(f"  Data dir  : {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  USGS lib  : {USGS_DIR}")
    print(f"  CUDA      : {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'})")

    t_all = time.time()
    try:
        ctx = stage0_setup(data_dir, output_dir)
        ctx["train_mode"] = TRAIN_MODE
        ctx = stage1_library(ctx)
        ctx = stage2_scene_spm(ctx)
        ctx = stage3_encoder(ctx)
        ctx = stage4_hdbscan(ctx)
        ctx = stage5_unmixing(ctx)
        ctx = stage6_smoothing(ctx)
        ctx = detect_rare_minerals(ctx)
        ctx = stage7_validation(ctx)
        ctx = write_mineral_proof(ctx)
        ctx = apply_map_proof_consistency(ctx)
        # Recompute multi-factor verdict now that mineral_proof exists; rewrite report
        if ctx.get("report") is not None and ctx.get("mineral_proof") is not None:
            report = ctx["report"]
            report["train_mode"] = TRAIN_MODE
            report["prior_dir"] = ctx.get("prior_dir") or SHARED_PRIOR_DIR
            report["prior_encoder_path"] = ctx.get("prior_encoder_path")
            report["prior_loaded"] = bool(ctx.get("prior_loaded", False))
            report["encoder_epochs_ran"] = ctx.get("encoder_epochs_ran")
            report["encoder_best_val"] = ctx.get("encoder_best_val")
            report["cluster_purity"] = ctx.get("cluster_purity")
            report["map_proof_demotions"] = int(ctx.get("map_proof_demotions") or 0)
            report["specimen_depth_demotions"] = int(
                ctx.get("specimen_depth_demotions") or 0
            )
            report["mineral_proof_summary"] = ctx["mineral_proof"].get("summary", [])
            present = [
                k for k, v in ctx["mineral_proof"].get("minerals", {}).items()
                if v.get("present")
            ]
            report["minerals_present"] = present
            report["minerals_present_evidence"] = (
                ctx.get("minerals_present_evidence")
                or ctx["mineral_proof"].get("minerals_present_evidence")
                or {}
            )
            report["clay_floor_sensitivity"] = (
                ctx.get("clay_floor_sensitivity")
                or ctx["mineral_proof"].get("clay_floor_sensitivity")
            )
            report["map_available"] = bool(
                ctx.get("map_available", report.get("n_clusters", 0) >= 1)
            )
            vdetail = compute_scene_verdict(ctx)
            report["overall_verdict"] = vdetail["overall_verdict"]
            report["clustering_quality"] = vdetail["clustering_quality"]
            report["mineral_detection_confidence"] = vdetail["mineral_detection_confidence"]
            report["verdict_detail"] = vdetail
            report["dbcv_score"] = vdetail["dbcv"]
            report["dbcv_status"] = vdetail["dbcv_status"]
            if present:
                report["disclaimer"] = (
                    "Minerals listed in minerals_present are supported by Viviano-Beck "
                    "SPM hotspots plus diagnostic-band CR depths and/or library SAM on the "
                    "bland-ratioed denoised S4 cube (see minerals_present_evidence and "
                    "mineral_proof.json). Cite n_hot/depth/hot-bland, not names alone. "
                    "overall_verdict uses mineral_proof evidence; clustering_quality "
                    "(DBCV) is reported separately and does not alone force WEAK. "
                    "Abundance maps are soft mixture estimates. Labels remain provisional "
                    "without external published geologic maps."
                    + (
                        " MAP UNAVAILABLE: 0 HDBSCAN clusters — presence is scene-level only."
                        if not report["map_available"] else ""
                    )
                )
            with open(os.path.join(ctx["output_dir"], "mineral_identification_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            ctx["report"] = report
            print(f"  Final verdict: {vdetail['overall_verdict']} "
                  f"(clustering={vdetail['clustering_quality']}, "
                  f"mineral={vdetail['mineral_detection_confidence']})")
        # Refresh offline gate report so it matches this run (not a stale pre-run file)
        try:
            from audit_scene_mineral_gates import run_offline_gates
            gate_payload = run_offline_gates(output_dir, data_dir=data_dir, quiet=False)
            if ctx.get("report") is not None:
                ctx["report"]["offline_gates"] = {
                    "all_pass": gate_payload.get("all_pass"),
                    "hard_pass": gate_payload.get("hard_pass"),
                    "n_fail": gate_payload.get("n_fail"),
                }
                with open(os.path.join(output_dir, "mineral_identification_report.json"), "w") as f:
                    json.dump(ctx["report"], f, indent=2)
        except Exception as e:
            print(f"  [WARN] offline gates refresh failed: {type(e).__name__}: {e}")
    except Exception:
        import traceback
        traceback.print_exc()
        return None
    finally:
        _clear_cuda()

    elapsed = time.time() - t_all
    report = ctx.get("report", {})
    vd = report.get("verdict_detail") or {}
    print("\n" + "=" * 65)
    print(f"  MINERAL IDENTIFICATION COMPLETE  ({elapsed / 60.0:.1f} min)")
    print(f"  Mode     : {TRAIN_MODE}")
    print(f"  Clusters : {report.get('n_clusters')}  |  clustering_quality={report.get('clustering_quality') or vd.get('clustering_quality')}"
          f"  |  map_available={report.get('map_available')}")
    print(f"  Present  : {report.get('minerals_present')}  ← proof authority (not cluster map colors)")
    ev = report.get("minerals_present_evidence") or {}
    for grp, row in ev.items():
        d = row.get("depth")
        hmb = row.get("hot_minus_bland")
        print(
            f"    evidence {grp}: n_hot={row.get('n_hot')} "
            f"{row.get('depth_key')}="
            f"{'nan' if d is None else f'{d:.4f}'} "
            f"hot-bland={'nan' if hmb is None else f'{hmb:.4f}'}"
        )
    cs = report.get("clay_floor_sensitivity") or {}
    if cs:
        print(
            f"  Clay sens (log-only): n_would_drop_at_008="
            f"{cs.get('n_would_drop')} (does not change PRESENT)"
        )
    print(f"  Verdict  : {report.get('overall_verdict')}  |  mineral_confidence={report.get('mineral_detection_confidence') or vd.get('mineral_detection_confidence')}")
    print(f"  DBCV     : {report.get('dbcv_score')} ({report.get('dbcv_status')})")
    print(f"  Encoder  : epochs_ran={report.get('encoder_epochs_ran')} best_val={report.get('encoder_best_val')}")
    print(f"  Outputs  : {output_dir}")
    print("=" * 65)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CRISM mineral identification (Stage 5)")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="CRISM scene directory (default: CRISM_DATA_DIR or Scene 12)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <data-dir>/mineral_output)",
    )
    parser.add_argument(
        "--mode", type=str, choices=("full", "adapt"), default=None,
        help="full=from-scratch; adapt=warm-start encoder from shared prior",
    )
    parser.add_argument(
        "--prior-dir", type=str, default=None,
        help="Shared prior directory (default: CRISM_SHARED_PRIOR)",
    )
    args = parser.parse_args()
    summary = run_mineral_identification(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        prior_dir=args.prior_dir,
    )
    raise SystemExit(0 if summary else 1)