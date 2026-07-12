"""
dataset.py  —  Dataset loaders for CRISM, M³, and generic HSI cubes
=====================================================================
Supports:
  • CRISMPairDataset   — uses Stage-3 (*_S3.npy) as noisy input,
                         Stage-4 (*_S4.npy) as clean target.
                         Optionally injects extra synthetic Gaussian noise.
  • CRISMInferDataset  — single-cube loader for inference (no labels).
  • GenericHSIDataset  — general-purpose loader for any (H,W,B) numpy array.
  • M3Dataset          — M³ PDS3 .img loader (same format as CRISM S-sensor).
"""

import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset

# ── local imports ─────────────────────────────────────────────────────────────
import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from crism_utils import (
    parse_pds3_label, load_crism_cube, find_label,
    get_sensor_waves, build_bad_mask,
    IF_MAX, IF_MIN,
)
from hsid_cnn.config import (
    K_ADJACENT, PATCH_H, PATCH_W, PATCH_STRIDE,
    NOISE_SIGMA_RANGE, IF_MIN, IF_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalise_cube(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-band z-score normalisation.
    Returns (normalised_cube, means[B], stds[B]).
    Inverse: cube * stds + means
    """
    B = cube.shape[2]
    means = np.zeros(B, dtype=np.float32)
    stds  = np.ones (B, dtype=np.float32)
    out   = np.zeros_like(cube, dtype=np.float32)
    for b in range(B):
        sl = cube[:, :, b]
        ok = np.isfinite(sl) & (sl > IF_MIN) & (sl < IF_MAX)
        if ok.sum() > 1:
            means[b] = float(sl[ok].mean())
            stds [b] = max(float(sl[ok].std()), 1e-6)
        v = sl.copy()
        v[~ok] = 0.0
        out[:, :, b] = (v - means[b]) / stds[b]
    out = np.clip(out, -10.0, 10.0)
    return out, means, stds


def denormalise_cube(cube_n: np.ndarray,
                     means:  np.ndarray,
                     stds:   np.ndarray) -> np.ndarray:
    """Invert per-band z-score normalisation."""
    out = np.zeros_like(cube_n)
    for b in range(cube_n.shape[2]):
        out[:, :, b] = cube_n[:, :, b] * stds[b] + means[b]
    return out


def extract_patches(cube_n: np.ndarray, patch_h: int, patch_w: int,
                    stride: int) -> list[dict]:
    """
    Extract overlapping patches from a (H, W, B) cube.
    Returns list of dicts with keys: 'patch_n' (H,W,B), 'row', 'col'.
    """
    H, W, _ = cube_n.shape
    patches  = []
    for r in range(0, max(1, H - patch_h + 1), stride):
        for c in range(0, max(1, W - patch_w + 1), stride):
            r2 = min(r + patch_h, H)
            c2 = min(c + patch_w, W)
            patches.append({'row': r, 'col': c,
                             'patch': cube_n[r:r2, c:c2, :]})
    # Always include the bottom-right corner patch
    if H > patch_h or W > patch_w:
        patches.append({'row': H - patch_h, 'col': W - patch_w,
                        'patch': cube_n[H-patch_h:, W-patch_w:, :]})
    return patches


def get_adjacent(cube: np.ndarray, band_idx: int, K: int) -> np.ndarray:
    """
    Return a (H, W, K) sub-cube of K bands centred on band_idx.
    Boundary bands are mirror-padded.
    """
    H, W, B = cube.shape
    half    = K // 2
    indices = []
    for offset in range(-half, half + (K % 2)):
        idx = band_idx + offset
        if idx < 0:
            idx = -idx                  # mirror at left
        elif idx >= B:
            idx = 2 * (B - 1) - idx    # mirror at right
        indices.append(max(0, min(idx, B - 1)))
    return cube[:, :, indices[:K]]     # (H, W, K)


import json

def load_noise_config():
    config_path = os.path.join(_PROJ, 'hsid_cnn', 'noise_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f).get('noise', {})
    return {}

def inject_synthetic_noise_patch(patch: np.ndarray, config: dict, is_3d: bool = False) -> np.ndarray:
    """
    Inject realistic CRISM noise dynamically into a raw physical patch.
    patch: (H, W) or (H, W, B). 
    """
    noisy = patch.copy()
    H, W = noisy.shape[:2]
    B = noisy.shape[2] if is_3d else 1
    
    if not is_3d:
        noisy = noisy[:, :, np.newaxis]
        
    c_gauss = config.get('gaussian', {})
    if c_gauss.get('enabled', True):
        for b in range(B):
            sig_min, sig_max = c_gauss.get('sigma_min', 0.005), c_gauss.get('sigma_max', 0.05)
            sigma = random.uniform(sig_min, sig_max)
            noisy[:, :, b] += np.random.normal(0, sigma, (H, W))
            
    c_pois = config.get('poisson', {})
    if c_pois.get('enabled', True):
        scale = c_pois.get('scale', 1000.0)
        # Shift to positive for Poisson
        shifted = np.clip(noisy * scale, 0, None)
        noisy = np.random.poisson(shifted).astype(np.float32) / scale
        
    c_stripe = config.get('striping', {})
    if c_stripe.get('enabled', True):
        prob = c_stripe.get('probability', 0.35)
        a_min, a_max = c_stripe.get('amplitude_min', 0.01), c_stripe.get('amplitude_max', 0.06)
        for b in range(B):
            if random.random() < prob:
                num_stripes = random.randint(1, max(1, W // 5))
                cols = random.sample(range(W), num_stripes)
                for c in cols:
                    noisy[:, c, b] += random.uniform(a_min, a_max) * random.choice([-1, 1])
                    
    c_off = config.get('detector_offsets', {})
    if c_off.get('enabled', True):
        prob = c_off.get('probability', 0.1)
        a_min, a_max = c_off.get('amplitude_min', 0.01), c_off.get('amplitude_max', 0.04)
        for b in range(B):
            if random.random() < prob:
                noisy[:, :, b] += random.uniform(a_min, a_max) * random.choice([-1, 1])
                
    c_dead = config.get('dead_pixels', {})
    if c_dead.get('enabled', True):
        prob = c_dead.get('probability', 0.002)
        mask = np.random.rand(H, W, B) < prob
        noisy[mask] = 0.0
        
    if not is_3d:
        return noisy[:, :, 0]
    return noisy

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC NOISE DATASET  (Clean Proxy → Synthetic Noisy)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticNoiseDataset(Dataset):
    """
    Supervised dataset for training HSID-CNN independent of the user's pipeline.
    Uses high-quality cubes (e.g., S4 output) strictly as a CLEAN PROXY target.
    Generates synthetic band-varying Gaussian noise, structured stripes, dead pixels
    and Poisson noise on-the-fly dynamically inside __getitem__.
    """

    def __init__(self,
                 ml_dir:      str,
                 file_list:   list[str] | None = None,
                 K:           int   = K_ADJACENT,
                 patch_h:     int   = PATCH_H,
                 patch_w:     int   = PATCH_W,
                 stride:      int   = PATCH_STRIDE,
                 augment:     bool  = True):

        self.K           = K
        self.patch_h     = patch_h
        self.patch_w     = patch_w
        self.augment     = augment
        self.samples     = []
        self.config      = load_noise_config()

        if file_list is not None:
            s4_files = [os.path.join(ml_dir, f) for f in file_list]
        else:
            s4_files = sorted(glob.glob(os.path.join(ml_dir, '**', '*_S4.npy'),
                                        recursive=True) +
                              glob.glob(os.path.join(ml_dir, '*_S4.npy')))

        for s4_path in s4_files:
            if not os.path.exists(s4_path):
                continue
                
            stem = os.path.basename(s4_path).replace('_S4.npy', '')
            clean_raw = np.load(s4_path).astype(np.float32)
            clean_raw = np.clip(clean_raw, IF_MIN, IF_MAX)
            clean_raw = np.nan_to_num(clean_raw, nan=0.0)

            # Generate ONE temporary noisy realization to extract stable statistics
            noisy_tmp = inject_synthetic_noise_patch(clean_raw.copy(), self.config, is_3d=True)
            _, means, stds = normalise_cube(noisy_tmp)

            H, W, B = clean_raw.shape
            ph = min(patch_h, H)
            pw = min(patch_w, W)

            # Store the raw cube in memory, extract patches lazily
            self.samples.append({
                'stem': stem,
                'clean_raw': clean_raw,
                'means': means,
                'stds': stds,
                'H': H, 'W': W, 'B': B, 'ph': ph, 'pw': pw
            })
            print(f"  [Dataset] Registered {stem}: {H}×{W}×{B} (Statistics cached)")

        # Unroll patches into an index
        self.index = []
        for s_idx, s in enumerate(self.samples):
            for band in range(s['B']):
                for r in range(0, max(1, s['H'] - s['ph'] + 1), stride):
                    for c in range(0, max(1, s['W'] - s['pw'] + 1), stride):
                        self.index.append((s_idx, band, r, c))
                # Corner
                if s['H'] > s['ph'] or s['W'] > s['pw']:
                    self.index.append((s_idx, band, s['H'] - s['ph'], s['W'] - s['pw']))

        print(f"  [Dataset] Total dynamically generated patches: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        s_idx, band, r, c = self.index[idx]
        s = self.samples[s_idx]
        
        ph, pw = s['ph'], s['pw']
        means, stds = s['means'], s['stds']
        clean_raw = s['clean_raw']
        
        # 1. Fetch RAW physical patch (spatial + spectral context)
        sp_clean = clean_raw[r:r+ph, c:c+pw, band].copy()
        
        half = self.K // 2
        indices = []
        B = s['B']
        for offset in range(-half, half + (self.K % 2)):
            i = band + offset
            if i < 0: i = -i
            elif i >= B: i = 2 * (B - 1) - i
            indices.append(max(0, min(i, B - 1)))
            
        adj_clean = clean_raw[r:r+ph, c:c+pw, indices].copy()
        
        # 2. Inject noise dynamically onto the 3D block
        adj_noisy = inject_synthetic_noise_patch(adj_clean, self.config, is_3d=True)
        
        # Extract the target band's noisy patch from the injected 3D block to guarantee consistency
        target_idx_in_adj = self.K // 2
        sp_noisy = adj_noisy[:, :, target_idx_in_adj].copy()
        
        # 3. Normalize BOTH using the predefined synthetic noisy statistics
        sp_clean_n  = (sp_clean - means[band]) / stds[band]
        sp_noisy_n  = (sp_noisy - means[band]) / stds[band]
        
        adj_noisy_n = np.zeros_like(adj_noisy)
        for i, b_idx in enumerate(indices):
            adj_noisy_n[:, :, i] = (adj_noisy[:, :, i] - means[b_idx]) / stds[b_idx]
            
        sp_clean_n = np.clip(sp_clean_n, -10.0, 10.0)
        sp_noisy_n = np.clip(sp_noisy_n, -10.0, 10.0)
        adj_noisy_n = np.clip(adj_noisy_n, -10.0, 10.0)

        if self.augment:
            if random.random() > 0.5:
                sp_noisy_n  = np.flip(sp_noisy_n,  axis=1).copy()
                sp_clean_n  = np.flip(sp_clean_n,  axis=1).copy()
                adj_noisy_n = np.flip(adj_noisy_n, axis=1).copy()
            if random.random() > 0.5:
                sp_noisy_n  = np.flip(sp_noisy_n,  axis=0).copy()
                sp_clean_n  = np.flip(sp_clean_n,  axis=0).copy()
                adj_noisy_n = np.flip(adj_noisy_n, axis=0).copy()

        spatial_noisy  = torch.tensor(sp_noisy_n,  dtype=torch.float32).unsqueeze(0)
        spatial_clean  = torch.tensor(sp_clean_n,  dtype=torch.float32).unsqueeze(0)
        spectral_noisy = torch.tensor(adj_noisy_n.transpose(2, 0, 1), dtype=torch.float32)

        return spatial_noisy, spectral_noisy, spatial_clean


# ─────────────────────────────────────────────────────────────────────────────
# CRISM INFERENCE DATASET  (single cube, no labels)
# ─────────────────────────────────────────────────────────────────────────────

class CRISMInferDataset(Dataset):
    """
    Inference-only loader.  Loads one cube and yields (band_idx, patch_info,
    spatial, spectral) tuples for band-by-band denoising.
    """

    def __init__(self, cube_path: str, K: int = K_ADJACENT,
                 patch_h: int = PATCH_H, patch_w: int = PATCH_W,
                 stride: int = PATCH_STRIDE):

        raw = np.load(cube_path).astype(np.float32)
        raw = np.clip(raw, IF_MIN, IF_MAX)
        raw = np.nan_to_num(raw, nan=0.0)

        self.cube_raw = raw
        H, W, B = raw.shape
        self.H, self.W, self.B = H, W, B
        self.K = K
        ph = min(patch_h, H)
        pw = min(patch_w, W)
        self.ph = ph
        self.pw = pw

        self.cube_n, self.means, self.stds = normalise_cube(raw)
        self.patches = extract_patches(self.cube_n, ph, pw, stride)

        self.index = []   # (band, patch_idx)
        for band in range(B):
            for pi, p in enumerate(self.patches):
                self.index.append((band, pi))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        band, pi = self.index[idx]
        p        = self.patches[pi]
        r, c     = p['row'], p['col']

        sp  = self.cube_n[r:r+self.ph, c:c+self.pw, band]          # (ph,pw)
        adj = get_adjacent(self.cube_n, band, self.K)                # (H,W,K)
        adj_patch = adj[r:r+self.ph, c:c+self.pw, :]                # (ph,pw,K)

        spatial  = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)
        spectral = torch.tensor(adj_patch.transpose(2, 0, 1), dtype=torch.float32)

        return band, r, c, spatial, spectral


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC HSI DATASET  (any H×W×B numpy array with optional clean pair)
# ─────────────────────────────────────────────────────────────────────────────

class GenericHSIDataset(Dataset):
    """
    Load an arbitrary HSI cube (H, W, B) as a numpy array.
    If clean_cube is provided, returns (noisy, spectral, clean) triplets.
    Otherwise returns (noisy, spectral) pairs for self-supervised use.
    """

    def __init__(self, noisy_cube: np.ndarray,
                 clean_cube: np.ndarray | None = None,
                 K: int = K_ADJACENT,
                 patch_h: int = 32, patch_w: int = 32,
                 stride: int = 16,
                 inject_noise: bool = False,
                 augment: bool = True):

        noisy_cube = np.clip(np.nan_to_num(
            noisy_cube.astype(np.float32), nan=0.0), IF_MIN, IF_MAX)

        self.cube_n, self.means, self.stds = normalise_cube(noisy_cube)
        H, W, B = self.cube_n.shape
        ph = min(patch_h, H)
        pw = min(patch_w, W)
        self.K = K
        self.ph, self.pw = ph, pw
        self.inject_noise = inject_noise
        self.augment = augment

        if clean_cube is not None:
            clean_cube = np.clip(np.nan_to_num(
                clean_cube.astype(np.float32), nan=0.0), IF_MIN, IF_MAX)
            self.clean_n = np.zeros_like(clean_cube, dtype=np.float32)
            for b in range(B):
                self.clean_n[:, :, b] = (
                    clean_cube[:, :, b] - self.means[b]) / self.stds[b]
        else:
            self.clean_n = None

        patches = extract_patches(self.cube_n, ph, pw, stride)
        self.index = [(band, p) for band in range(B) for p in patches]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        band, p  = self.index[idx]
        r, c     = p['row'], p['col']
        ph, pw   = self.ph, self.pw

        sp_n     = self.cube_n[r:r+ph, c:c+pw, band].copy()
        adj_n    = get_adjacent(self.cube_n, band, self.K)[r:r+ph, c:c+pw, :]

        if self.inject_noise:
            sig  = random.uniform(*NOISE_SIGMA_RANGE) / 100.0
            sp_n += np.random.normal(0, sig, sp_n.shape)
            for k in range(self.K):
                adj_n[:, :, k] += np.random.normal(0, sig, (ph, pw))

        if self.augment:
            if random.random() > 0.5:
                sp_n  = np.flip(sp_n, axis=1).copy()
                adj_n = np.flip(adj_n, axis=1).copy()
            if random.random() > 0.5:
                sp_n  = np.flip(sp_n, axis=0).copy()
                adj_n = np.flip(adj_n, axis=0).copy()

        spatial  = torch.tensor(sp_n,  dtype=torch.float32).unsqueeze(0)
        spectral = torch.tensor(adj_n.transpose(2, 0, 1), dtype=torch.float32)

        if self.clean_n is not None:
            sp_c = self.clean_n[r:r+ph, c:c+pw, band]
            if self.augment:
                pass   # same flips were applied above
            clean = torch.tensor(sp_c.copy(), dtype=torch.float32).unsqueeze(0)
            return spatial, spectral, clean

        return spatial, spectral


# ─────────────────────────────────────────────────────────────────────────────
# M³ DATASET  (Moon Mineralogy Mapper)
# ─────────────────────────────────────────────────────────────────────────────

class M3Dataset(CRISMInferDataset):
    """
    M³ PDS3 .img loader. M³ Global Mode: 85 bands; Target Mode: 260 bands.
    Uses the same PDS3 parsing infrastructure as CRISM.
    Inherits CRISMInferDataset for patch-based inference.
    """

    def __init__(self, img_path: str, K: int = K_ADJACENT,
                 patch_h: int = 32, patch_w: int = 32,
                 stride:  int = 16):

        lbl_path = find_label(img_path)
        meta     = parse_pds3_label(lbl_path) if lbl_path else {}
        cube     = load_crism_cube(img_path, meta)   # reuse CRISM loader
        cube     = cube.astype(np.float32)

        # Save as temp .npy then call super
        _tmp = img_path.replace('.img', '_m3tmp.npy')
        np.save(_tmp, cube)
        super().__init__(_tmp, K=K, patch_h=patch_h, patch_w=patch_w, stride=stride)
        os.remove(_tmp)
        self.waves = get_sensor_waves(cube.shape[2], meta.get('wavelengths'))
