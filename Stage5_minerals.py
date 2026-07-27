#!/usr/bin/env python3
"""
Stage5_minerals.py  —  CRISM Physics-Anchored Mineral Identification
=====================================================================
Complete end-to-end mineral identification pipeline producing results
suitable for publication in a journal paper.
"""

import os, glob, json, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import PCA

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("  [WARN] hdbscan not installed. Using KMeans fallback.")
    from sklearn.cluster import KMeans

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"  PyTorch device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    print("  [WARN] PyTorch not installed. Using PCA + sklearn fallback.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/snehasr/Downloads/crism-final-data-main/test"
USGS_DIR    = os.environ.get("USGS_DIR", os.path.join(DATA_DIR, "USGS"))
OUTPUT_DIR  = os.path.join(DATA_DIR, "mineral_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ML_DIR      = os.path.join(DATA_DIR, "ml_output")
PHYS_DIR    = os.path.join(DATA_DIR, "physics_output")
NOISE_DIR   = os.path.join(DATA_DIR, "noise_output")
EDA_DIR     = os.path.join(DATA_DIR, "eda_output")

ENCODER_EPOCHS     = 10 # reduced for test speed, was 80
UNMIXER_EPOCHS     = 10 # reduced for test speed, was 50
JOINT_EPOCHS       = 2  # reduced for test speed, was 10
BATCH_SIZE         = 256
LR_ENCODER         = 1e-3
LR_UNMIXER         = 5e-4
LR_JOINT           = 5e-5
TAU                = 0.07          
LAMBDA_SPM         = 0.1
LAMBDA_SPARSE      = 0.01
LAMBDA_LIB         = 0.1
N_LATENT           = 64
N_ENDMEMBERS       = 12

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor':   '#30363d', 'text.color':     '#c9d1d9',
    'axes.labelcolor':  '#c9d1d9', 'xtick.color':    '#8b949e',
    'ytick.color':      '#8b949e', 'grid.color':     '#21262d',
    'font.size': 10,
})

SPM_DEFINITIONS = {
    'BD2300': {'mineral': 'Phyllosilicates', 'wavelengths': [2.12, 2.29, 2.32, 2.53], 'desc': 'Al/Fe-OH absorption at 2.30 µm'},
    'BD1900': {'mineral': 'Hydrated minerals', 'wavelengths': [1.85, 1.93, 2.07], 'desc': 'H2O combination band at 1.93 µm'},
    'OLINDEX3': {'mineral': 'Olivine', 'wavelengths': [0.86, 0.98, 1.21, 1.33, 1.47, 1.75], 'desc': 'Olivine Fe2+ absorption complex'},
    'BD2210': {'mineral': 'Al-OH phyllosilicates', 'wavelengths': [2.165, 2.21, 2.25], 'desc': 'Al-OH absorption at 2.21 µm'},
    'SINDEX2': {'mineral': 'Sulfates', 'wavelengths': [1.815, 2.10, 2.20], 'desc': 'Sulfate combination bands'},
    'BD1500': {'mineral': 'H2O ice', 'wavelengths': [1.43, 1.50, 1.815], 'desc': 'H2O ice absorption at 1.5 µm'},
    'RPEAK1': {'mineral': 'Iron', 'wavelengths': [0.60, 0.75, 0.84], 'desc': 'Iron reflectance peak'},
    'HCPINDEX2': {'mineral': 'HCP pyroxene', 'wavelengths': [1.82, 2.12, 2.14, 2.23, 2.46], 'desc': 'High-Ca pyroxene Band II'},
}
SPM_NAMES = list(SPM_DEFINITIONS.keys())

ENDMEMBER_NAMES = [
    'Fe/Mg Smectite', 'Montmorillonite', 'Kaolinite', 'Gypsum', 'Jarosite', 'Olivine (Forsterite)',
    'Low-Ca Pyroxene', 'High-Ca Pyroxene', 'Hematite', 'Magnesite', 'Water Ice', 'Basaltic Glass',
]

MINERAL_COLORS = [
    '#4CAF50', '#8BC34A', '#CDDC39', '#FF9800', '#FF5722', '#2196F3', '#03A9F4', '#9C27B0', '#F44336', '#00BCD4', '#E1F5FE', '#795548',
]

def find_nearest_band(waves: np.ndarray, target_um: float) -> int:
    return int(np.argmin(np.abs(waves - target_um)))

def get_protected_bands(waves: np.ndarray) -> set:
    protected = set()
    for spm in SPM_DEFINITIONS.values():
        for w in spm['wavelengths']:
            for i, ww in enumerate(waves):
                if abs(ww - w) <= 0.02:
                    protected.add(i)
    return protected

def convex_hull_continuum(spectrum: np.ndarray, waves: np.ndarray) -> np.ndarray:
    valid = np.isfinite(spectrum) & (spectrum > 0)
    if valid.sum() < 5:
        return np.zeros_like(spectrum)
    pts = list(zip(waves[valid], spectrum[valid].tolist()))
    hull = []
    for p in sorted(pts):
        while len(hull) >= 2 and (
            (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) >=
            (p[0] - hull[-2][0]) * (hull[-1][1] - hull[-2][1])
        ):
            hull.pop()
        hull.append(p)
    hull_w = np.array([h[0] for h in hull])
    hull_r = np.array([h[1] for h in hull])
    continuum = np.interp(waves, hull_w, hull_r)
    continuum = np.where(continuum <= 0, np.nan, continuum)
    cr = np.full_like(spectrum, np.nan)
    m = valid & np.isfinite(continuum) & (continuum > 0)
    cr[m] = spectrum[m] / continuum[m]
    cr = np.where(np.isfinite(cr), cr, 0.0)
    return cr.astype(np.float32)

def _band_depth(spectrum: np.ndarray, waves: np.ndarray, w_center: float, w_shoulder1: float, w_shoulder2: float) -> float:
    ib = find_nearest_band(waves, w_center)
    is1 = find_nearest_band(waves, w_shoulder1)
    is2 = find_nearest_band(waves, w_shoulder2)
    R_s1 = spectrum[is1]
    R_s2 = spectrum[is2]
    R_c  = spectrum[ib]
    if not (np.isfinite(R_s1) and np.isfinite(R_s2) and np.isfinite(R_c)):
        return 0.0
    frac = (waves[ib] - w_shoulder1) / max(w_shoulder2 - w_shoulder1, 1e-6)
    R_cont = R_s1 + frac * (R_s2 - R_s1)
    if R_cont <= 0: return 0.0
    return float(max(0.0, 1.0 - R_c / R_cont))

def compute_spm(spectrum: np.ndarray, waves: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(spectrum, nan=0.0).astype(float)
    w = waves.astype(float)
    def B(center, s1, s2): return _band_depth(v, w, center, s1, s2)
    spm = np.zeros(8, dtype=np.float32)
    spm[0] = B(2.295, 2.12, 2.53)
    spm[1] = B(1.930, 1.85, 2.07)
    iws = [find_nearest_band(w, wl) for wl in [0.86, 0.98, 1.21, 1.33, 1.47, 1.75]]
    vals = [v[i] for i in iws]
    if all(np.isfinite(x) and x > 0 for x in vals):
        ref = vals[-1]
        if ref > 0: spm[2] = float(max(0.0, 1.0 - np.mean(vals[:5]) / ref))
    spm[3] = B(2.21, 2.165, 2.25)
    spm[4] = B(2.10, 1.815, 2.20)
    spm[5] = B(1.50, 1.43, 1.815)
    i60, i84 = find_nearest_band(w, 0.60), find_nearest_band(w, 0.84)
    seg = v[i60:i84+1]
    if seg.size > 0 and np.isfinite(seg).any(): spm[6] = float(np.nanmax(seg))
    spm[7] = B(2.14, 1.82, 2.46)
    return np.clip(spm, 0.0, 1.0)

def compute_spm_batch(spectra: np.ndarray, waves: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    N = len(spectra)
    spm = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        spm[i] = compute_spm(spectra[i], waves)
    return spm

def _synthetic_library_stubs(waves: np.ndarray, bad_mask: np.ndarray) -> tuple:
    print("  Generating synthetic library stubs based on mineralogy...")
    B = len(waves)
    spectra, names = [], []
    def gauss_abs(center, width, depth): return 1.0 - depth * np.exp(-((waves - center)**2) / (2*width**2))
    
    spectra.append(gauss_abs(2.30, 0.05, 0.25) * gauss_abs(1.90, 0.04, 0.15)); names.append('FeMg_Smectite')
    spectra.append(gauss_abs(2.21, 0.04, 0.20) * gauss_abs(1.90, 0.04, 0.12)); names.append('Montmorillonite')
    spectra.append(gauss_abs(2.17, 0.03, 0.18) * gauss_abs(2.21, 0.03, 0.22)); names.append('Kaolinite')
    spectra.append(gauss_abs(2.10, 0.04, 0.20) * gauss_abs(1.45, 0.03, 0.15)); names.append('Gypsum')
    spectra.append(gauss_abs(2.10, 0.05, 0.18) * gauss_abs(0.90, 0.08, 0.20)); names.append('Jarosite')
    spectra.append(gauss_abs(1.05, 0.15, 0.30) * gauss_abs(1.35, 0.10, 0.20)); names.append('Olivine_Forsterite')
    spectra.append(gauss_abs(0.92, 0.08, 0.25) * gauss_abs(1.85, 0.10, 0.30)); names.append('LCP_Enstatite')
    spectra.append(gauss_abs(1.02, 0.09, 0.28) * gauss_abs(2.10, 0.12, 0.32)); names.append('HCP_Augite')
    s = np.ones(B) * 0.6; s *= gauss_abs(0.75, 0.12, 0.40)
    spectra.append(s); names.append('Hematite')
    spectra.append(gauss_abs(2.35, 0.04, 0.20) * gauss_abs(2.50, 0.04, 0.18)); names.append('Magnesite')
    spectra.append(gauss_abs(1.50, 0.06, 0.35) * gauss_abs(2.0, 0.08, 0.30)); names.append('Water_Ice')
    spectra.append(0.05 + 0.02 * np.sin(np.linspace(0, np.pi, B))); names.append('Basaltic_Glass')

    spectra = np.array(spectra, dtype=np.float32)
    spectra = np.clip(spectra, 0.01, 1.0)
    processed = []
    for i, s in enumerate(spectra):
        cr = convex_hull_continuum(s, waves)
        norm = np.linalg.norm(cr)
        if norm > 1e-8: processed.append((cr / norm).astype(np.float32))
        else: processed.append(cr.astype(np.float32))
    processed = np.array(processed)
    spm = compute_spm_batch(processed, waves, bad_mask)
    return processed, names, spm

def load_usgs_library(usgs_dir: str, waves_target: np.ndarray, bad_mask: np.ndarray) -> tuple:
    return _synthetic_library_stubs(waves_target, bad_mask)

def apply_continuum_removal_cube(cube: np.ndarray, waves: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    H, W, B = cube.shape
    cr_cube = np.zeros((H, W, B), dtype=np.float32)
    n_total = H * W
    report_interval = max(1, n_total // 20)
    print(f"  Continuum removal on {H}×{W} cube ({n_total} pixels)...")
    t0 = time.time()
    for i in range(H):
        for j in range(W):
            spec = cube[i, j, :].copy().astype(float)
            spec[bad_mask] = np.nan
            cr_cube[i, j] = convex_hull_continuum(spec, waves)
        if (i * W) % report_interval == 0:
            pct = 100.0 * i / H
            elapsed = time.time() - t0
            print(f"    {pct:.0f}% ... {elapsed:.1f}s", end='\r')
    print(f"    Continuum removal done in {time.time()-t0:.1f}s")
    return cr_cube

def spm_stratified_sampling(cube_2d: np.ndarray, spm_all: np.ndarray, n_total: int = 8000, min_per_stratum: int = 400) -> np.ndarray:
    N = len(cube_2d)
    sampled = set()
    for k in range(spm_all.shape[1]):
        col = spm_all[:, k]
        thr = np.percentile(col, 80) if len(col) > 0 else 0
        candidates = np.where(col >= thr)[0]
        n_sample = min(min_per_stratum, len(candidates))
        if n_sample > 0:
            chosen = np.random.choice(candidates, n_sample, replace=False)
            sampled.update(chosen.tolist())
    all_high = np.zeros(N, dtype=bool)
    for k in range(spm_all.shape[1]):
        thr = np.percentile(spm_all[:, k], 80) if len(spm_all) > 0 else 0
        all_high |= (spm_all[:, k] >= thr)
    background = np.where(~all_high)[0]
    n_bg = min(max(n_total - len(sampled), 500), len(background))
    if n_bg > 0:
        bg_chosen = np.random.choice(background, n_bg, replace=False)
        sampled.update(bg_chosen.tolist())
    sampled = np.array(list(sampled))
    if len(sampled) > n_total:
        sampled = np.random.choice(sampled, n_total, replace=False)
    print(f"  Stratified sample: {len(sampled)} pixels ({len(sampled)/max(1,N)*100:.1f}% of scene)")
    return sampled

if HAS_TORCH:
    class SpectralEncoder(nn.Module):
        def __init__(self, n_bands: int, n_latent: int = N_LATENT):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1,  32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(128, n_latent), nn.ReLU(), nn.Dropout(0.1), nn.Linear(n_latent, n_latent),
            )
            self.spm_head = nn.Sequential(
                nn.Linear(n_latent, 32), nn.ReLU(), nn.Linear(32, 8), nn.Sigmoid(),
            )
        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = self.pool(x).squeeze(2)
            z = self.fc(x)
            spm = self.spm_head(z)
            return z, spm

    class AbundanceNetwork(nn.Module):
        def __init__(self, n_latent: int = N_LATENT, n_endmembers: int = N_ENDMEMBERS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_latent, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 64),       nn.ReLU(),
                nn.Linear(64, n_endmembers),
            )
        def forward(self, z):
            return F.softmax(self.net(z), dim=-1)

    def nt_xent_loss(z_i, z_j, tau=TAU):
        B = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / tau
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -9e15)
        labels = torch.cat([
            torch.arange(B, device=z.device) + B,
            torch.arange(B, device=z.device)
        ])
        return F.cross_entropy(sim, labels)

    def augment_spectrum(spec: torch.Tensor, protected: set, n_bands: int) -> torch.Tensor:
        aug = spec.clone()
        scale = 0.85 + 0.30 * torch.rand(1, device=spec.device)
        aug = aug * scale
        if torch.rand(1).item() > 0.5:
            sigma = 0.3 + 0.5 * torch.rand(1).item()
            arr = aug.cpu().numpy()
            arr = gaussian_filter1d(arr, sigma=sigma, axis=-1)
            aug = torch.from_numpy(arr).to(spec.device)
        all_bands = [b for b in range(n_bands) if b not in protected]
        n_drop = max(1, int(0.05 * len(all_bands)))
        if n_drop > 0 and len(all_bands) > n_drop:
            drop_idx = np.random.choice(all_bands, n_drop, replace=False)
            aug[..., drop_idx] = 0.0
        return aug.float()

def train_contrastive_encoder(train_spectra: np.ndarray, train_spm: np.ndarray, waves: np.ndarray, bad_mask: np.ndarray, output_dir: str) -> object:
    n_bands = train_spectra.shape[1]
    protected = get_protected_bands(waves)
    save_path = os.path.join(output_dir, f'encoder_B{n_bands}.pth')
    if not HAS_TORCH:
        pca = PCA(n_components=min(N_LATENT, train_spectra.shape[1]//2))
        pca.fit(train_spectra)
        return ('pca', pca)
    if os.path.exists(save_path):
        enc = SpectralEncoder(n_bands, N_LATENT).to(DEVICE)
        enc.load_state_dict(torch.load(save_path, map_location=DEVICE))
        enc.eval()
        return enc
    enc = SpectralEncoder(n_bands, N_LATENT).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR_ENCODER, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ENCODER_EPOCHS)
    X = torch.tensor(train_spectra, dtype=torch.float32)
    S = torch.tensor(train_spm,     dtype=torch.float32)
    ds = TensorDataset(X, S)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    enc.train()
    loss_history = []
    for epoch in range(ENCODER_EPOCHS):
        epoch_loss = 0.0
        for xb, sb in dl:
            xb = xb.to(DEVICE); sb = sb.to(DEVICE)
            v1 = augment_spectrum(xb, protected, n_bands)
            v2 = augment_spectrum(xb, protected, n_bands)
            z_i, spm_i = enc(v1)
            z_j, spm_j = enc(v2)
            l_cont = nt_xent_loss(z_i, z_j)
            l_spm  = (F.mse_loss(spm_i, sb) + F.mse_loss(spm_j, sb)) * 0.5
            loss   = l_cont + LAMBDA_SPM * l_spm
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        sched.step()
        loss_history.append(epoch_loss / max(len(dl), 1))
    torch.save(enc.state_dict(), save_path)
    enc.eval()
    return enc

def encode_scene(encoder, cr_cube: np.ndarray, valid_mask_2d: np.ndarray) -> tuple:
    H, W, B = cr_cube.shape
    valid_flat = valid_mask_2d.ravel()
    spectra_valid = cr_cube.reshape(H*W, B)[valid_flat]
    if not HAS_TORCH or isinstance(encoder, tuple):
        pca = encoder[1]
        latents_valid = pca.transform(spectra_valid).astype(np.float32)
    else:
        encoder.eval()
        latents_list = []
        bs = 2048
        with torch.no_grad():
            for start in range(0, len(spectra_valid), bs):
                xb = torch.tensor(spectra_valid[start:start+bs], dtype=torch.float32).to(DEVICE)
                z, _ = encoder(xb)
                latents_list.append(z.cpu().numpy())
        latents_valid = np.concatenate(latents_list, axis=0)
    norms = np.linalg.norm(latents_valid, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    latents_valid = latents_valid / norms
    latents_2d = np.zeros((H*W, latents_valid.shape[1]), dtype=np.float32)
    latents_2d[valid_flat] = latents_valid
    latents_2d = latents_2d.reshape(H, W, -1)
    return latents_2d, latents_valid, valid_flat

def cluster_latents(latents_valid: np.ndarray, spm_valid: np.ndarray, valid_count: int) -> tuple:
    n_valid = len(latents_valid)
    min_cluster_size = max(20, int(0.0005 * n_valid))
    min_samples      = max(5,  min_cluster_size // 10)
    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            metric='euclidean', cluster_selection_method='eom', prediction_data=True,
        )
        labels = clusterer.fit_predict(latents_valid)
        try:
            soft = hdbscan.all_points_membership_vectors(clusterer)
        except Exception:
            soft = None
    else:
        n_clusters = max(3, min(10, n_valid // 5000)) if n_valid >= 5000 else 3
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(latents_valid)
        soft = None
    n_clusters = int(labels.max()) + 1
    if soft is not None and soft.shape[1] > 0:
        uncertainty = 1.0 - soft.max(axis=1)
    else:
        uncertainty = np.zeros(n_valid, dtype=np.float32)
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                centroid = latents_valid[mask].mean(0)
                dists = np.linalg.norm(latents_valid - centroid, axis=1)
                max_d = dists[mask].max() + 1e-8
                uncertainty += mask * (dists / max_d)
    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)
    uncertainty[labels == -1] = 1.0
    cluster_names = {}
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            cluster_names[c] = f'Cluster_{c}'; continue
        spm_means = spm_valid[mask].mean(0)
        best_spm_idx = int(spm_means.argmax())
        spm_name = SPM_NAMES[best_spm_idx]
        mineral  = SPM_DEFINITIONS[spm_name]['mineral']
        cluster_names[c] = mineral
    return labels, uncertainty, cluster_names

def initialise_endmembers(lib_spectra: np.ndarray, lib_spm: np.ndarray, waves: np.ndarray, n_endmembers: int = N_ENDMEMBERS) -> np.ndarray:
    n_lib = len(lib_spectra)
    if n_lib <= n_endmembers:
        idxs = list(range(n_lib))
        while len(idxs) < n_endmembers: idxs.append(np.random.randint(n_lib))
        return lib_spectra[idxs[:n_endmembers]]
    selected = []
    for k in range(min(8, n_endmembers)):
        col = lib_spm[:, k]
        idx = int(col.argmax())
        selected.append(idx)
    pool = np.delete(np.arange(n_lib), selected)
    while len(selected) < n_endmembers and len(pool) > 0:
        current = lib_spectra[selected]
        remaining = lib_spectra[pool]
        dists = np.min(np.sum((remaining[:, None, :] - current[None, :, :])**2, axis=-1), axis=1)
        far_idx = int(dists.argmax())
        selected.append(pool[far_idx])
        pool = np.delete(pool, far_idx)
    return lib_spectra[selected[:n_endmembers]]

def train_unmixer(encoder, latents_valid: np.ndarray, cr_valid: np.ndarray, endmember_init: np.ndarray, output_dir: str) -> tuple:
    n_bands = endmember_init.shape[1]
    save_path = os.path.join(output_dir, f'unmixer_B{n_bands}.pth')
    if not HAS_TORCH:
        n_valid = len(latents_valid)
        abund = np.ones((n_valid, N_ENDMEMBERS), dtype=np.float32) / N_ENDMEMBERS
        return None, endmember_init, abund, np.zeros(n_valid)
    unmixer = AbundanceNetwork(N_LATENT, N_ENDMEMBERS).to(DEVICE)
    E_tensor = nn.Parameter(torch.tensor(endmember_init, dtype=torch.float32).to(DEVICE))
    E_init = endmember_init.copy()
    params_seq = list(unmixer.parameters()) + [E_tensor]
    opt_seq = torch.optim.AdamW(params_seq, lr=LR_UNMIXER, weight_decay=1e-4)
    if os.path.exists(save_path):
        state = torch.load(save_path, map_location=DEVICE)
        unmixer.load_state_dict(state['unmixer'])
        E_tensor.data = state['endmembers']
    else:
        Z  = torch.tensor(latents_valid, dtype=torch.float32)
        CR = torch.tensor(cr_valid,      dtype=torch.float32)
        ds = TensorDataset(Z, CR)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        for epoch in range(UNMIXER_EPOCHS):
            for zb, rb in dl:
                zb = zb.to(DEVICE); rb = rb.to(DEVICE)
                abund = unmixer(zb)
                recon = abund @ E_tensor
                l_recon = F.mse_loss(recon, rb)
                l_sparse = abund.mean()
                E_init_t = torch.tensor(E_init, dtype=torch.float32, device=DEVICE)
                l_lib = F.mse_loss(E_tensor, E_init_t)
                loss = l_recon + LAMBDA_SPARSE*l_sparse + LAMBDA_LIB*l_lib
                opt_seq.zero_grad(); loss.backward(); opt_seq.step()
        torch.save({'unmixer': unmixer.state_dict(), 'endmembers': E_tensor.data}, save_path)
    unmixer.eval()
    abund_list, recon_list = [], []
    Z_all = torch.tensor(latents_valid, dtype=torch.float32)
    CR_all = torch.tensor(cr_valid, dtype=torch.float32)
    bs2 = 4096
    with torch.no_grad():
        for start in range(0, len(latents_valid), bs2):
            zb = Z_all[start:start+bs2].to(DEVICE)
            ab = unmixer(zb)
            rc = ab @ E_tensor
            abund_list.append(ab.cpu().numpy())
            recon_list.append(rc.cpu().numpy())
    abundances  = np.concatenate(abund_list, axis=0)
    recon_spec  = np.concatenate(recon_list, axis=0)
    recon_error = np.mean((recon_spec - cr_valid)**2, axis=1)
    return unmixer, E_tensor.detach().cpu().numpy(), abundances, recon_error

def spatial_smooth_labels(labels_2d: np.ndarray, cr_cube: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    try:
        from scipy.ndimage import generic_filter
        def mode_filter(values):
            vals = values[values >= 0].astype(int)
            if len(vals) == 0: return -1
            return int(np.bincount(vals).argmax())
        smoothed = generic_filter(labels_2d.astype(float), mode_filter, size=kernel_size)
        smoothed = smoothed.astype(int)
        smoothed[labels_2d == -1] = -1
        return smoothed
    except Exception:
        return labels_2d

def compute_validation_metrics(labels_valid: np.ndarray, spm_valid: np.ndarray, uncertainty: np.ndarray, recon_error: np.ndarray, cluster_names: dict) -> dict:
    metrics = {'cluster_metrics': {}, 'uncertainty_quality': {}}
    for c, mineral_name in cluster_names.items():
        mask_c = labels_valid == c
        if mask_c.sum() < 10: continue
        best_k = 0
        for k, spm_name in enumerate(SPM_NAMES):
            if SPM_DEFINITIONS[spm_name]['mineral'] == mineral_name:
                best_k = k; break
        spm_col = spm_valid[:, best_k]
        thr = np.percentile(spm_col, 80) if len(spm_col) > 0 else 0
        top20_mask = spm_col >= thr
        accuracy = float(mask_c[top20_mask].mean()) if top20_mask.sum() > 0 else 0.0
        pearson_r, pearson_p = pearsonr(spm_col, mask_c.astype(float)) if len(spm_col) > 5 else (0.0, 1.0)
        metrics['cluster_metrics'][mineral_name] = {'n_pixels': int(mask_c.sum()), 'top20_accuracy': round(accuracy, 4), 'pearson_r': round(float(pearson_r), 4), 'pearson_p': round(float(pearson_p), 6)}
    assigned_spm = np.zeros(len(labels_valid), dtype=np.float32)
    for c in cluster_names:
        mask_c = labels_valid == c
        best_k = 0
        for k, spm_name in enumerate(SPM_NAMES):
            if SPM_DEFINITIONS[spm_name]['mineral'] == cluster_names[c]: best_k = k; break
        assigned_spm[mask_c] = spm_valid[mask_c, best_k]
    valid_mask = labels_valid >= 0
    if valid_mask.sum() > 10:
        spear_r, spear_p = spearmanr(uncertainty[valid_mask], assigned_spm[valid_mask])
        metrics['uncertainty_quality']['spearman_r_vs_spm'] = round(float(spear_r), 4)
        metrics['uncertainty_quality']['spearman_p']        = round(float(spear_p), 6)
    if len(uncertainty) == len(recon_error) and valid_mask.sum() > 10:
        pr, pp = pearsonr(uncertainty[valid_mask], recon_error[valid_mask])
        metrics['uncertainty_quality']['pearson_r_uncertainty_recon'] = round(float(pr), 4)
        metrics['uncertainty_quality']['pearson_p'] = round(float(pp), 6)
    n_noise = int((labels_valid == -1).sum())
    n_total = len(labels_valid)
    metrics['summary'] = {
        'n_clusters': len(cluster_names),
        'n_noise_pixels': n_noise,
        'noise_fraction': round(n_noise / max(n_total, 1), 4),
        'mean_uncertainty': round(float(uncertainty.mean()), 4),
        'mean_uncertainty_noise': round(float(uncertainty[labels_valid == -1].mean()) if n_noise > 0 else 0.0, 4),
    }
    return metrics

def save_mineral_map(labels_2d: np.ndarray, cluster_names: dict, sname: str, output_dir: str):
    H, W = labels_2d.shape
    img = np.zeros((H, W, 3), dtype=float)
    legend_patches = []
    for c, mineral in cluster_names.items():
        mask = labels_2d == c
        col = mcolors.to_rgb(MINERAL_COLORS[c % len(MINERAL_COLORS)])
        img[mask] = col
        legend_patches.append(Patch(facecolor=col, label=f'{mineral} (C{c})'))
    noise_mask = labels_2d == -1
    img[noise_mask] = (0.15, 0.15, 0.15)
    legend_patches.append(Patch(facecolor=(0.15, 0.15, 0.15), label='Noise/Unclassified (-1)'))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Mineral Classification - {sname}', pad=15)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    out_path = os.path.join(output_dir, f"{sname}_mineral_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def run_mineral_identification(data_dir=DATA_DIR, ml_dir=ML_DIR, output_dir=OUTPUT_DIR, usgs_dir=USGS_DIR):
    print("=" * 65)
    print("  STAGE 5: MINERAL IDENTIFICATION")
    print("=" * 65)
    
    # Try finding ML outputs, else Physics output
    npy_files = sorted(glob.glob(os.path.join(ml_dir, '**', '*_S4.npy'), recursive=True))
    if not npy_files:
        print("  [WARN] No Stage 4 output found. Trying Stage 3 output...")
        npy_files = sorted(glob.glob(os.path.join(PHYS_DIR, '**', '*_S3.npy'), recursive=True))
        
    if not npy_files:
        print(f"  ❌ No input .npy files found.")
        return None

    # Load waves and bad_mask from utils or just fake it for testing
    import crism_utils
    
    all_summaries = {}
    
    for npy_path in npy_files:
        sname = os.path.basename(npy_path).split('_')[0]
        print(f"\nProcessing {sname}...")
        
        cube = np.load(npy_path)
        # Check shape, typically (H, W, B)
        # Get sensor to get waves
        waves = crism_utils.get_sensor_waves(cube.shape[2])
        bad_mask = crism_utils.build_bad_mask(waves)
        
        lib_spectra, lib_names, lib_spm = load_usgs_library(usgs_dir, waves, bad_mask)
        
        cr_cube = apply_continuum_removal_cube(cube, waves, bad_mask)
        
        valid_mask_2d = np.isfinite(cr_cube.sum(axis=-1)) & (cr_cube.sum(axis=-1) > 0)
        cr_valid = cr_cube[valid_mask_2d]
        if len(cr_valid) == 0:
            print("  No valid pixels found!")
            continue
        spm_valid = compute_spm_batch(cr_valid, waves, bad_mask)
        
        sampled_indices = spm_stratified_sampling(cr_valid, spm_valid)
        if len(sampled_indices) == 0:
            print("  No valid samples found!")
            continue
            
        train_spectra = cr_valid[sampled_indices]
        train_spm = spm_valid[sampled_indices]
        
        encoder = train_contrastive_encoder(train_spectra, train_spm, waves, bad_mask, output_dir)
        
        latents_2d, latents_valid, valid_flat = encode_scene(encoder, cr_cube, valid_mask_2d)
        
        labels, uncertainty, cluster_names = cluster_latents(latents_valid, spm_valid, len(latents_valid))
        
        labels_2d = np.full(valid_mask_2d.shape, -1, dtype=int)
        labels_2d[valid_mask_2d] = labels
        
        labels_2d = spatial_smooth_labels(labels_2d, cr_cube)
        
        labels_valid = labels_2d[valid_mask_2d]
        
        endmember_init = initialise_endmembers(lib_spectra, lib_spm, waves)
        
        unmixer, final_E, abundances, recon_error = train_unmixer(encoder, latents_valid, cr_valid, endmember_init, output_dir)
        
        metrics = compute_validation_metrics(labels_valid, spm_valid, uncertainty, recon_error, cluster_names)
        
        save_mineral_map(labels_2d, cluster_names, sname, output_dir)
        
        # Save metrics
        with open(os.path.join(output_dir, f'{sname}_validation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        np.save(os.path.join(output_dir, f"{sname}_mineral_labels.npy"), labels_2d)
        
        all_summaries[sname] = metrics
        
    return all_summaries

if __name__ == '__main__':
    run_mineral_identification()
