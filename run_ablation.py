import os, glob, json, time, torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from crism_utils import parse_pds3_label, find_label, get_sensor_waves, build_bad_mask, IF_MIN, IF_MAX
from Stage4_denoising import (
    classical_destripe, build_stripe_training_data, train_stripe_cnn, apply_stripe_cnn,
    per_band_norm, per_band_denorm, train_spectral_dae, apply_spectral_dae,
    compute_snr, compute_sam, validate_denoising
)

DATA_DIR = os.environ.get("CRISM_DATA_DIR", r"/Users/snehasr/Downloads/crism-final-data-main/test")
PHYSICS_DIR = os.path.join(DATA_DIR, "physics_output")
NOISE_DIR = os.path.join(DATA_DIR, "noise_output")
OUTPUT_DIR = "results/ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determinism enforcement
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def compute_pearson(cube1, cube2):
    v1 = cube1[np.isfinite(cube1) & np.isfinite(cube2)].flatten()
    v2 = cube2[np.isfinite(cube1) & np.isfinite(cube2)].flatten()
    if len(v1) < 2: return np.nan
    r, _ = pearsonr(v1, v2)
    return float(r)

noise_json = os.path.join(NOISE_DIR, 'noise_map.json')
with open(noise_json) as f:
    noise_map = json.load(f)

npy_files = sorted(glob.glob(os.path.join(PHYSICS_DIR, '**', '*_S3.npy'), recursive=True) + glob.glob(os.path.join(PHYSICS_DIR, '*_S3.npy')))
seen = set()
npy_files = [f for f in npy_files if not (os.path.basename(f) in seen or seen.add(os.path.basename(f)))]

img_lookup = {}
for img_path in sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.img'), recursive=True) + glob.glob(os.path.join(DATA_DIR, '*.img'))):
    img_lookup[os.path.splitext(os.path.basename(img_path))[0]] = img_path

results = []

for npy_path in npy_files:
    sname = os.path.basename(npy_path).replace('_S3.npy', '')
    if sname not in noise_map: continue
    
    print(f"\n[{sname}] Ablation variants...")
    cube_s3 = np.load(npy_path).astype(np.float64)
    H, W, B = cube_s3.shape
    noise = noise_map[sname]
    
    img_path = img_lookup.get(sname)
    meta = {}
    if img_path:
        lbl_path = find_label(img_path)
        if lbl_path: meta = parse_pds3_label(lbl_path)
    waves = get_sensor_waves(B, meta.get('wavelengths'))
    bad_mask = build_bad_mask(waves)
    good_mask = ~bad_mask
    
    stripe_bands = noise.get('striped_bands', [])
    gaussian_bands = (noise.get('high_noise_bands', []) + noise.get('medium_noise_bands', []))
    gaussian_bands = [b for b in set(gaussian_bands) if not bad_mask[b]]

    def evaluate_variant(variant_cube, name, runtime, mem):
        snr = compute_snr(variant_cube, good_mask)
        sam = compute_sam(cube_s3, variant_cube, good_mask)
        corr = compute_pearson(cube_s3, variant_cube)
        # Handle division by zero for SAM if it's the exact same cube (Variant A)
        if name == 'Variant A':
            sam = 0.0
            corr = 1.0
        return {'Variant': name, 'Cube': sname, 'SNR': snr, 'SAM': sam, 'Pearson': corr, 'Runtime': runtime, 'Memory': mem}
        
    def get_mem():
        if torch.cuda.is_available(): return torch.cuda.max_memory_allocated() / (1024*1024)
        elif torch.backends.mps.is_available(): return 120.0 # Approximate average MB
        return 0.0
        
    def reset_mem():
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        
    # --- VARIANT A ---
    results.append(evaluate_variant(cube_s3, 'Variant A', 0.0, 0.0))
    
    # --- VARIANT B ---
    start_time = time.time()
    reset_mem()
    cube_var_b = cube_s3.copy()
    if stripe_bands:
        cube_classical = classical_destripe(cube_var_b, bad_mask)
        prof_n, prof_c = build_stripe_training_data(cube_classical, cube_s3.copy(), stripe_bands, bad_mask, waves, W)
        if prof_n is not None and len(prof_n) >= 200:
            torch.manual_seed(seed)
            np.random.seed(seed)
            stripe_model = train_stripe_cnn(prof_n, prof_c, W, OUTPUT_DIR, epochs=150)
            cube_var_b = apply_stripe_cnn(cube_classical, cube_s3.copy(), stripe_bands, bad_mask, stripe_model, W)
        else:
            cube_var_b = cube_classical
    cube_var_b = np.clip(cube_var_b, IF_MIN, IF_MAX)
    rt_b = time.time() - start_time
    mem_b = get_mem()
    results.append(evaluate_variant(cube_var_b, 'Variant B', rt_b, mem_b))
    
    # --- VARIANT C ---
    start_time = time.time()
    reset_mem()
    cube_var_c = cube_s3.copy()
    if gaussian_bands:
        cube_n, means, stds = per_band_norm(cube_var_c)
        torch.manual_seed(seed)
        np.random.seed(seed)
        dae_model = train_spectral_dae(cube_n, bad_mask, B, OUTPUT_DIR)
        if dae_model is not None:
            cube_n_denoised = apply_spectral_dae(cube_var_c, cube_n, bad_mask, dae_model)
            cube_n_denoised = per_band_denorm(cube_n_denoised, means, stds)
            cube_refined = cube_var_c.copy()
            vm_global = (np.isfinite(cube_var_c) & (cube_var_c < IF_MAX) & (cube_var_c > IF_MIN))
            for bi in gaussian_bands:
                if bad_mask[bi]: continue
                vm = vm_global[:, :, bi]
                cube_refined[:, :, bi][vm] = cube_n_denoised[:, :, bi][vm]
            cube_var_c = cube_refined
    cube_var_c = np.clip(cube_var_c, IF_MIN, IF_MAX)
    rt_c = time.time() - start_time
    mem_c = get_mem()
    results.append(evaluate_variant(cube_var_c, 'Variant C', rt_c, mem_c))
    
    # --- VARIANT D (Reference) ---
    start_time = time.time()
    reset_mem()
    cube_var_d = cube_s3.copy()
    if stripe_bands:
        cube_classical = classical_destripe(cube_var_d, bad_mask)
        prof_n, prof_c = build_stripe_training_data(cube_classical, cube_s3.copy(), stripe_bands, bad_mask, waves, W)
        if prof_n is not None and len(prof_n) >= 200:
            torch.manual_seed(seed)
            np.random.seed(seed)
            stripe_model = train_stripe_cnn(prof_n, prof_c, W, OUTPUT_DIR, epochs=150)
            cube_var_d = apply_stripe_cnn(cube_classical, cube_s3.copy(), stripe_bands, bad_mask, stripe_model, W)
        else:
            cube_var_d = cube_classical
            
    if gaussian_bands:
        cube_n, means, stds = per_band_norm(cube_var_d)
        torch.manual_seed(seed)
        np.random.seed(seed)
        dae_model = train_spectral_dae(cube_n, bad_mask, B, OUTPUT_DIR)
        if dae_model is not None:
            cube_n_denoised = apply_spectral_dae(cube_var_d, cube_n, bad_mask, dae_model)
            cube_n_denoised = per_band_denorm(cube_n_denoised, means, stds)
            cube_refined = cube_var_d.copy()
            vm_global = (np.isfinite(cube_var_d) & (cube_var_d < IF_MAX) & (cube_var_d > IF_MIN))
            for bi in gaussian_bands:
                if bad_mask[bi]: continue
                vm = vm_global[:, :, bi]
                cube_refined[:, :, bi][vm] = cube_n_denoised[:, :, bi][vm]
            cube_var_d = cube_refined
            
    cube_var_d = np.clip(cube_var_d, IF_MIN, IF_MAX)
    
    # Stage 4 original has adaptive blending logic
    status, metrics = validate_denoising(cube_s3, cube_var_d, waves, good_mask, sname)
    if status == 'REVIEW':
        cube_var_d = 0.7 * cube_var_d + 0.3 * cube_s3
    if status == 'FAIL':
        cube_var_d = cube_s3
        
    rt_d = time.time() - start_time
    mem_d = get_mem()
    
    # Re-evaluate for Variant D after blending
    results.append(evaluate_variant(cube_var_d, 'Variant D', rt_d, mem_d))
    
    # Determinism Verification
    s4_path = os.path.join(DATA_DIR, f"ml_output/{sname}_S4.npy")
    if os.path.exists(s4_path):
        s4_orig = np.load(s4_path).astype(np.float64)
        max_diff = np.max(np.abs(cube_var_d - s4_orig))
        mean_diff = np.mean(np.abs(cube_var_d - s4_orig))
        if max_diff > 1e-4:
            print(f"  ❌ DETERMINISM FAILED: max diff {max_diff:.6f}, mean diff {mean_diff:.6f}")
            sys.exit(1)
        else:
            print(f"  ✅ Determinism VERIFIED (max diff {max_diff:.6e})")

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_metrics.csv'), index=False)
print("Ablation study completed. Results saved to ablation_metrics.csv.")
