"""
infer.py  —  Inference script for HSID-CNN on new HSI cubes
===========================================================
Loads a trained model and applies it to a noisy HSI cube (CRISM or M³).
Performs patch-based inference and stitches the results.
Usage:
  python -m hsid_cnn.infer --ckpt checkpoints/hsid_cnn_best.pth --input data/noisy_cube.npy --output data/clean_cube.npy
"""

import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsid_cnn.config  import K_ADJACENT, PATCH_H, PATCH_W, PATCH_STRIDE
from hsid_cnn.dataset import CRISMInferDataset, denormalise_cube
from hsid_cnn.model   import build_model
from hsid_cnn.train   import get_device
from hsid_cnn.metrics import compute_snr


@torch.no_grad()
def run_inference(ckpt_path: str, input_path: str, output_path: str, batch_size: int = 64):
    device = get_device()
    print(f"\n  [Infer] Device: {device}")
    
    # ── Load Model ────────────────────────────────────────────────────────
    print(f"  [Infer] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('config', {})
    K    = cfg.get('K', K_ADJACENT)
    
    model = build_model(K=K, n_feat=cfg.get('n_feat', 20), n_ch=cfg.get('n_ch', 64)).to(device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # ── Load Dataset ──────────────────────────────────────────────────────
    print(f"  [Infer] Loading input cube: {input_path}")
    dataset = CRISMInferDataset(input_path, K=K, patch_h=PATCH_H, patch_w=PATCH_W, stride=PATCH_STRIDE)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    H, W, B = dataset.H, dataset.W, dataset.B
    
    # To stitch patches back, we keep a sum and a count array to average overlapping regions
    out_sum   = np.zeros((H, W, B), dtype=np.float32)
    out_count = np.zeros((H, W, B), dtype=np.float32)
    
    print(f"  [Infer] Running inference on {B} bands, {len(dataset)} total patches...")
    
    import time
    import json
    
    start_time = time.time()
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    for band_idx, r, c, spatial, spectral in loader:
        spatial  = spatial.to(device)
        spectral = spectral.to(device)
        
        pred = model(spatial, spectral)  # (batch, 1, ph, pw)
        pred = pred.cpu().numpy().squeeze(1)  # (batch, ph, pw)
        
        for i in range(len(band_idx)):
            b = band_idx[i].item()
            row = r[i].item()
            col = c[i].item()
            ph = dataset.ph
            pw = dataset.pw
            
            out_sum[row:row+ph, col:col+pw, b] += pred[i]
            out_count[row:row+ph, col:col+pw, b] += 1.0

    elapsed = time.time() - start_time
    peak_mem_mb = 0.0
    if device.type == 'mps':
        peak_mem_mb = torch.mps.driver_allocated_memory() / (1024 ** 2)
        
    print(f"  [Infer] Runtime: {elapsed:.2f}s | Peak Mem: {peak_mem_mb:.2f} MB")
    
    # Average overlapping patches
    out_n = np.divide(out_sum, np.maximum(out_count, 1))
    
    # Denormalise back to original scale
    out_denorm = denormalise_cube(out_n, dataset.means, dataset.stds)
    
    # Compute basic stats
    raw_cube = dataset.cube_raw
    snr_before = compute_snr(raw_cube)
    snr_after  = compute_snr(out_denorm)
    
    print(f"  [Infer] SNR Before: {snr_before:.3f} | SNR After: {snr_after:.3f}")
    
    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, out_denorm)
    
    # Save profiling stats
    stats_path = output_path.replace('.npy', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'runtime_seconds': elapsed,
            'peak_memory_mb': peak_mem_mb,
            'snr_before': snr_before,
            'snr_after': snr_after
        }, f, indent=2)
        
    print(f"  [Infer] Saved denoised cube to: {output_path}")
    
    return out_denorm


def _parse():
    p = argparse.ArgumentParser(description='Inference for HSID-CNN')
    p.add_argument('--ckpt',   required=True, help='Path to checkpoint')
    p.add_argument('--input',  required=True, help='Path to noisy cube (.npy)')
    p.add_argument('--output', required=True, help='Path to save denoised cube (.npy)')
    p.add_argument('--bs',     type=int, default=64, help='Batch size')
    return p.parse_args()

if __name__ == '__main__':
    args = _parse()
    run_inference(args.ckpt, args.input, args.output, args.bs)
