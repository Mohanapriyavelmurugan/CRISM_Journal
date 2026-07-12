"""
evaluate.py  —  Evaluation script for trained HSID-CNN models
===============================================================
Computes full metrics on a test set (S3 vs S4) or between two given directories.
Usage:
  python -m hsid_cnn.evaluate --ckpt checkpoints/hsid_cnn_best.pth
"""

import os, sys, glob, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsid_cnn.config import (
    PHYSICS_DIR, ML_DIR, CKPT_DIR,
    K_ADJACENT, PATCH_H, PATCH_W, PATCH_STRIDE,
)
from hsid_cnn.dataset import CRISMPairDataset
from hsid_cnn.model   import build_model
from hsid_cnn.metrics import compute_all_metrics
from hsid_cnn.train   import get_device

def evaluate_model(ckpt_path: str,
                   physics_dir: str = PHYSICS_DIR,
                   ml_dir:      str = ML_DIR,
                   batch_size:  int = 32):
    
    device = get_device()
    print(f"\n  [Evaluate] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Extract config if present, else use defaults
    cfg = ckpt.get('config', {})
    K      = cfg.get('K', K_ADJACENT)
    n_feat = cfg.get('n_feat', 20)
    n_ch   = cfg.get('n_ch', 64)
    
    model = build_model(K=K, n_feat=n_feat, n_ch=n_ch).to(device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"  [Evaluate] Building test dataset (no augment/inject)...")
    test_ds = CRISMPairDataset(
        physics_dir=physics_dir, ml_dir=ml_dir,
        K=K, patch_h=PATCH_H, patch_w=PATCH_W, stride=PATCH_STRIDE,
        inject_noise=False, augment=False
    )
    
    if len(test_ds) == 0:
        print("  [Evaluate] Dataset is empty.")
        return

    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_pred, all_tgt = [], []
    print("  [Evaluate] Running inference...")
    
    with torch.no_grad():
        for spatial_n, spectral_n, spatial_c in test_ld:
            spatial_n  = spatial_n .to(device)
            spectral_n = spectral_n.to(device)
            
            pred = model(spatial_n, spectral_n)
            
            all_pred.append(pred.cpu().numpy())
            all_tgt .append(spatial_c.numpy())

    pred_arr = np.concatenate(all_pred, axis=0).transpose(0, 2, 3, 1)  # (N,H,W,1)
    tgt_arr  = np.concatenate(all_tgt,  axis=0).transpose(0, 2, 3, 1)

    print("\n  [Evaluate] Computing Metrics across full test set...")
    metrics = compute_all_metrics(clean=tgt_arr, restored=pred_arr, label="Test Set Evaluation")
    
    return metrics


def _parse():
    p = argparse.ArgumentParser(description='Evaluate trained HSID-CNN')
    p.add_argument('--ckpt',        required=True, help='Path to model checkpoint')
    p.add_argument('--physics_dir', default=PHYSICS_DIR)
    p.add_argument('--ml_dir',      default=ML_DIR)
    p.add_argument('--batch_size',  type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    evaluate_model(args.ckpt, args.physics_dir, args.ml_dir, args.batch_size)
