"""
train.py  —  Training script for HSID-CNN on CRISM / generic HSI data
=======================================================================
Usage:
  python -m hsid_cnn.train                        # default settings
  python -m hsid_cnn.train --epochs 50 --lr 5e-4
  python -m hsid_cnn.train --physics_dir /path/s3 --ml_dir /path/s4
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsid_cnn.config import (
    PHYSICS_DIR, ML_DIR, CKPT_DIR, RESULTS_DIR,
    K_ADJACENT, PATCH_H, PATCH_W, PATCH_STRIDE,
    EPOCHS, BATCH_SIZE, LR, ADAM_BETAS, ADAM_EPS, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_FACTOR, EARLY_STOP_PATIENCE,
    TRAIN_VAL_SPLIT, N_FEAT_SCALE, N_BACKBONE_CH,
)
from hsid_cnn.dataset import SyntheticNoiseDataset
from hsid_cnn.model   import build_model
from hsid_cnn.metrics import compute_mpsnr, compute_mssim


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class HSIDLoss(nn.Module):
    """
    Combined loss: MSE + λ_l1 × L1 + λ_mean × mean-preservation.

    MSE  : primary reconstruction fidelity.
    L1   : robustness to outlier patches (heavy-tailed distributions).
    Mean : prevents overall brightness shift (DC offset artefact).
    """

    def __init__(self, lambda_l1: float = 0.1, lambda_mean: float = 0.05):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.l1          = nn.L1Loss()
        self.lambda_l1   = lambda_l1
        self.lambda_mean = lambda_mean

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss  = self.mse(pred, target)
        l1_loss   = self.l1 (pred, target)
        mean_loss = torch.abs(pred.mean() - target.mean())
        return mse_loss + self.lambda_l1 * l1_loss + self.lambda_mean * mean_loss


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device,
                scaler: torch.cuda.amp.GradScaler | None = None) -> float:
    model.train()
    total_loss = 0.0
    for spatial_n, spectral_n, spatial_c in loader:
        spatial_n  = spatial_n .to(device, non_blocking=True)
        spectral_n = spectral_n.to(device, non_blocking=True)
        spatial_c  = spatial_c .to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(spatial_n, spectral_n)
                loss = criterion(pred, spatial_c)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(spatial_n, spectral_n)
            loss = criterion(pred, spatial_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        total_loss += loss.item() * spatial_n.size(0)

    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_pred, all_tgt = [], []

    for spatial_n, spectral_n, spatial_c in loader:
        spatial_n  = spatial_n .to(device)
        spectral_n = spectral_n.to(device)
        spatial_c  = spatial_c .to(device)

        pred = model(spatial_n, spectral_n)
        loss = criterion(pred, spatial_c)
        total_loss += loss.item() * spatial_n.size(0)

        all_pred.append(pred.cpu().numpy())
        all_tgt .append(spatial_c.cpu().numpy())

    # Aggregate into arrays for MPSNR/MSSIM
    # Reshape to (N*H, W, 1) to match (H, W, B) expected by metrics
    pred_arr = np.concatenate(all_pred, axis=0).transpose(0, 2, 3, 1)
    tgt_arr  = np.concatenate(all_tgt , axis=0).transpose(0, 2, 3, 1)
    N, H, W, _ = pred_arr.shape
    pred_arr = pred_arr.reshape(N * H, W, 1)
    tgt_arr  = tgt_arr.reshape(N * H, W, 1)

    mpsnr = compute_mpsnr(tgt_arr, pred_arr)
    mssim = compute_mssim(tgt_arr, pred_arr)

    return {
        'loss':  total_loss / max(len(loader.dataset), 1),
        'mpsnr': mpsnr,
        'mssim': mssim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_training(ml_dir:      str = ML_DIR,
                 ckpt_dir:    str = CKPT_DIR,
                 epochs:      int = EPOCHS,
                 batch_size:  int = BATCH_SIZE,
                 lr:         float = LR,
                 K:           int = K_ADJACENT,
                 n_feat:      int = N_FEAT_SCALE,
                 n_ch:        int = N_BACKBONE_CH,
                 inject_noise: bool = True,
                 resume:      str | None = None) -> dict:

    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = get_device()
    print(f"\n  [Train] Device: {device}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\n  [Train] Building synthetic dataset from clean proxy: {ml_dir}")

    import glob
    all_s4 = sorted(glob.glob(os.path.join(ml_dir, '**', '*_S4.npy'), recursive=True) +
                    glob.glob(os.path.join(ml_dir, '*_S4.npy')))
    all_files = [os.path.basename(f) for f in all_s4]
    
    if len(all_files) == 0:
        raise RuntimeError("Dataset is empty. Check ml_dir path.")
        
    # Image-level split to prevent spatial data leakage
    random.seed(42)
    random.shuffle(all_files)
    split_idx = max(1, int(len(all_files) * TRAIN_VAL_SPLIT))
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]
    
    if not val_files:
        val_files = train_files
        
    print(f"  [Train] Image split: {len(train_files)} train images, {len(val_files)} val images")

    train_ds = SyntheticNoiseDataset(
        ml_dir=ml_dir, file_list=train_files,
        K=K, patch_h=PATCH_H, patch_w=PATCH_W, stride=PATCH_STRIDE,
        augment=True,
    )
    
    val_ds = SyntheticNoiseDataset(
        ml_dir=ml_dir, file_list=val_files,
        K=K, patch_h=PATCH_H, patch_w=PATCH_W, stride=PATCH_STRIDE,
        augment=False,
    )

    n_workers = 0  # Disable multiprocessing to avoid macOS PyTorch shm_manager permission issues
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=n_workers, pin_memory=True,
                          persistent_workers=(n_workers > 0))
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=n_workers, pin_memory=True,
                          persistent_workers=(n_workers > 0))

    print(f"  [Train] {len(train_ds)} train patches / {len(val_ds)} val patches")

    from hsid_cnn.dataset import load_noise_config
    metadata = {
        'seed': 42,
        'train_list': train_files,
        'val_list': val_files,
        'hyperparameters': {
            'lr': lr, 'batch_size': batch_size, 'epochs': epochs,
            'weight_decay': WEIGHT_DECAY, 'adam_betas': ADAM_BETAS, 'adam_eps': ADAM_EPS,
            'scheduler_step': LR_STEP_SIZE, 'scheduler_gamma': LR_FACTOR
        },
        'noise_config': load_noise_config(),
        'model_config': {
            'K_ADJACENT': K, 'N_FEAT_SCALE': n_feat, 'N_BACKBONE_CH': n_ch
        }
    }

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(K=K, n_feat=n_feat, n_ch=n_ch).to(device)
    print(f"  [Train] {model}")

    start_epoch = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"  [Train] Resumed from {resume} (epoch {start_epoch})")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        betas=ADAM_BETAS, eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_STEP_SIZE, min_lr=1e-6,
    )
    criterion = HSIDLoss(lambda_l1=0.1, lambda_mean=0.05)

    # AMP scaler for CUDA only
    scaler = (torch.cuda.amp.GradScaler()
              if device.type == 'cuda' else None)

    # ── Training history ──────────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'val_mpsnr': [], 'val_mssim': []}
    best_val_loss = float('inf')
    patience_ctr = 0
    best_ckpt_path   = os.path.join(ckpt_dir, 'hsid_cnn_best.pth')

    print(f"\n  [Train] Starting training for {epochs} epochs")
    print(f"  {'Epoch':>6} {'TrainLoss':>12} {'ValLoss':>10} "
          f"{'MPSNR':>8} {'MSSIM':>8} {'LR':>10}")
    print("  " + "─" * 62)

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_ld, optimizer, criterion,
                                 device, scaler)
        val_metrics = validate_epoch(model, val_ld, criterion, device)

        scheduler.step(val_metrics['loss'])
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'  ].append(val_metrics['loss'])
        history['val_mpsnr' ].append(val_metrics['mpsnr'])
        history['val_mssim' ].append(val_metrics['mssim'])

        print(f"  {epoch+1:>6} {train_loss:>12.6f} {val_metrics['loss']:>10.6f} "
              f"{val_metrics['mpsnr']:>8.3f} {val_metrics['mssim']:>8.4f} "
              f"{cur_lr:>10.2e}  "
              f"[{time.time()-t0:.1f}s]")

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_ctr = 0
            torch.save({
                'epoch':        epoch + 1,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict':    optimizer.state_dict(),
                'val_loss':     best_val_loss,
                'metadata':     metadata
            }, best_ckpt_path)
        else:
            patience_ctr += 1

        # ── Early stopping ────────────────────────────────────────────────
        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"\n  [Train] Early stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ── Save final model & history ─────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, 'hsid_cnn_final.pth')
    torch.save({
        'epoch': epoch + 1, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'metadata': metadata
    }, final_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    hist_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump({'metadata': metadata, 'history': history}, f, indent=2)

    print(f"\n  ✅ Training complete.")
    print(f"     Best checkpoint : {best_ckpt_path}")
    print(f"     Final model     : {final_path}")
    print(f"     History         : {hist_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description='Train HSID-CNN on synthetic noise data')
    p.add_argument('--ml_dir',      default=ML_DIR)
    p.add_argument('--ckpt_dir',    default=CKPT_DIR)
    p.add_argument('--epochs',      type=int,   default=EPOCHS)
    p.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',          type=float, default=LR)
    p.add_argument('--K',           type=int,   default=K_ADJACENT)
    p.add_argument('--n_feat',      type=int,   default=N_FEAT_SCALE)
    p.add_argument('--n_ch',        type=int,   default=N_BACKBONE_CH)
    p.add_argument('--no_inject',   action='store_true')
    p.add_argument('--resume',      default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    run_training(
        ml_dir       = args.ml_dir,
        ckpt_dir     = args.ckpt_dir,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        K            = args.K,
        n_feat       = args.n_feat,
        n_ch         = args.n_ch,
        inject_noise = not args.no_inject,
        resume       = args.resume,
    )
