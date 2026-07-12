import os, json, random, glob
import numpy as np
import matplotlib.pyplot as plt
from hsid_cnn.dataset import load_noise_config, inject_synthetic_noise_patch
from hsid_cnn.config import ML_DIR, IF_MIN, IF_MAX

config = load_noise_config()
all_s4 = sorted(glob.glob(os.path.join(ML_DIR, '**', '*_S4.npy'), recursive=True) +
                glob.glob(os.path.join(ML_DIR, '*_S4.npy')))

if not all_s4:
    print("No S4 files found!")
    exit(1)

cube_path = all_s4[0]
clean_raw = np.load(cube_path).astype(np.float32)
clean_raw = np.clip(clean_raw, IF_MIN, IF_MAX)
clean_raw = np.nan_to_num(clean_raw, nan=0.0)

H, W, B = clean_raw.shape
# Pick a random patch for better visual (center of image)
ph, pw = 128, 128
r = max(0, H // 2 - ph // 2)
c = max(0, W // 2 - pw // 2)

band = min(B // 2, B-1)  # middle band
sp_clean = clean_raw[r:r+ph, c:c+pw, band].copy()

# Generate noise dynamically
sp_noisy = inject_synthetic_noise_patch(sp_clean, config, is_3d=False)

noise_diff = sp_noisy - sp_clean

out_dir = os.path.join('results', 'figures', 'synthetic_noise_validation')
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sp_clean, cmap='gray')
plt.title(f'Clean Proxy (S4) - Band {band}')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 3, 2)
plt.imshow(sp_noisy, cmap='gray')
plt.title(f'Synthetic Noisy Version')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 3, 3)
plt.imshow(noise_diff, cmap='RdBu', vmin=-np.max(np.abs(noise_diff)), vmax=np.max(np.abs(noise_diff)))
plt.title(f'Noise Difference')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
out_path = os.path.join(out_dir, f'noise_validation_band_{band}.png')
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Noise validation figure saved to {out_path}")
