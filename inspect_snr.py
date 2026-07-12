import numpy as np
from hsid_cnn.metrics import compute_snr
import os

cube_s3 = np.load('test/physics_output/frt00014679_02_if155l_trr3_S3.npy')
cube_s4 = np.load('test/ml_output/frt00014679_02_if155l_trr3_S4.npy')
cube_hs = np.load('test/hsid_output/frt00014679_02_if155l_trr3_HSID.npy')

def get_stats(cube):
    means, stds = [], []
    for b in range(cube.shape[2]):
        v = cube[:, :, b]
        v = v[np.isfinite(v) & (v > -0.05) & (v < 1.30)]
        if v.size > 1:
            means.append(v.mean())
            stds.append(v.std())
    return np.mean(means), np.mean(stds)

m3, s3 = get_stats(cube_s3)
m4, s4 = get_stats(cube_s4)
mh, sh = get_stats(cube_hs)

print(f"S3 (Noisy)   : mean={m3:.4f}, std={s3:.4f}, SNR={m3/s3:.4f}")
print(f"S4 (Pipeline): mean={m4:.4f}, std={s4:.4f}, SNR={m4/s4:.4f}")
print(f"HSID (CNN)   : mean={mh:.4f}, std={sh:.4f}, SNR={mh/sh:.4f}")

# Check unclipped min/max to see if HSID CNN outputs are blowing up
print(f"HSID min={np.nanmin(cube_hs):.4f}, max={np.nanmax(cube_hs):.4f}")
