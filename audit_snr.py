import numpy as np

# Load cubes
stem = "frt00009907_01_if158l_trr3"
s3 = np.load(f"test/physics_output/{stem}_S3.npy")
s4 = np.load(f"test/ml_output/{stem}_S4.npy")
hsid = np.load(f"test/hsid_output/{stem}_HSID.npy")
n2n = np.load(f"results/n2n4m_output/{stem}_N2N4M.npy")

print(f"=== SNR Variance Audit for {stem} ===")

# Pick an arbitrary band in the middle (e.g., band 100)
b = 100

def get_stats(cube, name):
    v = cube[:, :, b]
    v = v[np.isfinite(v) & (v > -0.05) & (v < 1.30)]
    mean = abs(v.mean())
    std = v.std()
    snr = mean / (std + 1e-8)
    print(f"{name:15} | Mean: {mean:.6f} | Std (Variance term): {std:.6f} | Band SNR: {snr:.2f}")

get_stats(s3, "Original (S3)")
get_stats(s4, "Proposed (S4)")
get_stats(hsid, "HSID-CNN")
get_stats(n2n, "N2N4M")
