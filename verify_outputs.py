import os, glob, numpy as np

out_dir = "results/n2n4m_output"
files = glob.glob(os.path.join(out_dir, "*.npy"))

print("=== N2N4M Output Verification ===")
for f in sorted(files):
    arr = np.load(f)
    print(f"File: {os.path.basename(f)}")
    print(f"  Dimensions: {arr.shape}")
    print(f"  Datatype: {arr.dtype}")
    print(f"  Min: {np.nanmin(arr):.6f}")
    print(f"  Max: {np.nanmax(arr):.6f}")
    print(f"  Mean: {np.nanmean(arr):.6f}")
    print(f"  NaNs: {np.isnan(arr).sum()}")
    print(f"  Infs: {np.isinf(arr).sum()}")
    print("-" * 40)
