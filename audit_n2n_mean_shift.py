import os, sys, torch
import numpy as np

# Add n2n4m_repo to sys.path so we can import it
sys.path.insert(0, os.path.abspath('n2n4m_repo'))

from n2n4m.model_functions import predict
from n2n4m.n2n4m_denoise import clip_bands, combine_bands, instantiate_default_model, create_dataloader, load_scaler
import n2n4m.preprocessing as preprocessing

stem = "frt00009907_01_if158l_trr3"
s3_path = f"test/physics_output/{stem}_S3.npy"

im_array = np.load(s3_path)
im_shape = im_array.shape

print("=== N2N4M Pipeline Mean Shift Audit ===")
print(f"Target Cube: {stem}")
print(f"Original S3 shape: {im_shape}")

# Band to audit (e.g. 100 which maps to some wavelength)
b = 100

spectra = im_array.reshape(-1, im_shape[-1])
print(f"\n[Step 0] Original input (Band {b}) Mean: {np.nanmean(spectra[:, b]):.6f}")

bands_to_denoise, additional_bands = clip_bands(spectra)
# Which index does band 100 map to in bands_to_denoise?
# Let's just track the whole array mean and the specific column if we can find it.
print(f"[Step 1] After clip_bands (All bands) Mean: {np.nanmean(bands_to_denoise):.6f}")

bands_to_denoise, bad_value_mask = preprocessing.impute_bad_values_in_image(bands_to_denoise)
print(f"[Step 2] After impute_bad_values (All bands) Mean: {np.nanmean(bands_to_denoise):.6f}")

scaler = load_scaler("n2n4m_repo/n2n4m/data/n2n4m_feature_scaler.joblib")
scaled_bands = scaler.transform(bands_to_denoise)
print(f"[Step 3] After scaler.transform (All bands) Mean: {np.nanmean(scaled_bands):.6f}")

# Model Predict
model = instantiate_default_model("n2n4m_repo/n2n4m/data/trained_model_weights.pt")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
spectra_dataloader = create_dataloader(scaled_bands, batch_size=1000)
pred_spectra = predict(model, spectra_dataloader, device).cpu().numpy()

print(f"\n[Step 4] After Model predict (All bands) Mean: {np.nanmean(pred_spectra):.6f}")
print(f"         Prediction differs from scaled input by mean shift: {np.nanmean(pred_spectra) - np.nanmean(scaled_bands):.6f}")

inverted_pred = scaler.inverse_transform(pred_spectra)
print(f"[Step 5] After scaler.inverse_transform (All bands) Mean: {np.nanmean(inverted_pred):.6f}")

denoised_spectra = combine_bands(inverted_pred, additional_bands)
print(f"[Step 6] After combine_bands (All bands) Mean: {np.nanmean(denoised_spectra):.6f}")
