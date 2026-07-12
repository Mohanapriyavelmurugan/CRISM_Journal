import sys
import os
import glob
import time
import json
import torch
import numpy as np

# Add n2n4m_repo to sys.path so we can import it
sys.path.insert(0, os.path.abspath('n2n4m_repo'))

from n2n4m.model_functions import predict
from n2n4m.n2n4m_denoise import clip_bands, combine_bands, instantiate_default_model, create_dataloader, load_scaler
import n2n4m.preprocessing as preprocessing

def run_n2n4m_inference(physics_dir="test/physics_output", out_dir="results/n2n4m_output"):
    os.makedirs(out_dir, exist_ok=True)
    
    s3_files = sorted(list(set(
        glob.glob(os.path.join(physics_dir, '**', '*_S3.npy'), recursive=True) +
        glob.glob(os.path.join(physics_dir, '*_S3.npy'))
    )))
    
    results = {}
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"[N2N4M] Using device: {device}")
    
    # Load model and scaler once
    scaler_path = "n2n4m_repo/n2n4m/data/n2n4m_feature_scaler.joblib"
    model_path = "n2n4m_repo/n2n4m/data/trained_model_weights.pt"
    
    print("[N2N4M] Loading pretrained model and scaler...")
    scaler = load_scaler(scaler_path)
    model = instantiate_default_model(model_path)
    model.to(device)
    
    for s3_path in s3_files:
        stem = os.path.basename(s3_path).replace('_S3.npy', '')
        out_path = os.path.join(out_dir, f"{stem}_N2N4M.npy")
        
        im_array = np.load(s3_path)
        im_shape = im_array.shape
        
        # Check sensor compatibility (only L-sensor 438 bands supported)
        if im_shape[-1] != 438:
            print(f"[N2N4M] Skipping {stem} (S-sensor, {im_shape[-1]} bands not supported by N2N4M)")
            continue
            
        print(f"[N2N4M] Processing {stem}...")
        
        # Start Profiling
        start_time = time.time()
        if device.type == 'mps':
            torch.mps.empty_cache()
            peak_mem_mb = 0.0
        elif device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            peak_mem_mb = 0.0
            
        # N2N4M Preprocessing
        spectra = im_array.reshape(-1, im_shape[-1])
        bands_to_denoise, additional_bands = clip_bands(spectra)
        bands_to_denoise, bad_value_mask = preprocessing.impute_bad_values_in_image(bands_to_denoise)
        
        # Scale
        bands_to_denoise = scaler.transform(bands_to_denoise)
        
        # DataLoader
        spectra_dataloader = create_dataloader(bands_to_denoise, batch_size=1000)
        
        # Predict
        denoised_spectra = predict(model, spectra_dataloader, device)
        denoised_spectra = denoised_spectra.cpu().numpy()
        
        # *** FIX INVERSE TRANSFORM BUG ***
        # The original code returned scaled outputs. We must inverse transform to physical I/F scale.
        denoised_spectra = scaler.inverse_transform(denoised_spectra)
        
        # Recombine with clipped bands
        denoised_spectra = combine_bands(denoised_spectra, additional_bands)
        denoised_image = denoised_spectra.reshape(*im_shape)
        
        # End Profiling
        elapsed_time = time.time() - start_time
        if device.type == 'mps':
            peak_mem_mb = torch.mps.driver_allocated_memory() / (1024 ** 2)
        elif device.type == 'cuda':
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        np.save(out_path, denoised_image)
        
        results[stem] = {
            'runtime_seconds': elapsed_time,
            'peak_memory_mb': peak_mem_mb
        }
        
        print(f"  -> Done in {elapsed_time:.2f}s, Peak Mem: {peak_mem_mb:.2f} MB")
        
    with open(os.path.join(out_dir, "n2n4m_profiling.json"), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    run_n2n4m_inference()
