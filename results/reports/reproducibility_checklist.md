# Reproducibility Checklist

To ensure full reproducibility of the experimental methodology and statistical evaluation presented in this manuscript, the following software, hardware, and configuration parameters are explicitly documented.

## 1. Software Environment
- **Operating System:** macOS (Apple Silicon architecture)
- **Python Version:** 3.13
- **PyTorch:** 2.10.0 (Compiled with Metal Performance Shaders `mps` support)
- **NumPy:** 2.3.1
- **SciPy:** 1.16.0
- **scikit-learn:** 1.7.0

## 2. Hardware Environment
- **Processing Unit:** Apple Silicon (M-series)
- **Device Backend:** PyTorch MPS (Metal Performance Shaders) for GPU-accelerated inference.

## 3. Dataset Configuration
- **Total Evaluated CRISM TRR3 Scenes:** 11
- **L-sensor Scenes (350 clipped bands):** 5 (`frt000098f7`, `frt000098fe`, `frt00009907`, `frt0001458f`, `frt00014679_02`)
- **S-sensor Scenes (107 bands):** 6
- **Random Seed:** `42` (Enforced universally for `numpy.random`, `torch.manual_seed`, and `random.seed` during HSID-CNN synthetic data generation and data loader instantiation).
- **Train/Validation Split (HSID-CNN):** 80% Training / 20% Validation split strictly separated at the independent spatial image level to avoid spatial data leakage.

## 4. Hyperparameters and Training Configuration
### Proposed Pipeline (Self-Supervised Spectral DAE)
- **Architecture:** 1D Spectral Autoencoder
- **Optimizer:** Adam
- **Learning Rate:** Dynamically scaled per scene.
- **Batch Size:** Processed entirely in memory via localized spatial chunks.

### HSID-CNN Benchmark (Supervised)
- **Architecture:** 3D Spatial-Spectral CNN
- **Optimizer:** Adam
- **Learning Rate:** $1 \times 10^{-4}$
- **Batch Size:** 64
- **Epochs:** 50
- **Early Stopping:** Triggered dynamically via validation loss plateaus.
- **Noise Model:** Additive Gaussian and Periodic Striping matching physical empirical distributions.

### N2N4M Benchmark (Self-Supervised, Pretrained)
- **Architecture:** Noise2Noise1D (4 Blocks, Depth 3, Kernel 5, 350 Input Features)
- **Weights:** Official `trained_model_weights.pt` provided by Platt et al. (2025). No fine-tuning applied.
- **Normalization:** Scikit-learn `RobustScaler`. Custom implementation explicitly executes `inverse_transform` to enforce physical absolute I/F reflectance bounds.
- **Compatibility Restriction:** Evaluated exclusively on the 5 L-sensor scenes to strictly preserve the required 350 input dimensional structure.

## 5. Execution Workflow (Reproduction Commands)
To reproduce the complete experimental phase from raw data to final manuscript tables:

1. **Synthetic Noise Generation & Training (HSID-CNN):**
   ```bash
   python3 -m hsid_cnn.train
   ```
2. **Execute N2N4M Third-Party Inference:**
   ```bash
   python3 run_n2n4m.py
   ```
3. **Execute Proposed Pipeline, Generate Comprehensive Figures, and Compute Statistics:**
   ```bash
   python3 run_eval_pipeline.py
   ```

*Upon execution, all statistical tables (paired t-tests, Wilcoxon signed-rank, Cohen's d), high-DPI (600) publication figures, and metric CSVs are programmatically written to the `/results` and `hsid_cnn/figures` output directories.*
