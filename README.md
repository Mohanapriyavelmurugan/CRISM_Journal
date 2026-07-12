# CRISM Hybrid Denoising Pipeline vs HSID-CNN Baseline

This repository contains the evaluation suite comparing a novel proposed spatial-spectral hybrid denoising pipeline against an advanced baseline architecture (HSID-CNN) on real hyperspectral data from the CRISM and Moon Mineralogy Mapper (M³) instruments.

## Independent Baseline Methodology (HSID-CNN)

To ensure a scientifically fair and methodologically sound comparison between the proposed pipeline and the HSID-CNN baseline, we implemented an **Independent Supervised Proxy** training strategy for the neural network.

### Why is this necessary?
Unlike natural images, no pristine ground truth exists for hyperspectral planetary data (e.g., CRISM TRR3). If we trained the HSID-CNN to map from the raw noisy input (S3) directly to the proposed pipeline's output (S4), the CNN would merely act as an imitation model, learning to replicate the specific algorithmic choices of our pipeline rather than learning a generalized denoising mapping.

### The Training Protocol
Instead, HSID-CNN is trained as a completely independent supervised baseline using the following strategy:
1. **Clean Proxy Targets**: The high-SNR outputs from our proposed pipeline (Stage-4 / S4) are utilized strictly as "clean proxies" to provide valid spectral absorption features and structural targets. They are *not* used as ground truth for evaluating real data.
2. **Dynamic Synthetic Noise Injection**: During training, we generate realistic simulated CRISM noise (a combination of band-varying Gaussian noise, vertical stripping, dead pixels, detector line offsets, and Poisson shot noise). This synthetic degradation is injected dynamically, per-patch, on-the-fly (`hsid_cnn/dataset.py:__getitem__`).
3. **Generalization**: Because the synthetic noise is generated dynamically at runtime, every epoch exposes the model to a completely novel realization of the noise. The model learns to invert these physical degradations rather than memorizing a fixed dataset.
4. **Fair Normalization**: Normalization statistics (mean and standard deviation) are extracted directly from the synthesized noisy patches and applied identically to both the noisy and clean inputs. This mirrors the exact data constraints present during inference.

### Unbiased Evaluation Strategy
During the final inference stage, the trained HSID-CNN and the proposed hybrid pipeline both operate directly on the *exact same* real noisy CRISM cubes (Stage-3 / S3).

Neither model has seen the real noise distributions of the evaluation patches during training. 

**Metrics Reported:**
Because ground-truth pixels do not exist, we evaluate performance using no-reference and spectral fidelity metrics:
- **Signal-to-Noise Ratio (SNR)**
- **Spectral Angle Mapper (SAM)** (radians)
- **Pearson Correlation ($r$)**
- **Runtime** (seconds)
- **Peak GPU Memory** (MB)

*Note: PSNR and SSIM are explicitly omitted from real-data tables because they require pristine reference imagery, which is mathematically invalid to approximate using our pipeline's output.*

## Reproducibility
We mandate strict deterministic research protocols.
- **Random Seed**: `42` across Python `random`, `numpy.random`, and `torch.manual_seed`.
- **Metadata Logging**: Checkpoints and the `training_history.json` permanently embed the training/validation list splits, hyperparameters, seed, and exact synthetic noise generator parameters (found in `hsid_cnn/noise_config.json`).
- **Figures**: The script `run_eval_pipeline.py` orchestrates the complete execution from uncompiled inference through metric gathering and automated PDF/PNG publication-figure generation.