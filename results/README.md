# Evaluation Pipeline Reproduction

This directory contains the final evaluation artifacts (Figures, Tables, and Report) for comparing the proposed hybrid denoising pipeline against the HSID-CNN architecture.

## Pipeline Architecture
- **Original Noisy (S2):** The raw cube after basic noise characterization but before physics correction. This is the common starting point.
- **Physics Output (S3):** The cube after physics-based correction (spike removal, thermal emission). This serves as the input to both ML models.
- **HSID-CNN:** Deep learning baseline. Takes S3 as input and generates the `_HSID.npy` output.
- **Proposed Method (S4):** Spectral Autoencoder + 1D CNN pipeline. Takes S3 as input and generates the `_S4.npy` output.

## How Figures Were Produced
- Run `python3 hsid_cnn/plot_comprehensive.py --s2 <S2> --s3 <S3> --hsid <HSID> --s4 <S4>`
- **comprehensive_bX.png**: Multi-panel visualization comparing the 4 stages, their residuals (Proposed - S3), spatial gradient maps (Sobel filter), and zoomed insets.
- **spectral_preservation.png**: Extracts spectra from 5 geographical quadrants. Includes a magnified inset of the 1.0/2.0 µm absorption features to demonstrate preservation of mineralogical signatures.

## How Tables Were Produced
- Run `python3 hsid_cnn/compare.py`
- It computes SNR, Spectral Angle Mapper (SAM), and Pearson Correlation ($r$) using S2 as the noisy reference.
- **per_cube_metrics.csv**: Contains the raw scores for every evaluated CRISM/M³ cube.
- **aggregate_comparison**: Averages the scores across all cubes and computes the Standard Deviation. Saved as CSV, Markdown, and LaTeX (`.tex`) for direct manuscript insertion.

## Reproduction
To reproduce this entire folder from scratch (without retraining the model), run:
```bash
python3 run_eval_pipeline.py
```
