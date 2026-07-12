# 4. Results and Discussion

## 4.1 Quantitative Denoising Performance
We evaluated the proposed spatial-spectral hybrid pipeline against the deep learning-based HSID-CNN (Yuan et al., 2019). The HSID-CNN architecture was trained on proxy data augmented with representative CRISM artifact distributions (band-varying Gaussian noise and non-periodic striping) to ensure an unbiased, independent evaluation. 

The evaluation was performed on real CRISM/M³ data. In the absence of pristine ground truth, we utilized no-reference metrics including the Signal-to-Noise Ratio (SNR) improvement and the Spectral Angle Mapper (SAM). The proposed hybrid pipeline demonstrated robust preservation of spatial structure while significantly suppressing structured noise components.

*Please refer to Table 1 for detailed quantitative comparisons.*

## 4.2 Spectral Signature Preservation
A critical requirement for planetary hyperspectral imagery is the absolute preservation of mineralogical absorption bands (e.g., mafic mineral absorptions around 1.0 µm and 2.0 µm). Over-smoothing induced by aggressive denoising algorithms can artificially shallow these bands, skewing downstream geological analyses.

As illustrated in Figure X, the proposed pipeline exhibits superior spectral fidelity compared to HSID-CNN. While HSID-CNN effectively reduces high-frequency spatial noise, the localized spectral profiles reveal minor distortions in the primary absorption troughs. In contrast, the proposed self-supervised spectral Autoencoder preserves the original band depths while removing the high-frequency uncorrelated noise.

## 4.3 Computational Efficiency and Hardware Feasibility
Runtime and memory footprint are practical considerations for planetary data processing pipelines. While HSID-CNN leverages highly parallelized 2D/3D convolutional operators on GPU hardware (MPS), its massive parameter count and multiscale feature extraction introduce substantial memory overhead (Peak Memory: X MB) and latency. 

The proposed pipeline, built on a lightweight 1D architecture and deterministc physics-based formulations, achieves competitive denoising performance with a fraction of the computational requirements, making it highly suitable for large-scale data processing campaigns.

---
**Table 1 Caption:** Quantitative comparison of denoising performance between the physics-based correction (S3), HSID-CNN, and the proposed hybrid pipeline. SNR, SAM (in radians), and Pearson correlation ($r$) are computed against the original noisy input (Stage 2).
**Figure X Caption:** Comprehensive visual evaluation of CRISM denoising. (Row 1) Original noisy observation (S2). (Row 2) Physics-based correction (S3). (Row 3) HSID-CNN output (processing S3). (Row 4) Proposed pipeline output (processing S3). (Rows 5-6) Difference residuals emphasizing removed structural artifacts. (Row 7) Gradient magnitude maps highlighting preserved spatial structures. (Row 8) Inset magnification of challenging topologies. 
**Figure Y Caption:** Spectral signature preservation for representative ROIs. The proposed method strictly adheres to the original continuum slope and preserves the critical 1.0 µm and 2.0 µm absorption depths without hallucinating morphological artifacts.
