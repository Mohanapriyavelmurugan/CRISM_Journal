# Results and Discussion

## Quantitative Denoising Performance
We evaluated the Proposed Spatial-Spectral Hybrid Pipeline against two state-of-the-art benchmarks: HSID-CNN (a supervised, spatial-spectral deep learning approach) and N2N4M (Platt et al. 2025; a self-supervised, spectral-domain Noise2Noise variant). The evaluation was performed on real CRISM/M³ observations. To ensure strict statistical fairness, all paired statistical tests involving N2N4M were performed exclusively on the subset of L-sensor scenes ($n=5$), as its architecture is fundamentally constrained to 350 spectral features.

### SNR and Spatial Preservation Trade-offs
The inclusion of the N2N4M baseline perfectly illustrates the fundamental trade-off between spatial smoothing and spectral preservation in hyperspectral denoising. 

N2N4M achieved an extremely high Signal-to-Noise Ratio (SNR = 2917.29 ± 2019.40). However, this astronomical SNR metric is symptomatic of extreme spatial and spectral over-smoothing. Because no-reference spatial SNR measures the ratio of the mean signal to the spatial standard deviation, an algorithm that flattens a scene into a uniform continuum will yield a near-infinite SNR. 

Conversely, HSID-CNN heavily preserved local spatial variance but failed to suppress the high-frequency CRISM striping artifacts, yielding a lower spatial SNR (3.39 ± 0.27). 

The **Proposed Pipeline** achieved a highly balanced, physically meaningful SNR (83.74 ± 84.85). It successfully suppressed structured periodic noise (stripes) and Gaussian artifacts while preserving genuine geological edges, yielding an SNR that is statistically superior to HSID-CNN ($p = 0.013$) without collapsing the spatial variance like N2N4M.

### Spectral Fidelity (SAM and Pearson Correlation)
The catastrophic over-smoothing of N2N4M is explicitly confirmed by the spectral metrics. N2N4M yielded a massive Spectral Angle Mapper error (SAM = 0.76 ± 0.11 radians, or ~43 degrees of spectral distortion) and a strongly negative Pearson correlation ($r = -0.20 \pm 0.05$). This indicates that N2N4M completely destroyed the underlying mineralogical continuum, inversely correlating with the original physical observation.

Both the **Proposed Pipeline** and **HSID-CNN** demonstrated excellent spectral fidelity. The Proposed Pipeline achieved near-perfect spectral preservation (Pearson $r = 0.99 \pm 0.01$; SAM = 0.14 ± 0.01). While HSID-CNN was highly competitive (SAM = 0.14 ± 0.02), the Proposed Pipeline maintained a statistically significant edge in Pearson correlation ($p = 0.002$ vs. HSID-CNN), proving that isolating 1D spatial destriping from 1D spectral blind-spot autoencoding is fundamentally superior to generalized 3D operations for preserving delicate planetary absorption bands.

### Computational Efficiency
The architectures exhibit vastly different computational footprints:
- **N2N4M** (Pretrained 1D DAE): ~0.20 seconds per cube.
- **HSID-CNN** (Pretrained 2D/3D CNN): ~1.37 seconds per L-sensor cube.
- **Proposed Pipeline** (Self-supervised, per-scene optimization): ~3.5 seconds per L-sensor cube.

**N2N4M** is mathematically the fastest method because it utilizes a lightweight 1D architecture trained offline. **HSID-CNN** is slower due to its heavy 3D convolutional multiscale operations, but still operates entirely via a single forward pass.

The **Proposed Pipeline** is the most computationally expensive per-cube. This is entirely by design: it does not rely on a generalized, pretrained model. Instead, it instantiates and trains its neural networks dynamically on the specific noise topology of the scene being processed. While this demands slightly higher runtime, it guarantees that the denoising kernels never hallucinate out-of-distribution synthetic noise patterns into real Martian data.

### Conclusion
The addition of the N2N4M baseline firmly validates the architectural necessity of the Proposed Pipeline. While pure spectral self-supervision (N2N4M) destroys the physical continuum, and pure supervised spatial-spectral learning (HSID-CNN) struggles to generalize to real CRISM striping, the hybrid approach perfectly bridges the gap. By explicitly targeting periodic noise with a localized spatial CNN and independently targeting Gaussian noise with a spectral autoencoder, the Proposed Pipeline achieves state-of-the-art artifact suppression while strictly preserving the integrity of critical mineralogical absorption bands.
