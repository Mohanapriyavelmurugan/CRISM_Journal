# CRISM Mineral Identification — Exact Code Explainer

**Scope:** What the code in this repository **actually does** after the wavelength / denoising / mineral-ID fixes.  
**Not speculation:** every step below maps to a named function or constant in the source files.  
**Primary files:** `crism_utils.py`, `Stage3_physics.py`, `Stage4_denoising.py`, `Stage5_minerals.py`, `run_mineral_fullfrt.py`

---

## 1. End-to-end order of execution

When you run:

```text
python run_mineral_fullfrt.py --scene 1
```

or

```text
python resume_mineral_scene1.py
```

the mineral pipeline calls `Stage5_minerals.run_mineral_identification()`, which runs **in this order**:

| Order | Function | What it does |
|------:|----------|--------------|
| 0 | `stage0_setup` | Load cube, wavelengths, masks |
| 1 | `stage1_library` | Load USGS CRISM-convolved library → 438 bands, CR, SPM |
| 2 | `stage2_scene_spm` | Bland ratio, SPM maps, continuum-removed cube, training pixels |
| 3 | `stage3_encoder` | Train contrastive encoder; encode all valid pixels → latent vectors |
| 4 | `stage4_hdbscan` | Cluster SPM-active pixels; name clusters (hybrid SPM + library SAM) |
| 5 | `stage5_unmixing` | Softmax abundance network with frozen library endmembers |
| 6 | `stage6_smoothing` | SLIC vote on clustered pixels only |
| — | `detect_rare_minerals` | Extra rare-mineral scans |
| 7 | `stage7_validation` | Figures + `mineral_identification_report.json` |
| 7b | `write_mineral_proof` | `mineral_proof.json` + `fig7_mineral_proof.png` |

Stages 1–4 of the **broader** pipeline (EDA → noise → physics → ML denoise) are separate scripts. Mineral ID **prefers** their product `*_S4.npy`.

---

## 2. Wavelength axis (`crism_utils.py`)

### What the code does

1. `get_sensor_waves(n_bands, label_waves, data_dir, cube)`  
   - If the PDS label has `BAND_BIN_CENTER` of length `n_bands`, use that.  
   - Else if `data_dir/wavelengths_L438.npy` exists, load it.  
   - Else for 438-band L cubes, call `build_crism_L_waves_usgs(438)`.

2. `build_crism_L_waves_usgs`  
   - Reads USGS file  
     `ASCIIdata_splib07b_cvCRISMjMTR3/s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt`  
     (or `USGS_CRISM_WAVES` env var).  
   - Takes median spacing of channels with λ > 1.0 µm (≈ 0.0066101 µm/channel).  
   - Builds a **descending** axis: band 0 = long wavelength, band 437 = short wavelength.  
   - Places the short-λ end so the mid-detector invalid run (~band 175.5) sits near 2.75 µm (saturated CO₂).

3. `assert_crism_L_wavelengths(waves, cube)`  
   - Raises if axis is ascending.  
   - Raises if endpoints are not roughly ~3.9 → ~1.0 µm.  
   - If `cube` is given: deepest scene-median absorption must fall in 2.55–2.85 µm.

4. Caches successful L axis to `data_dir/wavelengths_L438.npy`.

### What this is **not**

- Not a mineral match.  
- Not “using Earth geology as Mars truth.”  
- The USGS file is used as a **CRISM channel wavelength list** packaged with the CRISM-convolved library.

### Bad-band mask (`build_bad_mask`)

Hard excludes only (current `BAD_BAND_REGIONS_L`):

- 0.970–1.025 µm (cut-on)  
- 1.370–1.420 µm (telluric H₂O)  
- 1.980–2.070 µm (CO₂ core)  
- 2.650–2.800 µm (saturated CO₂)  
- 3.500–4.000 µm (thermal)

Mineral diagnostics near ~1.05, ~1.93, ~2.21 µm are **not** blanked by this mask.

---

## 3. How the denoised cube is built (`Stage3` + `Stage4`)

### Stage 3 — `run_physics_correction`

On the raw TRR3 `.img`:

1. Build wavelengths via `get_sensor_waves(..., data_dir, cube)`.  
2. Set hard-bad bands to NaN.  
3. Illumination correction on good bands.  
4. Spike / saturation / thermal corrections as listed in `noise_map.json`.  
5. Multiplicative atmospheric anchor correction.  
6. Save `physics_output/<scene>_S3.npy`.

### Stage 4 — `run_ml_denoising`

Starts from `*_S3.npy`:

1. **N1 stripes:** `classical_destripe(..., only_bands=stripe_bands)`  
   - Only Stage-2 striped bands (not all good bands).  
   - Offset = column mean minus smooth cubic cross-track trend (capped).  
   - Optional 1D stripe CNN residual on those bands.

2. **N2 Gaussian:** Spectral DAE on `high_noise_bands` + up to **40** `medium_noise_bands` (capped in code to avoid over-smoothing).

3. Restore any non-bad band that became NaN from S3 values.

4. Soft global gain if mean S4/S3 median ratio differs from 1.0 by > 2%.

5. `validate_denoising` → PASS / REVIEW / FAIL.  
   - FAIL → revert to S3.  
   - REVIEW → 70% ML + 30% S3 blend, re-validate.

6. Save `ml_output/<scene>_S4.npy` and `stage4_validation.json`.

### Stage 4 validation fields written (exact keys)

Examples from Scene 1 after rebuild:

- `snr_before`, `snr_after`, `snr_delta_pct`  
- `sam_rad`  
- `status`  
- `s4_s3_ratio_mean`, `s4_s3_ratio_std`  
- `diagnostic_bands_restored`  
- `finite_frac_good`  
- `output_path`

---

## 4. Mineral ID Stage 0 — which cube is loaded

`stage0_setup`:

1. Prefer `ml_output/**/*_S4.npy` → `ctx["cube_denoised"]`, `cube_source = "S4:..."`.  
2. Else S3, else raw `.img`.  
3. Always try to load matching TRR3 `.img` as `ctx["cube_spm"]` for backup SPM.  
4. Builds:
   - `bad_mask` (`build_bad_mask`)  
   - `spm_bad_mask` / `mineral_bad_mask` (`SPM_HARD_EXCLUDE` / `MINERAL_CR_EXCLUDE`)  
   - `valid_pixel_mask` (≥ 50% finite good bands)

---

## 5. Stage 1 — library (`stage1_library`)

1. Load USGS wavelengths (489 joined channels).  
2. Map each of the 438 scene wavelengths to nearest library channel within 0.010 µm.  
3. For each mineral/soil ASCII spectrum that passes `is_quality_spectrum`:  
   - Fill NaNs, compute SPM vector,  
   - Continuum-remove with **`continuum_removal_match`** (same algorithm as the scene),  
   - L2-normalize CR.  
4. Saves `lib_spectra_438_cr.npy`, `lib_spectra_438_cr_l2.npy`, `lib_spm.npy`, `lib_names.json`.

**Note:** Library is already CRISM-convolved (`cvCRISMjMTR3`). Code does **not** apply an extra SRF.

---

## 6. Stage 2 — SPM + continuum removal (`stage2_scene_spm`)

Exact sequence:

1. Mask edge columns (`EDGE_COL_MARGIN = 20`).  
2. Fill finite holes in IMG and S4 (except skip-masked bands).  
3. Copy diagnostic wavelengths into S4 from IMG where S4 is NaN  
   (1.054, 1.330, 1.810, 1.930, 2.120, 2.210, 2.300 µm).  
4. If cube source starts with `S4`: **skip** column destripe on S4; destripe IMG backup only.  
5. Estimate bland spectra separately on IMG and S4 (`BLAND_PCT = 20`).  
6. Apply bland ratio → `cube_spm` (IMG), `cube_ratio` (S4).  
7. **Primary SPM** on `cube_ratio` (S4).  
8. For any index not `ok*` in `sanitize_spm_map`, try IMG ratio; for OLINDEX3 optionally 3-point map on S4.  
9. `build_adaptive_spm_active`:  
   - Threshold = absolute `SPM_DETECT_THRESH[name]` (percentile is **not** used as a floor when `SPM_GATE_USE_PERCENTILE_FLOOR = False`).  
   - Keep connected components ≥ `SPM_MIN_COMPONENT` (50).  
   - Require hot median > scene median + max(0.5·scene_std, 0.5·thr).  
10. Continuum removal: **`continuum_removal_match_cube` on bland-ratioed S4** (`CR_MATCH_FILTER_SIZE = 41`).  
11. Interpolate saturated CO₂ window 2.65–2.80 µm in CR (`suppress_cr_artifact_notch`).  
12. L2-normalize → `cube_cr_l2`.  
13. Sample up to 12 000 training pixels from SPM-active domain.  
14. Sets `ctx["cr_source"] = "bland_ratioed_S4"`.

### SPM indices computed (`SPM_INDICES` / `compute_spm_map_vectorized`)

| Name | Type in code |
|------|----------------|
| BD1900_2, BD2100_2, BD2210_2, BD2290, BD2300, BD2500_2 | 3-point band depth |
| OLINDEX3 | Multipoint weighted reflectance formula in code |
| LCPINDEX2, HCPINDEX2, SINDEX2 | 3-point band depth |
| BD1500_2, BD1435 | 3-point; excluded from SPM-active domain (`SPM_ACTIVE_EXCLUDE`) |

`sanitize_spm_map` can set an index to NaN with notes such as `disabled_low_contrast`, `disabled_scene_wide_artifact`, etc.

---

## 7. Stage 3 — contrastive encoder (`stage3_encoder`)

### Network (`SpectralEncoder`)

- Input: `(batch, 1, 438)` continuum-removed L2 spectra.  
- 1D CNN → GAP → FC → **latent dim = 128** (`LATENT_DIM`).  
- Heads:  
  - `predict_spm` → 12 SPM values (sigmoid)  
  - `reconstruct` → 438-band reconstruction  

### Training loss (exact combination in code)

For each batch:

1. Two augmentations of each spectrum (`augment_spectrum`: scale, optional smooth, drop ~5% unprotected bands).  
2. `loss_c = nt_xent_loss(z1, z2)` (NT-Xent, temperature 0.07).  
3. `loss_s = MSE(predict_spm(z1), target_spm)`.  
4. `loss_r = MSE(reconstruct(z1)[protected_bands], spectrum[protected_bands])`.  
5. **`loss = loss_c + 0.1 * loss_s + 0.2 * loss_r`**.

Training pool = scene training pixels + library samples + synthetic library mixes.

After training: encode all valid pixels → `latent_vectors.npy` (L2-normalized rows). Save `encoder_B438.pth`.

### What compression does and does not do

- Latent is used for **clustering / unmixing input**, not as the sole mineral proof.  
- Reconstruction loss + protected diagnostic wavelengths + SPM head are explicit mechanisms in code to keep mineral-relevant structure in the 128-D code.  
- Library SAM and proof band depths still use full CR spectra (`cube_cr`), not the 128-D vector alone.

---

## 8. Stage 4 — clustering (`stage4_hdbscan`)

1. Build features per valid pixel:  
   `[latent (128) | 0.35 * z-scored SPM (12, ice downweighted) | 0.25 * z-scored diagnostic CR bands at 1.054,1.330,1.930,2.120,2.210,2.300 | 0.12 * normalized xy]`.  
2. PCA to ≤16 components.  
3. HDBSCAN **only on SPM-active pixels** (bland stays label −1).  
4. Optional UMAP only if `CRISM_USE_UMAP=1` (default off).  
5. Stripe-like clusters filtered (`STRIPE_X_STD_MAX`, `STRIPE_Y_RANGE_MIN`).  
6. For each cluster: mean CR spectrum → `match_library_topk` + SPM enrichment → `hybrid_cluster_label`.  
   - If SAM is tied / non-discriminative → **SPM group preferred** (`spm_first_tied_sam`).  
7. Saves `label_map.npy`, `cluster_names.json`, etc.

---

## 9. Stage 5 — unmixing (`stage5_unmixing`)

1. Select library endmembers (SPM-supported seeds + farthest-point sampling; prune unused).  
2. `AbundanceNetwork`: latent → hidden → softmax abundances (ASC: ≥0, sum≈1).  
3. Library endmember matrix frozen in phase A (`UNMIX_FREEZE_E_PHASE_A = True`).  
4. Outputs `abundance_map.npy`, `endmember_names.json`, reconstruction error map.

**Exact meaning of abundance values in code:** softmax mixture weights under the learned model.  
They are **not** independently validated laboratory mass percentages.

---

## 10. Stage 6 — smoothing (`stage6_smoothing`)

1. SLIC segments from selected CR channels.  
2. Majority vote of cluster labels **only where label ≥ 0**.  
3. Noise (−1) stays unlabeled.

---

## 11. How the code decides `present: true` (`write_mineral_proof`)

For each SPM index that passed the Stage-2 gate:

1. Build hotspot mask: valid & SPM ≥ `thr_adaptive` (absolute threshold).  
2. Mean CR of hotspot → band depths via `simple_band_depth` at fixed windows (1.05, 1.33, 1.93, 2.12, 2.21, 2.30 µm).  
3. Also depths on a bland mask (lowest 10% of max-SPM).  
4. Library top-5 via `match_library_topk` (Mars-prior names, `sam_mask`).  
5. Cluster overlap recorded for reporting.  
6. **`spectral_hits`** = CR-depth / hot>bland / library-family strings only.  
7. **PRESENT rules in code:**  
   - OLINDEX3: `n_hot ≥ max(250, 500)` **and** `len(spectral_hits) ≥ 1`.  
   - Other indices: `n_hot ≥ 50` **and** `len(spectral_hits) ≥ 1`.  
8. Cluster overlap is **logged** but **does not alone** set `present = true`.

Writes `mineral_proof.json` and `fig7_mineral_proof.png`.  
`minerals_present` in the report = groups with `present: true`.

---

## 12. Constants that control behavior (current)

| Constant | Value | File |
|----------|------:|------|
| `LATENT_DIM` | 128 | `Stage5_minerals.py` |
| `ENCODER_EPOCHS` | 80 | same |
| `CR_MATCH_FILTER_SIZE` | 41 | same |
| `EDGE_COL_MARGIN` | 20 | same |
| `BLAND_PCT` | 20 | same |
| `SPM_MIN_COMPONENT` | 50 | same |
| `SPM_GATE_USE_PERCENTILE_FLOOR` | False | same |
| `SAM_MARGIN_STRONG` | 0.015 rad | same |
| Stage4 medium_noise cap | 40 bands | `Stage4_denoising.py` |

---

## 13. Outputs written under `mineral_output/`

| File | Produced by |
|------|-------------|
| `wavelengths_L438.npy` | `crism_utils` (scene root) |
| `*_S4.npy` | Stage 4 (`ml_output/`) |
| `cube_cr.npy`, `cube_cr_l2.npy` | Stage 2 |
| `spm_map.npy`, `spm_gate_info.json`, `spm_sanity.json` | Stage 2 |
| `encoder_B438.pth`, `latent_vectors.npy` | Stage 3 |
| `label_map.npy`, `cluster_names.json` | Stage 4 |
| `abundance_map.npy`, `endmember_names.json` | Stage 5 |
| `smoothed_label_map.npy` | Stage 6 |
| `mineral_proof.json`, `fig7_mineral_proof.png` | `write_mineral_proof` |
| `mineral_identification_report.json` | Stage 7 |
| `fig1`–`fig6` | Stage 7 |

---

## 14. How to run (exact)

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CRISM_USE_UMAP = "0"

# Full rebuild Stages 1–5 for Scene 1
python -u rebuild_scene1_fixed.py

# Or mineral ID only (requires existing S4)
python -u resume_mineral_scene1.py

# Or orchestrator style
python run_mineral_fullfrt.py --scene 1
```

---

## 15. What the code does **not** claim

- It does not download or use the WA CDR unless you add it later.  
- It does not treat abundance maps as certified mineral percentages.  
- It does not require an external geologic map (labels remain provisional in the report).  
- It does not use UMAP unless `CRISM_USE_UMAP=1`.  
- If an SPM index is `disabled_low_contrast` after sanitize (e.g. OLINDEX3 on a weak scene), that index does not enter the SPM-active domain and cannot become `present` via that index.

---

## 16. Scene 1 measured run (after these fixes)

The following are **from the completed log / JSON**, not assumptions:

| Item | Value |
|------|--------|
| Cube used | `ml_output/frt0000bf9e_07_if165l_trr3_S4.npy` |
| `cr_source` | `bland_ratioed_S4` |
| Wavelengths | descending 3.892 → 1.004 µm; deepest median absorption @ 2.802 µm |
| Stage4 S4 status | `PASS`; `s4_s3_ratio_mean≈1.000`; `s4_s3_ratio_std≈0.0026`; `finite_frac_good≈0.931` |
| Diagnostic CR finite @ 1.054–2.300 µm | 100% of valid pixels |
| Latent dim | 128 |
| Clusters after merge | 1 |
| Overall verdict | `WEAK` |
| `minerals_present` | `["Al-OH Clays"]` only |
| OLINDEX3 / LCPINDEX2 | `disabled_low_contrast` (IMG 3-pt OLINDEX p90=0.008 < 0.02) |
| Proof | Al-OH Clays PRESENT with `CR_depth_d2.21=0.0066` and `d2.21_hot>bland`; other gated SPM groups stayed `CANDIDATE/WEAK` without spectral hits |

Explainer source for this table: `mineral_resume.log`, `mineral_identification_report.json`, `stage4_validation.json`, `mineral_proof.json`.

---

*Generated to document the implemented pipeline behavior. If code and this file disagree, trust the Python source and update this file.*
