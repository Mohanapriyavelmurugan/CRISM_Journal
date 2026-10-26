# CRISM Mineral Identification — Implementation Report

**Date:** 2026-10-04  
**File:** `Stage5_minerals.py`  
**Guide:** `CRISM_MINERAL_IDENTIFICATION_GUIDE.md`  
**Validated run:** Scene 12 (`frt0000991b_01_if156l_trr3`) on NVIDIA GTX 1650 (4 GB)

---

## What was implemented

A single linear pipeline (`run_mineral_identification`) covering Stages **0–7** plus rare-mineral detection. It is wired into `Run_pipeline.py` stage 5.

| Stage | Function | Role |
|------:|----------|------|
| 0 | `stage0_setup` | Load `*_S4.npy`, wavelengths, bad/dead masks, valid pixels |
| 1 | `stage1_library` | Parse USGS CRISM-convolved library; 489→438 subset; quality filter; CR + L2 |
| 2 | `stage2_scene_spm` | Block-wise continuum removal; 12 SPM maps; stratified training coords |
| 3 | `stage3_encoder` | `SpectralEncoder` + NT-Xent + SPM anchor; CUDA AMP; full-scene latents |
| 4 | `stage4_hdbscan` | HDBSCAN + DBCV; SPM naming + USGS SAM matching |
| 5 | `stage5_unmixing` | Farthest-point endmembers; `AbundanceNetwork` softmax; Phase A/B training |
| 6 | `stage6_smoothing` | SLIC superpixels; weighted majority vote; abundance renormalization |
| 7 | `stage7_validation` | Tiered validation, 6 figures, JSON report |
| + | `detect_rare_minerals` | Strategy 1 (noise SAM) + Strategy 2 (secondary abundance >10%) |

---

## Scientific methodology (followed + one accuracy fix)

### Followed from the guide

1. **No SRF convolution** on `ASCIIdata_splib07b_cvCRISMjMTR3` (already CRISM-convolved).
2. **489 → 438 channel mapping** by nearest wavelength (±10 nm).
3. **Bad-band aware SPM**: nearest good band within 30 nm, else NaN.
4. **Preprocessing order for matching/encoding:** NaN fill → modified upper-hull CR → restore bad/dead NaNs → L2.
5. **Contrastive encoder** with protected diagnostic-band dropout and NT-Xent (T=0.07).
6. **Softmax abundances** enforcing non-negativity and sum-to-one.
7. **Memory:** row blocks for CR; batch inference 2048; encoder batch 128 for 4 GB VRAM.

### Scientific accuracy adjustment (SPM domain)

Viviano-Beck et al. (2014) summary parameters are defined on **I/F reflectance**, not on spectra that are already continuum-removed. Computing band depths on CR spectra double-applies continuum and inflated broad indices (especially simplified OLINDEX3), which biased every cluster to “Olivine” in the first run.

**Implementation choice:**

- **SPM indices** are computed on filled **I/F / library reflectance**.
- **OLINDEX3** uses the multi-point Viviano-Beck formula from the guide (RB1054/1211/1329/1474/1750).
- **Encoder / library matching / unmixing** still use continuum-removed (+ L2 where specified) spectra, matching the guide’s spectral-comparison path.

---

## Dependencies added

```
hdbscan>=0.8.33
scikit-image>=0.19.0
```

Already present: `numpy`, `scipy`, `torch`, `matplotlib`, `scikit-learn`.

---

## How to run (identify minerals)

### Scene 12 (default / guide target)

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 12\Crism"
python Stage5_minerals.py
```

### Scene 11

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 11\Crism"
python Stage5_minerals.py --data-dir $env:CRISM_DATA_DIR
```

### Via orchestrator (Stage 5 only)

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 12\Crism"
python Run_pipeline.py --stages 5
```

**Prerequisite:** Stage 4 denoising products at `ml_output/*_S4.npy` (438-band L-sensor).

---

## Outputs (`mineral_output/`)

- Library: `lib_spectra_438_cr.npy`, `lib_spectra_438_cr_l2.npy`, `lib_spm.npy`, `lib_names.json`
- Scene: `cube_cr.npy`, `cube_cr_l2.npy`, `spm_map.npy`
- Models: `encoder_B438.pth`, `unmixer_B438.pth`
- Clustering: `label_map.npy`, `cluster_names.json`, `uncertainty_map.npy`
- Unmixing: `abundance_map.npy`, `smoothed_abundance_map.npy`, `dominant_map.npy`
- Rare minerals: `rare_mineral_detections.json`
- Validation: `mineral_identification_report.json`, `fig1`–`fig6` PNGs

---

## Scene 12 validation snapshot (final run)

| Metric | Value |
|--------|------:|
| Valid pixels | 900 / 960 |
| USGS spectra kept | 1092 / 1485 (309 unique names) |
| Encoder pool | 7900 spectra |
| HDBSCAN clusters | 7 (+ 173 noise) |
| DBCV | 0.230 |
| Mean max abundance | 0.672 (physical range OK) |
| Softmax row-sum / positivity | Pass |
| SPM cluster groups | Fe/Mg Phyllosilicates, Carbonates, Water Ice (enrichment-based) |
| Library matches | Mostly Olivine/Acmite at WEAK SAM (~0.32 rad) on this small subset |
| Overall verdict | **MODERATE** (0.1 < DBCV ≤ 0.3) |
| Runtime (GTX 1650) | ~10 min |

Interpret **both** SPM group names and USGS library matches in `cluster_names.json`. On this 15×64 subset, library SAM is WEAK — treat SPM enrichment groups and abundance maps as the primary mineralogy summary, and treat specific specimen names as tentative until a fuller FRT strip is processed.

---

## Agent support for repeated tasks

| Artifact | Path | Purpose |
|----------|------|---------|
| Skill | `.cursor/skills/crism-mineral-identification/SKILL.md` | Repeatable run/interpret workflow |
| Rule | `.cursor/rules/crism-mineral-pipeline.mdc` | Constraints when editing Stage 5 |

Invoke the skill when re-running mineral ID, tuning HDBSCAN/SPM/unmixing, or reading `mineral_output` products.

---

## Notes / limitations

- Current Scene 11/12 `_S4.npy` cubes are small spatial subsets `(15, 64, 438)` — good for pipeline validation; full FRT height will increase CR and HDBSCAN runtime.
- Indices whose centers lie in bad-band windows (e.g. BD1900_2, BD2210_2) correctly return NaN when no valid band is within 30 nm.
- Rare Strategy 1 uses SAM < 0.20 on noise pixels; tighten/loosen via `RARE_ANGLE_THRESH`.
- For larger cubes on 4 GB GPU, keep AMP on and reduce `ENCODER_BATCH` / `BATCH_INF` if OOM occurs.
