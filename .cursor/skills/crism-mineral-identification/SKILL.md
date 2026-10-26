---
name: crism-mineral-identification
description: >-
  Runs and maintains the CRISM mineral identification pipeline in Stage5_minerals.py
  (Stages 0-7: USGS library, SPM, contrastive encoder, HDBSCAN, neural unmixing,
  spatial smoothing, rare minerals, validation). Use when the user asks to identify
  minerals in CRISM cubes, rerun Stage 5, interpret mineral_output products, tune
  SPM/HDBSCAN/unmixing, or extend rare-mineral detection.
---

# CRISM Mineral Identification

## When to use

Apply this skill whenever work involves:
- `Stage5_minerals.py` or `mineral_output/`
- CRISM mineral maps, SPM indices, spectral unmixing, HDBSCAN clusters
- USGS `ASCIIdata_splib07b_cvCRISMjMTR3` library matching
- Commands to identify minerals after denoising (`*_S4.npy`)

## Non-negotiable methodology

Follow `CRISM_MINERAL_IDENTIFICATION_GUIDE.md` as the source of truth:

1. **Linear pipeline only** — Stages 0→7 every run. No branching, no baseline-first shortcuts.
2. **No SRF convolution** — USGS `cvCRISMjMTR3` spectra are already CRISM-convolved.
3. **489 → 438 subset** — map joined library channels to the L-sensor scene wavelengths.
4. **Bad-band aware SPM** — nearest good band within 30 nm; else NaN.
5. **Order of preprocessing** — NaN fill → continuum removal → restore bad/dead NaNs → L2 normalize.
6. **Softmax abundances** — never sigmoid for mixture fractions (non-negative + sum-to-one).
7. **Memory** — continuum removal in row blocks; encoder/unmixing inference in batches (2048 on 4 GB GPUs).

## How to execute

From `cv-crism/`:

```powershell
# Scene 12 (default in Stage5)
$env:PYTHONIOENCODING = "utf-8"
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 12\Crism"
python Stage5_minerals.py

# Scene 11
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 11\Crism"
python Stage5_minerals.py --data-dir $env:CRISM_DATA_DIR

# Via orchestrator (Stage 5 only)
python Run_pipeline.py --stages 5
```

Requires: denoised `ml_output/*_S4.npy`, `eda_output/eda_summary.json` (optional dead bands), CUDA optional but preferred.

## Key outputs (`mineral_output/`)

| File | Meaning |
|------|---------|
| `spm_map.npy` | 12 Viviano-Beck SPM indices |
| `cluster_names.json` | SPM group + USGS library match per cluster |
| `abundance_map.npy` / `smoothed_abundance_map.npy` | Endmember fractions |
| `rare_mineral_detections.json` | Noise-pixel SAM + secondary abundance hits |
| `mineral_identification_report.json` | Validation verdict + metrics |
| `fig1`–`fig6` PNG | Publication figures |

## Agent workflow for repeated runs

1. Confirm `*_S4.npy` exists and is L-sensor (438 bands).
2. Confirm USGS path: `D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata\ASCIIdata_splib07b_cvCRISMjMTR3`.
3. Run `Stage5_minerals.py` with `CRISM_DATA_DIR` set.
4. Read `mineral_identification_report.json` — check `overall_verdict`, DBCV, library SAM, physical validity.
5. For rare minerals, inspect `rare_mineral_detections.json` (Strategy 1 noise + Strategy 2 abundance >10%).
6. If GPU OOM on 4 GB card: lower `ENCODER_BATCH` / `BATCH_INF` in `Stage5_minerals.py`; keep AMP enabled.

## Scientific checks before claiming detections

- Library match SAM < 0.15 rad → STRONG; < 0.30 → MODERATE.
- Abundances ≥ 0 and row sums ≈ 1.0 (softmax).
- SPM centers/shoulders must not silently use bad bands.
- Do not invent mineral labels outside USGS ChapterM / matched library names.

## Related files

- Guide: `CRISM_MINERAL_IDENTIFICATION_GUIDE.md`
- Implementation: `Stage5_minerals.py`
- Orchestrator: `Run_pipeline.py`
- Shared I/O: `crism_utils.py`
