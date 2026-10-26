# CRISM Mineral Identification — Complete Implementation Guide for Cursor Agent

> **Purpose**: This document is the **sole implementation blueprint** for building `Stage5_minerals.py` (and supporting modules) in the `D:\CRISM Journal\cv-crism` project. The Cursor agent must follow this guide **stage by stage, in order**, producing one Python file per stage (or one monolithic file with clearly separated stage functions). No implementation ambiguity should remain after reading this document.

> **Critical Context**: Denoising (Stages 1–4) has already been completed and validated. Two CRISM scenes are available with `PASS` validation status. This guide builds on top of that work.

---

## Table of Contents

1. [Project Context & Current State](#1-project-context--current-state)
2. [CRISM Sensor Technical Specifications](#2-crism-sensor-technical-specifications)
3. [USGS Spectral Library Technical Specifications](#3-usgs-spectral-library-technical-specifications)
4. [CRISM Spectral Parameters (SPM) — Complete Reference](#4-crism-spectral-parameters-spm--complete-reference)
5. [Stage 0 — Setup and Loading](#stage-0--setup-and-loading)
6. [Stage 1 — Spectral Library Preprocessing](#stage-1--spectral-library-preprocessing)
7. [Stage 2 — Continuum Removal and SPM Computation on Full Scene](#stage-2--continuum-removal-and-spm-computation-on-full-scene)
8. [Stage 3 — Contrastive Spectral Encoder Training](#stage-3--contrastive-spectral-encoder-training)
9. [Stage 4 — HDBSCAN Density-Based Clustering](#stage-4--hdbscan-density-based-clustering)
10. [Stage 5 — Physics-Constrained Neural Unmixing](#stage-5--physics-constrained-neural-unmixing)
11. [Stage 6 — CRF Spatial Smoothing](#stage-6--crf-spatial-smoothing)
12. [Stage 7 — Validation and Publication Figures](#stage-7--validation-and-publication-figures)
13. [File & Directory Conventions](#file--directory-conventions)
14. [Dependency List](#dependency-list)
15. [Critical Implementation Warnings](#critical-implementation-warnings)
16. [Rare Mineral Detection Strategy](#rare-mineral-detection-strategy)

---

## 1. Project Context & Current State

### Existing Pipeline Architecture

```
D:\CRISM Journal\cv-crism\
├── crism_utils.py          ← Shared utilities (PDS3 parser, cube loader, wavelengths, bad bands)
├── Stage1_eda.py           ← EDA / basic checking
├── Stage2_noise.py         ← Noise characterisation (N1–N5)
├── Stage3_physics.py       ← Physics-based correction (spikes, thermal, saturation, illumination, atmosphere)
├── Stage4_denoising.py     ← ML denoising (Stripe CNN + Spectral DAE)
├── Run_pipeline.py         ← Orchestrator (runs stages 1–5)
└── Stage5_minerals.py      ← **TO BE IMPLEMENTED** (this guide)
```

### Denoised Scenes Available

| Scene | Cube ID | Sensor | Shape | Wavelength Range | SNR Before → After | SAM | Status |
|-------|---------|--------|-------|------------------|--------------------|----|--------|
| Scene 11 | `frt0000991a_01_if156l_trr3` | L (438 bands) | `(H, 64, 438)` | 1.028–3.920 µm | 28.4 → 35.6 | 0.0261 | PASS |
| Scene 12 | `frt0000991b_01_if156l_trr3` | L (438 bands) | `(H, 64, 438)` | 1.028–3.920 µm | 43.6 → 91.9 | 0.0431 | PASS |

### Denoised Cube Locations

```
D:\NEW CROSS MISSION\Data\Scene 11\Crism\ml_output\frt0000991a_01_if156l_trr3_S4.npy
D:\NEW CROSS MISSION\Data\Scene 12\Crism\ml_output\frt0000991b_01_if156l_trr3_S4.npy
```

### Band Depth Preservation Verified

- **BD2300** (phyllosilicate indicator): preserved within 4–10% delta
- **OLINDEX3** (olivine indicator): preserved within 0.2–1.9% delta

Both cubes are `float32`, shape `(H, W, 438)`, with NaN for invalid pixels and values clipped to `[-0.05, 1.30]` I/F range.

---

## 2. CRISM Sensor Technical Specifications

### L-Sensor (IR)

| Property | Value |
|----------|-------|
| Bands | 438 |
| Wavelength Range | 1.028–3.920 µm |
| Spectral Sampling | ~6.55 nm (non-uniform, varies at filter boundaries) |
| FWHM | ~6.55 nm (varies: narrower at short-λ, wider at long-λ) |
| Spatial Resolution | ~18 m/pixel (FRT mode) |
| Swath Width | 64 pixels (FRT mode) |
| Data Format | PDS3 BIL (Band Interleaved by Line), IEEE float32 |

### Known Bad Band Regions (Already Implemented in `crism_utils.py`)

```python
BAD_BAND_REGIONS_L = [
    (1.000, 1.070),   # detector startup / stray light
    (1.370, 1.420),   # telluric + atmospheric H2O
    (1.810, 1.950),   # H2O + CO2 overlap
    (2.010, 2.060),   # CO2 core
    (2.140, 2.260),   # detector filter-boundary artefact
    (2.600, 2.760),   # deep atmospheric CO2 + H2O
    (3.160, 3.220),   # minor CO2
    (3.550, 4.000),   # thermal emission tail
]
```

### Wavelength Construction

The existing `crism_utils.py` builds L-sensor wavelengths using piecewise-linear anchor interpolation. **The USGS library provides EXACT CRISM joined wavelengths** at:

```
D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata\ASCIIdata_splib07b_cvCRISMjMTR3\
  s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt     ← 489 channels, 0.436–3.897 µm
  s07CRSMj_CRISM_Bandpass_JOINED_MTR3_microns_489ch.txt   ← FWHM per channel
```

The library uses **489 channels** (S+L joined), but the scene cubes have **438 bands** (L-sensor only, 1.028–3.920 µm). This mismatch MUST be resolved by extracting only the L-sensor channels (channels that fall within the cube's wavelength range) from the library.

> **CRITICAL**: The USGS library spectra in `ASCIIdata_splib07b_cvCRISMjMTR3` are **already convolved to the CRISM instrument response function**. They do NOT need SRF convolution. They only need channel subsetting to match the 438-band L-sensor.

---

## 3. USGS Spectral Library Technical Specifications

### Library Location

```
D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata\ASCIIdata_splib07b_cvCRISMjMTR3\
```

### Relevant Chapters

| Chapter | Contents | Mineral Identification Use |
|---------|----------|--------------------------|
| `ChapterM_Minerals` | **1,276 spectra**, 212 unique mineral species | **PRIMARY** — all endmember matching |
| `ChapterS_SoilsAndMixtures` | Soil and mixture spectra | Mixture reference, secondary matching |
| `ChapterC_Coatings` | Coated mineral spectra | Surface alteration detection |
| `ChapterL_Liquids` | Liquid spectra | Water/brine detection |

### File Format

Each spectrum file is a plain-text ASCII file:

```
s07CRSMj Record=31: Acmite NMNH133746 Pyroxene    BECKa AREF    ← Header line
 4.2970572e-002                                                   ← reflectance value, channel 1
 4.3105531e-002                                                   ← reflectance value, channel 2
 ...                                                              ← 489 data lines total
```

- **Line 1**: Header — contains mineral name, sample ID, spectrometer, measurement type
- **Lines 2–490**: One reflectance value per line, one per CRISM joined channel (489 total)
- **Value `-1.23e+034`**: Sentinel for "no data" / deleted channel — **treat as NaN**
- **Total data lines per file**: 489 (matching the 489-channel joined wavelength table)
- **Wavelength units**: Already in µm (from the wavelength file)

### Filename Convention

```
s07CRSMj_{MineralName}_{SampleID}_{Spectrometer}_{MeasType}.txt
```

Examples:
- `s07CRSMj_Olivine_KI3005_Fo89_BECKb_AREF.txt` → Olivine, Fo89 composition
- `s07CRSMj_Nontronite_SWa-1.1_ASDFRb_AREF.txt` → Nontronite clay
- `s07CRSMj_Hematite_FE2602_BECKb_AREF.txt` → Hematite iron oxide

### Mineral Name Extraction from Filename

```python
# Parse: s07CRSMj_{MineralName}_{rest}.txt
name = filename.replace('s07CRSMj_', '').split('_')[0]
# e.g. "Olivine", "Nontronite", "Hematite"
```

### Complete List of 212 Unique Mineral Species in ChapterM

This is the FULL list of minerals available for identification. The pipeline should be able to identify ANY of these:

**Pyroxenes**: Acmite, Augite, Bronzite, Diopside, Enstatite, Fassaite, Hedenbergite, Hypersthene, Jadeite, Pigeonite, Pyroxene, Pyroxmangite
**Feldspars**: Adularia, Albite, Andesine, Anorthite, Bytownite, Labradorite, Microcline, Oligoclase, Orthoclase, Perthite, Sanidine
**Olivines**: Forsterite, Monticellite, Olivine, Knebelite
**Clays**: Beidellite, Dickite, Endellite, Halloysite, Hectorite, Illite, Kaolinite, Montmorillonite, Nacrite, Nontronite, Palygorskite, Rectorite, Saponite, Sauconite, Sepiolite, Vermiculite
**Micas**: Annite, Biotite, Glauconite, Lepidolite, Margarite, Muscovite, Paragonite, Phlogopite, Roscoelite, Siderophyllite
**Sulfates**: Alunite, Anhydrite, Barite, Bassanite, Bloedite, Copiapite, Coquimbite, Epsomite, Eugsterite, Gypsum, Jarosite, Kainite, Kieserite, Polyhalite, Schwertmannite, Syngenite, Thenardite
**Carbonates**: Aragonite, Calcite, Dolomite, Gaylussite, Magnesite (check under the full set), Rhodochrosite, Siderite, Strontianite, Witherite
**Iron Oxides/Hydroxides**: Cronstedtite, Ferrihydrite, Goethite, Hematite, Ilmenite, Lepidocrocite, Limonite, Maghemite, Magnetite, Nanohematite
**Serpentines**: Antigorite, Chrysotile, Lizardite, Serpentine
**Chlorites**: Chamosite (under Chlorite), Clinochlore, Cookeite, Prochlorite, Thuringite
**Amphiboles**: Actinolite, Anthophyllite, Cummingtonite, Glaucophane, Hornblende, Nephrite, Richterite, Riebeckite, Smaragdite, Tremolite, Uralite
**Zeolites**: Analcime, Clinoptilolite, Heulandite, Laumontite, Mordenite, Natrolite, Scolecite, Stilbite
**Garnets**: Almandine, Andradite, Grossular, Hydrogrossular, Spessartine, Uvarovite
**Phosphates**: Chlorapatite, Fluorapatite, Hydroxyl-Apatite, Monazite, Pyromorphite, Xenotime
**Silica**: Chalcedony, Chert, Opal, Quartz
**Others**: Allanite, Andalusite, Axinite, Azurite, Beryl, Brucite, Buddingtonite, Butlerite, Carbon, Carnallite, Cassiterite, Celestite, Chalcopyrite, Chromite, Chrysocolla, Cinnabar, Clintonite, Colemanite, Corundum, Cuprite, Datolite, Diaspore, Dipyre, Dumortierite, Elbaite, Epidote, Galena, Gibbsite, Halite, Howlite, Karpatite, Lazurite, Malachite, Marialite, Mascagnite, Meionite, Mirabilite, Mizzonite, Nepheline, Niter, Parisite, Pectolite, Pinnoite, Pitch, Portlandite, Prehnite, Psilomelane, Pyrite, Pyrolusite, Pyrophyllite, Pyrrhotite, Rivadavite, Rutile, Sphalerite, Sphene, Spodumene, Staurolite, Sulfur, Talc, Tincalconite, Topaz, Tourmaline, Trona, Ulexite, Vesuvianite, Wollastonite, Zircon, Zoisite, Zunyite

---

## 4. CRISM Spectral Parameters (SPM) — Complete Reference

These are the validated spectral parameter indices from Viviano-Beck et al. (2014) adapted for CRISM L-sensor. They use band depth (BD) or band ratio calculations on continuum-removed spectra. Each index targets a specific mineral group.

### Primary SPM Indices for Mars Mineralogy

> **IMPORTANT**: All wavelength positions below are approximate. The implementation MUST find the nearest available band index in the cube's wavelength array using `np.argmin(np.abs(waves - target_wavelength))`. If the nearest band falls within a bad-band region, use the next nearest good band.

#### 1. BD1900_2 — Hydrated Minerals (H₂O bound in minerals)

```
Purpose: Detects bound water (~1.9 µm absorption)
Minerals: Gypsum, polyhydrated sulfates, hydrated silica, phyllosilicates
Formula: BD = 1 - (R(1.93) / (a * R(1.857) + b * R(2.067)))
  where a = (2.067 - 1.93) / (2.067 - 1.857) ≈ 0.652
        b = (1.93 - 1.857) / (2.067 - 1.857) ≈ 0.348
Center: 1.930 µm
Shoulders: 1.857 µm, 2.067 µm
Positive detection threshold: > 0.01
```

#### 2. BD2100_2 — Monohydrated Sulfates

```
Purpose: Detects monohydrated sulfate ~2.1 µm absorption
Minerals: Kieserite, szomolnokite
Formula: BD = 1 - (R(2.132) / (a * R(1.930) + b * R(2.250)))
Center: 2.132 µm
Shoulders: 1.930 µm, 2.250 µm
Positive detection threshold: > 0.01
```

#### 3. BD2210_2 — Al-OH Minerals

```
Purpose: Detects Al-OH absorption at ~2.21 µm
Minerals: Kaolinite, Al-smectite (montmorillonite), Al-muscovite
Formula: BD = 1 - (R(2.210) / (a * R(2.165) + b * R(2.250)))
Center: 2.210 µm
Shoulders: 2.165 µm, 2.250 µm
Positive detection threshold: > 0.005
WARNING: This wavelength range falls partially in a bad-band region (2.140–2.260 µm).
         Implementation MUST check band validity before computing.
```

#### 4. BD2290 — Mg,Fe-OH / CO₃ Minerals

```
Purpose: Detects Mg-OH or Fe-OH or carbonate absorption at ~2.29 µm
Minerals: Fe/Mg-smectites (nontronite, saponite), Mg-carbonates,
          serpentine, chlorite, prehnite, epidote
Formula: BD = 1 - (R(2.290) / (a * R(2.250) + b * R(2.350)))
Center: 2.290 µm
Shoulders: 2.250 µm, 2.350 µm
Positive detection threshold: > 0.005
```

#### 5. BD2300 — Fe/Mg Phyllosilicates (Already validated in denoising)

```
Purpose: Detects 2.3 µm metal-OH absorption
Minerals: Chlorite, prehnite, serpentine, Mg-carbonate, vermiculite
Formula: BD = 1 - (R(2.295) / (a * R(2.120) + b * R(2.530)))
Center: 2.295 µm
Shoulders: 2.120 µm, 2.530 µm
Positive detection threshold: > 0.01
```

#### 6. D2300 — Hydroxylated Fe/Mg Silicates (Drop index)

```
Purpose: Modified 2.3 µm index with wider shoulders for better discrimination
Minerals: Same as BD2300 but more sensitive to vermiculite, Mg-chlorite
Formula: Modified band depth using 3-point continuum
Center: 2.290 µm
Left shoulder: 2.120 µm
Right shoulder: 2.530 µm
```

#### 7. OLINDEX3 — Olivine (Already validated in denoising)

```
Purpose: Broad 1.0–1.8 µm olivine absorption complex
Minerals: Forsterite, fayalite, olivine solid solutions
Formula: Multi-point — uses reflectance at 6 wavelengths:
  RB1054 = R(1.054)
  RB1211 = R(1.211)
  RB1329 = R(1.329)
  RB1474 = R(1.474)
  RB1750 = R(1.750)
  OLINDEX3 = (RB1054 * 0.1) + (RB1211 * 0.1) + (RB1329 * 0.4) + (RB1474 * 0.4) - RB1750
  Normalize by RB1750
  
Simplified practical version:
  OLINDEX3 = BD at center ~1.33 µm with shoulders ~1.05 and ~1.75 µm
  
Positive detection threshold: > 0.02
```

#### 8. LCPINDEX2 — Low-Calcium Pyroxene (Orthopyroxene)

```
Purpose: Detects ~1.81 µm pyroxene absorption
Minerals: Enstatite, bronzite, hypersthene (orthopyroxenes)
Formula: BD-style using:
  Center: ~1.810 µm
  Left shoulder: ~1.560 µm
  Right shoulder: ~2.450 µm
Positive detection threshold: > 0.01
```

#### 9. HCPINDEX2 — High-Calcium Pyroxene (Clinopyroxene)

```
Purpose: Detects ~2.12 µm pyroxene absorption
Minerals: Augite, diopside, pigeonite (clinopyroxenes)
Formula: BD-style using:
  Center: ~2.120 µm
  Left shoulder: ~1.810 µm
  Right shoulder: ~2.530 µm
Positive detection threshold: > 0.01
```

#### 10. BD1500_2 — Water Ice

```
Purpose: Detects water ice absorption at 1.5 µm
Minerals: Water ice (not liquid water)
Formula: BD = 1 - (R(1.510) / (a * R(1.367) + b * R(1.810)))
Center: 1.510 µm
Shoulders: 1.367 µm, 1.810 µm
Positive detection threshold: > 0.01
```

#### 11. BD1435 — CO₂ Ice

```
Purpose: Detects CO₂ ice absorption at 1.435 µm
Minerals: CO₂ ice
Formula: BD = 1 - (R(1.435) / (a * R(1.370) + b * R(1.470)))
Center: 1.435 µm
Shoulders: 1.370 µm, 1.470 µm
Positive detection threshold: > 0.01
NOTE: Shoulder at 1.370 µm is in a bad-band region. Use nearest valid band.
```

#### 12. SINDEX2 — Sulfate Index

```
Purpose: Broad sulfate absorption at ~2.1 µm
Minerals: Gypsum, bassanite, anhydrite, polyhydrated sulfates
Formula: BD-style using:
  Center: 2.100 µm
  Left shoulder: 1.815 µm
  Right shoulder: 2.200 µm
Positive detection threshold: > 0.005
```

#### 13. BD2500_2 — Carbonate

```
Purpose: Carbonate absorption at ~2.5 µm
Minerals: Calcite, dolomite, magnesite, siderite
Formula: BD = 1 - (R(2.500) / (a * R(2.380) + b * R(2.570)))
Center: 2.500 µm
Shoulders: 2.380 µm, 2.570 µm
Positive detection threshold: > 0.005
NOTE: Near the edge of a bad-band region (2.600–2.760 µm)
```

#### 14. MIN2295_2480 — Fe/Mg Smectites

```
Purpose: Ratio-based index for Fe/Mg smectites
Minerals: Nontronite, saponite, Fe-chlorite
Formula: min(BD(2.295), BD(2.480))
Uses the minimum of two band depths to require BOTH absorptions
```

#### 15. MIN2250 — Opaline Silica and Al-clay Doublet

```
Purpose: Hydrated silica absorption at ~2.25 µm
Minerals: Opal, Al-bearing phyllosilicates
Formula: BD = 1 - (R(2.245) / (a * R(2.165) + b * R(2.350)))
```

#### 16. RPEAK1 — Reflectance Peak Position (Iron indicator)

```
Purpose: Position of maximum reflectance in 0.6–0.84 µm range
Minerals: Iron-bearing minerals shift the peak position
NOTE: These wavelengths are in the S-sensor range (0.362–1.053 µm).
      For L-sensor-only data, this index is NOT available.
      The L-sensor starts at 1.028 µm. Skip this index for L-sensor cubes.
```

### SPM Summary for L-Sensor Implementation

The following indices are **computable** on the 438-band L-sensor (1.028–3.920 µm):

| Index | Center (µm) | Shoulder L (µm) | Shoulder R (µm) | Target Mineral Group | Bad-Band Conflict? |
|-------|-------------|-----------------|-----------------|---------------------|-------------------|
| BD1435 | 1.435 | 1.370 | 1.470 | CO₂ ice | Left shoulder marginal |
| BD1500_2 | 1.510 | 1.367 | 1.810 | Water ice | Left shoulder marginal |
| BD1900_2 | 1.930 | 1.857 | 2.067 | Hydrated minerals | Center in bad region |
| BD2100_2 | 2.132 | 1.930 | 2.250 | Monohydrated sulfates | Right shoulder in bad |
| BD2210_2 | 2.210 | 2.165 | 2.250 | Al-OH clays | In bad region |
| BD2290 | 2.290 | 2.250 | 2.350 | Fe/Mg-OH, CO₃ | Marginal |
| BD2300 | 2.295 | 2.120 | 2.530 | Fe/Mg phyllosilicates | Clean |
| BD2500_2 | 2.500 | 2.380 | 2.570 | Carbonates | Near bad region edge |
| OLINDEX3 | ~1.33 | ~1.05 | ~1.75 | Olivine | Clean |
| LCPINDEX2 | ~1.81 | ~1.56 | ~2.45 | Low-Ca pyroxene | Center near bad |
| HCPINDEX2 | ~2.12 | ~1.81 | ~2.53 | High-Ca pyroxene | Clean |
| SINDEX2 | 2.100 | 1.815 | 2.200 | Sulfates | Marginal |

**Indices NOT available for L-sensor-only data**: RPEAK1 (requires S-sensor, 0.6–0.84 µm)

### Band Depth Computation Formula (General)

```python
def band_depth(spectrum, waves, center_um, left_um, right_um, bad_mask):
    """
    Compute band depth for a single spectrum.
    Returns NaN if any required band is invalid.
    """
    # Find nearest good band indices
    def nearest_good(target_wl):
        dists = np.abs(waves - target_wl)
        dists[bad_mask] = np.inf  # exclude bad bands
        bi = np.argmin(dists)
        if dists[bi] > 0.030:  # tolerance: 30 nm
            return None
        return bi

    ci = nearest_good(center_um)
    li = nearest_good(left_um)
    ri = nearest_good(right_um)

    if any(x is None for x in (ci, li, ri)):
        return np.nan

    rc = spectrum[ci]
    rl = spectrum[li]
    rr = spectrum[ri]

    if any(not np.isfinite(x) for x in (rc, rl, rr)):
        return np.nan

    # Linear continuum at center position
    frac = (waves[ci] - waves[li]) / (waves[ri] - waves[li] + 1e-9)
    cont = rl + frac * (rr - rl)

    if cont < 1e-6:
        return np.nan

    return float(1.0 - rc / cont)
```

---

## Stage 0 — Setup and Loading

### What to Do

1. **Accept `DATA_DIR` as environment variable** (same pattern as existing stages):
   ```python
   DATA_DIR = os.environ.get("CRISM_DATA_DIR", r"D:\NEW CROSS MISSION\Data\Scene 12\Crism")
   ```

2. **Load the denoised cube** from Stage 4 output:
   ```python
   # Find the _S4.npy file in ml_output/
   ml_dir = os.path.join(DATA_DIR, "ml_output")
   s4_files = glob.glob(os.path.join(ml_dir, '**', '*_S4.npy'), recursive=True)
   cube_denoised = np.load(s4_file).astype(np.float32)  # (H, W, 438)
   ```

3. **Load wavelengths** using existing `crism_utils.get_sensor_waves()`:
   ```python
   from crism_utils import get_sensor_waves, build_bad_mask, parse_pds3_label, find_label
   # Find original .img to get label metadata
   waves = get_sensor_waves(B, meta.get('wavelengths'))  # (438,) float32
   ```

4. **Load bad-band mask** using existing `crism_utils.build_bad_mask()`:
   ```python
   bad_mask = build_bad_mask(waves)  # (438,) bool, True = bad
   good_mask = ~bad_mask
   ```

5. **Load EDA summary** to get dead bands:
   ```python
   eda_json = os.path.join(DATA_DIR, "eda_output", "eda_summary.json")
   with open(eda_json) as f:
       eda = json.load(f)
   dead_bands = eda[sname].get('dead_bands', [])
   ```

6. **Build valid pixel mask**:
   ```python
   # A pixel is valid if it has finite values in at least 50% of good bands
   good_bi = np.where(good_mask)[0]
   finite_count = np.sum(np.isfinite(cube_denoised[:, :, good_bi]), axis=2)
   valid_pixel_mask = finite_count >= (len(good_bi) * 0.5)  # (H, W) bool
   ```

7. **Create output directory**:
   ```python
   OUTPUT_DIR = os.path.join(DATA_DIR, "mineral_output")
   os.makedirs(OUTPUT_DIR, exist_ok=True)
   ```

8. **Print diagnostics**:
   ```
   Cube shape: (H, W, 438)
   Wavelength range: 1.028–3.920 µm
   Good bands: X / 438
   Dead bands: Y
   Valid pixels: Z / (H*W)
   ```

### Output of Stage 0

- `cube_denoised` — `(H, W, 438)` float32
- `waves` — `(438,)` float32
- `bad_mask` — `(438,)` bool
- `good_mask` — `(438,)` bool
- `valid_pixel_mask` — `(H, W)` bool
- `dead_bands` — list of int

---

## Stage 1 — Spectral Library Preprocessing

### Why This Stage Runs First

The library is used in **two** subsequent stages:
- Stage 3: as additional training examples for the contrastive encoder
- Stage 5: as endmember initializers for the unmixer

Both uses require the library to be in **exactly the same spectral domain** as the CRISM satellite data.

### Step 1.1 — Load USGS CRISM-Convolved Library

**Source directory**: `D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata\ASCIIdata_splib07b_cvCRISMjMTR3`

**Load the wavelength table first**:
```python
USGS_DIR = r"D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata\ASCIIdata_splib07b_cvCRISMjMTR3"
waves_file = os.path.join(USGS_DIR, "s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt")
bandpass_file = os.path.join(USGS_DIR, "s07CRSMj_CRISM_Bandpass_JOINED_MTR3_microns_489ch.txt")

# Parse: skip header line, read 489 float values
lib_waves_489 = parse_usgs_ascii_values(waves_file)  # (489,) in µm, range 0.436–3.897 µm
lib_bandpass_489 = parse_usgs_ascii_values(bandpass_file)  # (489,) FWHM in µm
```

**Load spectra from `ChapterM_Minerals`** (and optionally `ChapterS_SoilsAndMixtures`):
```python
mineral_dir = os.path.join(USGS_DIR, "ChapterM_Minerals")
soil_dir = os.path.join(USGS_DIR, "ChapterS_SoilsAndMixtures")

for txt_file in glob.glob(os.path.join(mineral_dir, "*.txt")):
    header, values = parse_usgs_spectrum(txt_file)
    # header: "s07CRSMj Record=31: Acmite NMNH133746 Pyroxene    BECKa AREF"
    # values: (489,) float array
    mineral_name = extract_mineral_name(txt_file)
    # Store (mineral_name, sample_id, values_489)
```

**Parse function**:
```python
def parse_usgs_spectrum(filepath):
    """Parse USGS ASCII spectrum file. Returns (header_string, values_array_489)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    values = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            val = float(line)
            # Sentinel values: -1.23e+034 means no data
            if val < -1e+030:
                values.append(np.nan)
            else:
                values.append(val)
    return header, np.array(values, dtype=np.float32)
```

### Step 1.2 — Subset to L-Sensor Channels

The library has 489 channels (S+L joined). The scene cube has 438 bands (L-sensor only).

```python
# Find which of the 489 library channels correspond to the scene's wavelength range
scene_wl_min = waves[0]   # ~1.028 µm
scene_wl_max = waves[-1]  # ~3.920 µm

# Method: For each of the 438 scene wavelengths, find the nearest library channel
lib_to_scene_map = []  # indices into the 489-channel library for each scene band
for si, sw in enumerate(waves):
    closest_lib_idx = np.argmin(np.abs(lib_waves_489 - sw))
    if np.abs(lib_waves_489[closest_lib_idx] - sw) < 0.010:  # 10 nm tolerance
        lib_to_scene_map.append(closest_lib_idx)
    else:
        lib_to_scene_map.append(None)  # no match — will be NaN

# Extract the 438-band subset from every library spectrum
lib_spectra_438 = []
for spectrum_489 in all_library_spectra_489:
    spec_438 = np.full(438, np.nan, dtype=np.float32)
    for si, li in enumerate(lib_to_scene_map):
        if li is not None:
            spec_438[si] = spectrum_489[li]
    lib_spectra_438.append(spec_438)
```

### Step 1.3 — Quality Filtering

Apply three criteria to each library spectrum:

```python
def is_quality_spectrum(spectrum_438, bad_mask_438):
    """Returns True if spectrum passes quality filters."""
    good_bi = np.where(~bad_mask_438)[0]
    good_vals = spectrum_438[good_bi]

    # Criterion 1: NaN fraction in good bands must be < 30%
    nan_frac = np.sum(np.isnan(good_vals)) / len(good_vals)
    if nan_frac > 0.30:
        return False

    # Criterion 2: No adjacent-band discontinuity > 0.5 reflectance units
    finite_vals = good_vals[np.isfinite(good_vals)]
    if len(finite_vals) > 1:
        diffs = np.abs(np.diff(finite_vals))
        if np.max(diffs) > 0.5:
            return False

    # Criterion 3: Values must be in physically plausible range [0, 1.5]
    finite_all = spectrum_438[np.isfinite(spectrum_438)]
    if len(finite_all) > 0:
        if np.min(finite_all) < -0.05 or np.max(finite_all) > 1.5:
            return False

    return True
```

### Step 1.4 — Apply Bad-Band Mask to Library Spectra

```python
for spec in lib_spectra_438:
    spec[bad_mask] = np.nan
    for d in dead_bands:
        if d < 438:
            spec[d] = np.nan
```

### Step 1.5 — NaN Interpolation for Library Spectra

Before continuum removal, fill NaN gaps in library spectra using linear interpolation:

```python
from scipy.interpolate import interp1d

def fill_nans_interp(spectrum, waves):
    """Fill NaN values using linear interpolation from valid neighbors."""
    finite = np.isfinite(spectrum)
    if finite.sum() < 10:
        return None  # too few valid points
    if finite.all():
        return spectrum
    f = interp1d(waves[finite], spectrum[finite],
                 kind='linear', fill_value='extrapolate', bounds_error=False)
    filled = spectrum.copy()
    filled[~finite] = f(waves[~finite])
    return filled
```

### Step 1.6 — Convex Hull Continuum Removal on Library Spectra

```python
from scipy.spatial import ConvexHull

def continuum_removal(spectrum, waves):
    """
    Upper convex hull continuum removal.
    Returns continuum-removed spectrum where absorption features dip below 1.0.
    """
    finite = np.isfinite(spectrum) & (spectrum > 0)
    if finite.sum() < 5:
        return np.full_like(spectrum, np.nan)

    # Work only with finite values
    w_valid = waves[finite]
    s_valid = spectrum[finite]

    # Build 2D points (wavelength, reflectance)
    points = np.column_stack([w_valid, s_valid])

    # Find upper convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: simple linear continuum between first and last point
        cont = np.interp(waves, [w_valid[0], w_valid[-1]], [s_valid[0], s_valid[-1]])
        result = spectrum / np.maximum(cont, 1e-8)
        return np.clip(result, 0, 2.0)

    # Extract upper hull vertices (those where y is above the centroid)
    vertices = hull.vertices
    upper = [v for v in sorted(vertices) if points[v, 1] >= np.median(s_valid)]

    if len(upper) < 2:
        upper = sorted(vertices)

    hull_x = points[upper, 0]
    hull_y = points[upper, 1]

    # Sort by wavelength
    order = np.argsort(hull_x)
    hull_x = hull_x[order]
    hull_y = hull_y[order]

    # Interpolate continuum at all wavelengths
    continuum = np.interp(waves, hull_x, hull_y, left=hull_y[0], right=hull_y[-1])

    # Divide spectrum by continuum
    result = np.full_like(spectrum, np.nan)
    valid = (continuum > 1e-8) & np.isfinite(spectrum)
    result[valid] = spectrum[valid] / continuum[valid]

    return result
```

> **CRITICAL WARNING**: The upper convex hull implementation above is a simplified version. For production quality, implement the **modified upper convex hull** algorithm:
> 1. Start from the leftmost point
> 2. At each step, find the point that creates the steepest upward slope from the current position
> 3. Continue until the rightmost point is reached
> 4. This traces the true upper envelope

### Step 1.7 — L2 Normalization on Library Spectra

```python
def l2_normalize(spectrum):
    """L2-normalize a spectrum. Returns unit vector."""
    finite = np.isfinite(spectrum)
    if not finite.any():
        return spectrum
    # Replace NaN with 0 for norm computation
    s = np.where(finite, spectrum, 0.0)
    norm = np.linalg.norm(s)
    if norm < 1e-8:
        return s
    return s / norm
```

### Step 1.8 — Compute SPM Values for Library Spectra

After continuum removal, compute all available SPM indices for each library spectrum:

```python
spm_indices_to_compute = {
    'BD1900_2': (1.930, 1.857, 2.067),
    'BD2100_2': (2.132, 1.930, 2.250),
    'BD2210_2': (2.210, 2.165, 2.250),
    'BD2290':   (2.290, 2.250, 2.350),
    'BD2300':   (2.295, 2.120, 2.530),
    'BD2500_2': (2.500, 2.380, 2.570),
    'OLINDEX3': (1.330, 1.054, 1.750),  # simplified 3-point
    'LCPINDEX2':(1.810, 1.560, 2.450),
    'HCPINDEX2':(2.120, 1.810, 2.530),
    'SINDEX2':  (2.100, 1.815, 2.200),
    'BD1500_2': (1.510, 1.367, 1.810),
    'BD1435':   (1.435, 1.370, 1.470),
}

# For each library spectrum (continuum-removed, NOT L2-normalized):
lib_spm = np.zeros((N_lib, len(spm_indices_to_compute)), dtype=np.float32)
for i, spec_cr in enumerate(lib_spectra_cr):
    for j, (name, (c, l, r)) in enumerate(spm_indices_to_compute.items()):
        lib_spm[i, j] = band_depth(spec_cr, waves, c, l, r, bad_mask)
```

### Step 1.9 — Save Preprocessed Library

```python
np.save(os.path.join(OUTPUT_DIR, 'lib_spectra_438_cr_l2.npy'), lib_spectra_cr_l2)  # (N, 438)
np.save(os.path.join(OUTPUT_DIR, 'lib_spectra_438_cr.npy'), lib_spectra_cr)  # (N, 438) - CR but not L2
np.save(os.path.join(OUTPUT_DIR, 'lib_spm.npy'), lib_spm)  # (N, 12)
with open(os.path.join(OUTPUT_DIR, 'lib_names.json'), 'w') as f:
    json.dump(lib_names, f)  # list of mineral names
```

### Expected Output

- `lib_spectra_438_cr_l2.npy` — (N_valid, 438) float32, continuum-removed & L2-normalized
- `lib_spectra_438_cr.npy` — (N_valid, 438) float32, continuum-removed only
- `lib_spm.npy` — (N_valid, 12) float32, SPM values
- `lib_names.json` — list of N_valid mineral names

Expected N_valid: 800–1100 out of 1276 (after quality filtering)

---

## Stage 2 — Continuum Removal and SPM Computation on Full Scene

### Why This Stage Exists

Every pixel must go through the **same** preprocessing as the library (continuum removal, L2 normalization) before entering the encoder. SPM values are also needed for physics-anchored training.

### Step 2.1 — Continuum Removal on Full Cube

```python
# Process in spatial blocks to manage memory
cube_cr = np.full_like(cube_denoised, np.nan)
BLOCK_SIZE = 64  # rows at a time

for r_start in range(0, H, BLOCK_SIZE):
    r_end = min(r_start + BLOCK_SIZE, H)
    for r in range(r_start, r_end):
        for c in range(W):
            if not valid_pixel_mask[r, c]:
                continue
            spec = cube_denoised[r, c, :].copy()
            # Fill NaN with band mean for continuum computation
            spec_filled = fill_nans_interp(spec, waves)
            if spec_filled is None:
                continue
            cube_cr[r, c, :] = continuum_removal(spec_filled, waves)
            # Restore NaN at bad band positions
            cube_cr[r, c, bad_mask] = np.nan
            for d in dead_bands:
                if d < 438:
                    cube_cr[r, c, d] = np.nan
    if (r_start // BLOCK_SIZE) % 5 == 0:
        print(f"  CR progress: {r_end}/{H} rows ({100*r_end/H:.1f}%)")

np.save(os.path.join(OUTPUT_DIR, 'cube_cr.npy'), cube_cr.astype(np.float32))
```

### Step 2.2 — L2 Normalization on All Pixel Spectra

```python
cube_cr_l2 = np.full_like(cube_cr, np.nan)
for r in range(H):
    for c in range(W):
        if valid_pixel_mask[r, c]:
            cube_cr_l2[r, c, :] = l2_normalize(cube_cr[r, c, :])

np.save(os.path.join(OUTPUT_DIR, 'cube_cr_l2.npy'), cube_cr_l2.astype(np.float32))
```

> **Optimization**: Vectorize these loops. Use `np.linalg.norm` on reshaped arrays:
> ```python
> flat = cube_cr.reshape(-1, 438)
> norms = np.linalg.norm(np.nan_to_num(flat), axis=1, keepdims=True)
> norms[norms < 1e-8] = 1.0
> flat_l2 = np.nan_to_num(flat) / norms
> cube_cr_l2 = flat_l2.reshape(H, W, 438)
> ```

### Step 2.3 — SPM Index Computation for All Valid Pixels

```python
N_spm = len(spm_indices_to_compute)
spm_map = np.full((H, W, N_spm), np.nan, dtype=np.float32)

for r in range(H):
    for c in range(W):
        if not valid_pixel_mask[r, c]:
            continue
        spec_cr = cube_cr[r, c, :]
        for j, (name, (cen, left, right)) in enumerate(spm_indices_to_compute.items()):
            spm_map[r, c, j] = band_depth(spec_cr, waves, cen, left, right, bad_mask)

np.save(os.path.join(OUTPUT_DIR, 'spm_map.npy'), spm_map)
```

> **Optimization**: Vectorize with pre-computed band indices.

### Step 2.4 — SPM-Stratified Sampling for Training

```python
TARGET_TRAINING_SAMPLES = 12000

valid_coords = np.argwhere(valid_pixel_mask)  # (N_valid, 2)
N_valid = len(valid_coords)
spm_flat = spm_map[valid_pixel_mask]  # (N_valid, N_spm)

selected_indices = set()

# For each SPM index, select top 20% pixels
for j in range(N_spm):
    col = spm_flat[:, j]
    finite = np.isfinite(col)
    if finite.sum() == 0:
        continue
    threshold = np.percentile(col[finite], 80)
    top_indices = np.where(finite & (col >= threshold))[0]
    # Sample at least 500 from each stratum
    n_sample = min(500, len(top_indices))
    chosen = np.random.choice(top_indices, n_sample, replace=False)
    selected_indices.update(chosen.tolist())

# Fill remaining budget with background pixels
remaining = TARGET_TRAINING_SAMPLES - len(selected_indices)
if remaining > 0:
    background = set(range(N_valid)) - selected_indices
    if len(background) > remaining:
        bg_sample = np.random.choice(list(background), remaining, replace=False)
    else:
        bg_sample = list(background)
    selected_indices.update(bg_sample)

selected_indices = sorted(selected_indices)
training_coords = valid_coords[selected_indices]  # (N_train, 2) — (row, col) pairs
```

### Output of Stage 2

- `cube_cr.npy` — (H, W, 438) continuum-removed
- `cube_cr_l2.npy` — (H, W, 438) CR + L2-normalized
- `spm_map.npy` — (H, W, 12) SPM values
- `training_coords` — (N_train, 2) training pixel coordinates

---

## Stage 3 — Contrastive Spectral Encoder Training

### Why Self-Supervised Contrastive Learning

No mineral labels exist for the scene pixels. The encoder learns through:
1. **Contrastive principle**: Two augmented views of the same spectrum → similar representations
2. **Physics anchor**: Latent representation must predict SPM mineral indices

### Step 3.1 — Build the Training Pool

Combine three sources. **No labels** are attached.

```python
# Source A — Real scene spectra (from stratified sample)
scene_spectra = cube_cr_l2[training_coords[:, 0], training_coords[:, 1], :]  # (N_train, 438)
scene_spm = spm_map[training_coords[:, 0], training_coords[:, 1], :]  # (N_train, 12)

# Source B — Library spectra (sample 2000 with replacement if needed)
n_lib_sample = min(2000, len(lib_spectra_cr_l2) * 3)
lib_indices = np.random.choice(len(lib_spectra_cr_l2), n_lib_sample, replace=True)
lib_sample = lib_spectra_cr_l2[lib_indices]  # (2000, 438)
lib_spm_sample = lib_spm[lib_indices]  # (2000, 12)

# Source C — Synthetic mixed spectra (5000)
n_synthetic = 5000
synth_spectra = []
synth_spm = []
for _ in range(n_synthetic):
    i1, i2 = np.random.choice(len(lib_spectra_cr_l2), 2, replace=False)
    alpha = np.random.uniform(0.1, 0.9)
    mixed = alpha * lib_spectra_cr_l2[i1] + (1 - alpha) * lib_spectra_cr_l2[i2]
    mixed = l2_normalize(mixed)
    synth_spectra.append(mixed)
    # SPM of mixture: approximate as linear combination
    mixed_spm = alpha * lib_spm[i1] + (1 - alpha) * lib_spm[i2]
    synth_spm.append(mixed_spm)

# Combine all sources
pool_spectra = np.concatenate([scene_spectra, lib_sample, np.array(synth_spectra)])
pool_spm = np.concatenate([scene_spm, lib_spm_sample, np.array(synth_spm)])
# Replace NaN with 0 in pool
pool_spectra = np.nan_to_num(pool_spectra, nan=0.0)
pool_spm = np.nan_to_num(pool_spm, nan=0.0)
```

### Step 3.2 — Encoder Architecture

```python
class SpectralEncoder(nn.Module):
    """
    1D CNN encoder for CRISM spectral data.
    Input: (batch, 1, 438) — one channel, 438 spectral bands
    Output: (batch, 64) — 64-dim latent vector
    """
    def __init__(self, n_bands=438, latent_dim=64, n_spm=12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pool → (batch, 128, 1)
        self.fc = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),  # no activation
        )
        # SPM prediction head (only used during training)
        self.spm_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_spm),
            nn.Sigmoid(),  # SPM values are bounded [0, 1] approximately
        )

    def forward(self, x):
        """x: (batch, 1, n_bands) → latent: (batch, 64)"""
        h = self.conv(x)          # (batch, 128, n_bands)
        h = self.gap(h).squeeze(2)  # (batch, 128)
        latent = self.fc(h)       # (batch, 64)
        return latent

    def predict_spm(self, latent):
        """latent: (batch, 64) → spm: (batch, n_spm)"""
        return self.spm_head(latent)
```

### Step 3.3 — Mineralogically-Informed Augmentation

```python
def augment_spectrum(spectrum_1d, spm_diagnostic_bands):
    """
    Apply three augmentations to a single spectrum.
    spectrum_1d: (438,) numpy array
    spm_diagnostic_bands: set of band indices that are protected from dropout
    Returns augmented spectrum.
    """
    aug = spectrum_1d.copy()

    # Aug 1: Continuum scaling [0.85, 1.15]
    scale = np.random.uniform(0.85, 1.15)
    aug *= scale

    # Aug 2: Gaussian spectral smoothing (50% probability)
    if np.random.random() < 0.5:
        sigma = np.random.uniform(0.3, 0.8)
        from scipy.ndimage import gaussian_filter1d
        aug = gaussian_filter1d(aug, sigma)

    # Aug 3: Protected band dropout (5% of unprotected bands → 0)
    all_bands = set(range(len(aug)))
    unprotected = list(all_bands - spm_diagnostic_bands)
    n_drop = max(1, int(0.05 * len(unprotected)))
    drop_bi = np.random.choice(unprotected, n_drop, replace=False)
    aug[drop_bi] = 0.0

    return aug

# Pre-compute protected bands: all bands within ±3 of any SPM diagnostic wavelength
spm_wavelengths = [1.930, 1.857, 2.067, 2.132, 2.250, 2.210, 2.165, 2.290,
                   2.350, 2.295, 2.120, 2.530, 2.500, 2.380, 2.570,
                   1.330, 1.054, 1.750, 1.810, 1.560, 2.450, 2.120,
                   1.510, 1.367, 1.435, 1.370, 1.470, 2.100, 1.815, 2.200]
protected_bands = set()
for wl in spm_wavelengths:
    bi = np.argmin(np.abs(waves - wl))
    for offset in range(-3, 4):
        if 0 <= bi + offset < 438:
            protected_bands.add(bi + offset)
```

### Step 3.4 — NT-Xent Contrastive Loss

```python
def nt_xent_loss(z1, z2, temperature=0.07):
    """
    NT-Xent contrastive loss.
    z1, z2: (batch, latent_dim) — L2-normalized latent vectors of augmented pairs
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)

    return F.cross_entropy(sim, labels)
```

### Step 3.5–3.7 — Total Loss

```python
# In training loop:
spec_batch = pool_spectra[batch_idx]  # (B, 438)
spm_batch = pool_spm[batch_idx]  # (B, 12)

# Create two augmented views
view1 = torch.stack([augment_spectrum(s, protected_bands) for s in spec_batch])
view2 = torch.stack([augment_spectrum(s, protected_bands) for s in spec_batch])

# Forward
z1 = encoder(view1.unsqueeze(1))  # (B, 64)
z2 = encoder(view2.unsqueeze(1))  # (B, 64)

# Losses
loss_contrastive = nt_xent_loss(z1, z2, temperature=0.07)

spm_pred1 = encoder.predict_spm(z1)
spm_target = torch.tensor(spm_batch).to(device)
loss_spm = F.mse_loss(spm_pred1, spm_target)

loss_total = loss_contrastive + 0.1 * loss_spm
```

### Step 3.8 — Training Configuration

```python
optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
EPOCHS = 100
BATCH_SIZE = 256
# Use mixed precision if CUDA available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
# Gradient clipping: max_norm=1.0
torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
# Save model
torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder_B438.pth'))
```

### Step 3.9 — Full Scene Encoding (Inference)

```python
encoder.eval()
valid_coords_all = np.argwhere(valid_pixel_mask)  # (N_valid, 2)
N_valid = len(valid_coords_all)

latent_vectors = np.zeros((N_valid, 64), dtype=np.float32)
BATCH_INF = 4096

with torch.no_grad():
    for start in range(0, N_valid, BATCH_INF):
        end = min(start + BATCH_INF, N_valid)
        coords = valid_coords_all[start:end]
        specs = cube_cr_l2[coords[:, 0], coords[:, 1], :]  # (batch, 438)
        specs = np.nan_to_num(specs, nan=0.0)
        t = torch.tensor(specs[:, np.newaxis, :]).to(device)  # (batch, 1, 438)
        z = encoder(t).cpu().numpy()  # (batch, 64)
        latent_vectors[start:end] = z

# L2 normalize latent vectors
norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
norms[norms < 1e-8] = 1.0
latent_vectors /= norms

np.save(os.path.join(OUTPUT_DIR, 'latent_vectors.npy'), latent_vectors)
np.save(os.path.join(OUTPUT_DIR, 'latent_coords.npy'), valid_coords_all)
```

---

## Stage 4 — HDBSCAN Density-Based Clustering

### Step 4.1 — Parameter Selection

```python
import hdbscan

N_valid = len(latent_vectors)
min_cluster_size = max(20, int(0.0005 * N_valid))
min_samples = max(5, min_cluster_size // 10)

print(f"  HDBSCAN params: min_cluster_size={min_cluster_size}, "
      f"min_samples={min_samples}, N_valid={N_valid}")
```

### Step 4.2 — Run HDBSCAN

```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    core_dist_n_jobs=-1,
)
clusterer.fit(latent_vectors.astype(np.float64))

labels = clusterer.labels_  # (N_valid,) — -1 for noise
probabilities = clusterer.probabilities_  # (N_valid,)
n_clusters = labels.max() + 1

print(f"  Found {n_clusters} clusters, {(labels == -1).sum()} noise pixels")
```

### Step 4.3 — DBCV Quality Score

```python
try:
    dbcv = hdbscan.validity.validity_index(
        latent_vectors.astype(np.float64), labels)
    print(f"  DBCV score: {dbcv:.4f}")
    if dbcv < 0.1:
        print("  ⚠️ DBCV < 0.1 — results may need review")
except Exception as e:
    print(f"  DBCV computation failed: {e}")
    dbcv = -1.0
```

### Step 4.4 — Automatic Cluster Naming Using SPM + Library Matching

**Two-tier naming strategy** for both common and rare minerals:

**Tier 1 — SPM-Based Naming** (fast, covers major mineral groups):

```python
spm_names = list(spm_indices_to_compute.keys())
spm_mineral_map = {
    'BD1900_2': 'Hydrated Minerals',
    'BD2100_2': 'Monohydrated Sulfates',
    'BD2210_2': 'Al-OH Clays',
    'BD2290': 'Fe/Mg-OH Phyllosilicates',
    'BD2300': 'Fe/Mg Phyllosilicates',
    'BD2500_2': 'Carbonates',
    'OLINDEX3': 'Olivine',
    'LCPINDEX2': 'Low-Ca Pyroxene',
    'HCPINDEX2': 'High-Ca Pyroxene',
    'SINDEX2': 'Sulfates',
    'BD1500_2': 'Water Ice',
    'BD1435': 'CO2 Ice',
}

cluster_names = {}
for cl in range(n_clusters):
    cl_mask = labels == cl
    cl_spm = spm_map_flat[cl_mask]  # (n_pixels_in_cluster, 12)
    mean_spm = np.nanmean(cl_spm, axis=0)
    best_idx = np.nanargmax(mean_spm)
    spm_name = spm_mineral_map[spm_names[best_idx]]
    cluster_names[cl] = spm_name
```

**Tier 2 — Library Spectral Angle Matching** (precise, identifies specific minerals):

```python
from scipy.spatial.distance import cosine

def spectral_angle(s1, s2):
    """Spectral angle in radians between two spectra."""
    finite = np.isfinite(s1) & np.isfinite(s2)
    if finite.sum() < 10:
        return np.pi
    a, b = s1[finite], s2[finite]
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return np.arccos(np.clip(cos_sim, -1, 1))

cluster_mineral_matches = {}
for cl in range(n_clusters):
    cl_mask = labels == cl
    # Mean CR spectrum of cluster
    cl_coords = valid_coords_all[cl_mask]
    cl_spectra = cube_cr[cl_coords[:, 0], cl_coords[:, 1], :]
    mean_spec = np.nanmean(cl_spectra, axis=0)

    # Compare to all library spectra
    best_angle = np.pi
    best_name = "Unknown"
    best_lib_idx = -1
    for li, lib_spec in enumerate(lib_spectra_cr):
        angle = spectral_angle(mean_spec, lib_spec)
        if angle < best_angle:
            best_angle = angle
            best_name = lib_names[li]
            best_lib_idx = li

    cluster_mineral_matches[cl] = {
        'spm_name': cluster_names[cl],
        'library_match': best_name,
        'spectral_angle_rad': float(best_angle),
        'match_quality': 'STRONG' if best_angle < 0.15 else
                         'MODERATE' if best_angle < 0.30 else 'WEAK',
    }
```

### Step 4.5 — Uncertainty Map

```python
# Soft membership (if available)
try:
    soft_membership = hdbscan.all_points_membership_vectors(clusterer)
    uncertainty = 1.0 - np.max(soft_membership, axis=1)
except:
    uncertainty = 1.0 - probabilities
    soft_membership = None

# Noise pixels get uncertainty = 1.0
uncertainty[labels == -1] = 1.0

# Reconstruct 2D maps
label_map = np.full((H, W), -1, dtype=np.int32)
uncertainty_map = np.ones((H, W), dtype=np.float32)
for i, (r, c) in enumerate(valid_coords_all):
    label_map[r, c] = labels[i]
    uncertainty_map[r, c] = uncertainty[i]
```

### Step 4.6 — Save Clustering Outputs

```python
np.save(os.path.join(OUTPUT_DIR, 'cluster_labels.npy'), labels)
np.save(os.path.join(OUTPUT_DIR, 'cluster_uncertainty.npy'), uncertainty)
np.save(os.path.join(OUTPUT_DIR, 'label_map.npy'), label_map)
np.save(os.path.join(OUTPUT_DIR, 'uncertainty_map.npy'), uncertainty_map)
if soft_membership is not None:
    np.save(os.path.join(OUTPUT_DIR, 'soft_membership.npy'), soft_membership)
with open(os.path.join(OUTPUT_DIR, 'cluster_names.json'), 'w') as f:
    json.dump(cluster_mineral_matches, f, indent=2)
```

---

## Stage 5 — Physics-Constrained Neural Unmixing

### Purpose

HDBSCAN gave each pixel a **dominant** mineral label. But each pixel contains a **mixture** of minerals. The unmixer estimates what fraction of each endmember is present in every pixel.

### Step 5.1 — Endmember Selection Using Farthest-Point Sampling

Select a **diverse** subset of library endmembers. Unlike M³ (12 lunar minerals), CRISM on Mars targets a wider mineral palette. Select **N_endmembers** dynamically:

```python
# Start with a diversity-driven selection
N_ENDMEMBERS = min(24, n_clusters + 5)  # More endmembers than clusters
# Ensure we have at least one endmember per detected cluster

# Farthest-point sampling from library
selected = [np.argmax(np.nanmean(lib_spm, axis=1))]  # Start with highest SPM
for _ in range(N_ENDMEMBERS - 1):
    dists_to_selected = np.array([
        min(np.linalg.norm(np.nan_to_num(lib_spectra_cr_l2[i]) -
                           np.nan_to_num(lib_spectra_cr_l2[s]))
            for s in selected)
        for i in range(len(lib_spectra_cr_l2))
    ])
    dists_to_selected[selected] = -1  # exclude already selected
    selected.append(np.argmax(dists_to_selected))

E_init = np.nan_to_num(lib_spectra_cr[selected])  # (N_ENDMEMBERS, 438) - CR, not L2
endmember_names = [lib_names[i] for i in selected]
```

### Step 5.2 — Abundance Network Architecture

```python
class AbundanceNetwork(nn.Module):
    """Predicts mineral abundance fractions from latent vectors."""
    def __init__(self, latent_dim=64, n_endmembers=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_endmembers),
        )

    def forward(self, z):
        """z: (batch, 64) → abundances: (batch, N_endmembers), sum=1, all≥0"""
        logits = self.net(z)
        return F.softmax(logits, dim=1)

# Endmember matrix as learnable parameter
E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32))  # (N_endmembers, 438)
```

### Step 5.3 — Sequential Training Phase A (Encoder Frozen)

```python
# Freeze encoder
for param in encoder.parameters():
    param.requires_grad = False

abundance_net = AbundanceNetwork(latent_dim=64, n_endmembers=N_ENDMEMBERS).to(device)
optimizer_unmix = torch.optim.AdamW(
    list(abundance_net.parameters()) + [E],
    lr=5e-4, weight_decay=1e-4
)
scheduler_unmix = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_unmix, T_max=50)

# Pre-compute latent vectors for training pool
with torch.no_grad():
    pool_t = torch.tensor(pool_spectra[:, np.newaxis, :]).to(device)
    pool_latent = encoder(pool_t).cpu().numpy()  # (N_pool, 64)

# Training targets: continuum-removed spectra (not L2-normalized)
pool_targets = pool_spectra_cr  # (N_pool, 438) - CR, not L2

for epoch in range(50):
    # ... batch training loop ...
    z = torch.tensor(pool_latent[batch_idx]).to(device)
    target_spec = torch.tensor(pool_targets[batch_idx]).to(device)

    abundances = abundance_net(z)  # (batch, N_endmembers)
    reconstructed = torch.mm(abundances, E)  # (batch, 438)

    # Loss 1: Reconstruction MSE
    loss_recon = F.mse_loss(reconstructed, target_spec)

    # Loss 2: Sparsity (L1 on abundances)
    loss_sparse = torch.mean(torch.abs(abundances))

    # Loss 3: Library regularization (endmembers stay near initialization)
    E_init_t = torch.tensor(E_init).to(device)
    loss_lib = F.mse_loss(E, E_init_t)

    loss = loss_recon + 0.01 * loss_sparse + 0.1 * loss_lib
```

### Step 5.4 — Joint Fine-Tuning Phase B

```python
# Unfreeze encoder
for param in encoder.parameters():
    param.requires_grad = True

optimizer_joint = torch.optim.AdamW(
    list(encoder.parameters()) + list(abundance_net.parameters()) + [E],
    lr=5e-5  # 10x lower
)

for epoch in range(10):
    # Same loss as Phase A
    # This allows encoder to adapt representations for better unmixing
    pass

# Re-freeze encoder after fine-tuning
for param in encoder.parameters():
    param.requires_grad = False
```

### Step 5.5 — Full Scene Inference

```python
abundance_net.eval()
abundance_map = np.full((H, W, N_ENDMEMBERS), np.nan, dtype=np.float32)
recon_error_map = np.full((H, W), np.nan, dtype=np.float32)

with torch.no_grad():
    for start in range(0, N_valid, 4096):
        end = min(start + 4096, N_valid)
        z = torch.tensor(latent_vectors[start:end]).to(device)
        abundances = abundance_net(z)  # (batch, N_endmembers)
        reconstructed = torch.mm(abundances, E)  # (batch, 438)

        coords = valid_coords_all[start:end]
        ab_np = abundances.cpu().numpy()
        re_np = reconstructed.cpu().numpy()

        for i, (r, c) in enumerate(coords):
            abundance_map[r, c, :] = ab_np[i]
            actual = cube_cr[r, c, :]
            recon_error_map[r, c] = np.nanmean((re_np[i] - actual) ** 2)

# Dominant mineral map
dominant_map = np.full((H, W), -1, dtype=np.int32)
for r in range(H):
    for c in range(W):
        if valid_pixel_mask[r, c]:
            dominant_map[r, c] = np.nanargmax(abundance_map[r, c, :])

np.save(os.path.join(OUTPUT_DIR, 'abundance_map.npy'), abundance_map)
np.save(os.path.join(OUTPUT_DIR, 'recon_error_map.npy'), recon_error_map)
np.save(os.path.join(OUTPUT_DIR, 'dominant_map.npy'), dominant_map)
```

### Step 5.6 — Physical Validity Checks

```python
# Check 1: All abundances ≥ 0 (guaranteed by softmax)
assert np.all(abundance_map[valid_pixel_mask] >= 0), "Negative abundances found!"

# Check 2: Row sums = 1.0
row_sums = np.nansum(abundance_map[valid_pixel_mask], axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-4), "Row sums not 1.0!"

# Check 3: Mean of max abundance should be 0.3–0.9
max_abundances = np.nanmax(abundance_map[valid_pixel_mask], axis=1)
mean_max = np.mean(max_abundances)
print(f"  Mean max abundance: {mean_max:.3f} (expected 0.3–0.9)")

# Check 4: Dominant mineral agreement with HDBSCAN
# Compare unmixer dominant mineral with cluster label
agreement = np.mean(dominant_map[valid_pixel_mask] == labels)  # approximate
print(f"  Unmixer-HDBSCAN agreement: {agreement*100:.1f}%")
```

---

## Stage 6 — CRF Spatial Smoothing

### Step 6.1 — SLIC Superpixel Generation

```python
from skimage.segmentation import slic

# Create 3-channel false-color from CR cube
# Use bands near 1.3, 1.8, 2.3 µm (mineralogically meaningful)
ch1_bi = np.argmin(np.abs(waves - 1.30))
ch2_bi = np.argmin(np.abs(waves - 1.80))
ch3_bi = np.argmin(np.abs(waves - 2.30))

rgb = np.stack([
    np.nan_to_num(cube_cr[:, :, ch1_bi]),
    np.nan_to_num(cube_cr[:, :, ch2_bi]),
    np.nan_to_num(cube_cr[:, :, ch3_bi]),
], axis=2)
# Normalize each channel to [0, 1]
for ch in range(3):
    vmin, vmax = np.percentile(rgb[:, :, ch][valid_pixel_mask], [2, 98])
    rgb[:, :, ch] = np.clip((rgb[:, :, ch] - vmin) / (vmax - vmin + 1e-8), 0, 1)

n_segments = max(50, valid_pixel_mask.sum() // 50)
segments = slic(rgb, n_segments=n_segments, compactness=10, sigma=1.0,
                start_label=0, mask=valid_pixel_mask)
```

### Step 6.2–6.3 — Weighted Majority Vote Smoothing

```python
from collections import Counter
from scipy import ndimage

def smooth_labels(label_map, segments, uncertainty_map, cube_cr, waves, n_iterations=5):
    """Weighted majority vote smoothing using superpixels."""
    smoothed = label_map.copy()
    H, W = label_map.shape

    for iteration in range(n_iterations):
        new_labels = smoothed.copy()
        unique_segs = np.unique(segments)

        for seg_id in unique_segs:
            if seg_id < 0:
                continue
            seg_mask = segments == seg_id
            if not seg_mask.any():
                continue

            # Current segment's dominant label
            seg_labels = smoothed[seg_mask]
            valid_labels = seg_labels[seg_labels >= 0]
            if len(valid_labels) == 0:
                continue

            # Find adjacent segments
            dilated = ndimage.binary_dilation(seg_mask, iterations=1)
            neighbor_mask = dilated & ~seg_mask

            # Gather neighbor votes weighted by inverse spectral distance
            votes = Counter(valid_labels.tolist())
            # Weight own evidence by inverse uncertainty
            own_unc = np.mean(uncertainty_map[seg_mask])
            own_weight = 1.0 / (own_unc + 0.01)
            for lbl, count in votes.items():
                votes[lbl] = count * own_weight

            # Neighbor contributions
            neighbor_segs = np.unique(segments[neighbor_mask])
            for nseg in neighbor_segs:
                if nseg < 0:
                    continue
                nseg_mask = segments == nseg
                nseg_labels = smoothed[nseg_mask]
                nseg_valid = nseg_labels[nseg_labels >= 0]
                if len(nseg_valid) == 0:
                    continue

                # Spectral distance between segments
                seg_spec = np.nanmean(cube_cr[seg_mask], axis=0)
                nseg_spec = np.nanmean(cube_cr[nseg_mask], axis=0)
                spec_dist = spectral_angle(seg_spec, nseg_spec)

                # Boundary preservation: don't smooth across large spectral differences
                if spec_dist > 0.3:
                    continue

                weight = 1.0 / (spec_dist + 0.01)
                for lbl in nseg_valid:
                    votes[lbl] = votes.get(lbl, 0) + weight

            # Assign most-voted label
            if votes:
                best_label = max(votes, key=votes.get)
                new_labels[seg_mask] = best_label

        smoothed = new_labels

    return smoothed

smoothed_label_map = smooth_labels(label_map, segments, uncertainty_map, cube_cr, waves)
np.save(os.path.join(OUTPUT_DIR, 'smoothed_label_map.npy'), smoothed_label_map)
```

### Step 6.4 — Smooth Abundance Maps

```python
smoothed_abundance = abundance_map.copy()

unique_segs = np.unique(segments)
for seg_id in unique_segs:
    if seg_id < 0:
        continue
    seg_mask = segments == seg_id
    if not seg_mask.any():
        continue
    # Mean abundance per superpixel
    seg_abund = np.nanmean(abundance_map[seg_mask], axis=0)
    smoothed_abundance[seg_mask] = seg_abund

# Renormalize so abundances sum to 1.0
for r in range(H):
    for c in range(W):
        if valid_pixel_mask[r, c]:
            total = np.nansum(smoothed_abundance[r, c, :])
            if total > 1e-6:
                smoothed_abundance[r, c, :] /= total

np.save(os.path.join(OUTPUT_DIR, 'smoothed_abundance_map.npy'), smoothed_abundance)
```

---

## Stage 7 — Validation and Publication Figures

### Tier 1 — SPM Detection Accuracy

```python
validation_results = {}

for cl in range(n_clusters):
    cl_mask = labels == cl
    match = cluster_mineral_matches[cl]
    spm_name = match['spm_name']

    # Find the corresponding SPM index
    spm_idx = None
    for j, (name, mineral_group) in enumerate(spm_mineral_map.items()):
        if mineral_group == spm_name:
            spm_idx = j
            break

    if spm_idx is None:
        continue

    spm_vals = spm_map_flat[:, spm_idx]
    finite = np.isfinite(spm_vals)
    if finite.sum() == 0:
        continue

    threshold = np.percentile(spm_vals[finite], 80)
    top_mask = finite & (spm_vals >= threshold)

    # What fraction of top-20% SPM pixels are in this cluster?
    recall = np.sum(cl_mask & top_mask) / max(1, top_mask.sum())

    validation_results[f"cluster_{cl}"] = {
        'spm_name': spm_name,
        'library_match': match['library_match'],
        'spectral_angle': match['spectral_angle_rad'],
        'spm_recall_top20': float(recall),
        'n_pixels': int(cl_mask.sum()),
    }
```

### Tier 2 — Cluster-Library Spectral Angle

Already computed in Stage 4 Step 4.4 Tier 2 naming.

### Tier 3 — Uncertainty Quality

```python
# Sub-metric A: Spearman correlation between uncertainty and SPM signal
from scipy.stats import spearmanr
spm_max = np.nanmax(spm_map_flat, axis=1)
finite = np.isfinite(spm_max) & np.isfinite(uncertainty)
if finite.sum() > 100:
    corr_spm_unc, _ = spearmanr(spm_max[finite], uncertainty[finite])
    print(f"  Spearman(uncertainty, SPM): {corr_spm_unc:.3f} (expected: -0.3 to -0.8)")

# Sub-metric B: Boundary vs interior uncertainty
# Interior = pixel with all 8 neighbors having same label
interior_mask = np.zeros(N_valid, dtype=bool)
boundary_mask = np.zeros(N_valid, dtype=bool)
for i, (r, c) in enumerate(valid_coords_all):
    if labels[i] == -1:
        continue
    is_boundary = False
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W:
            if label_map[nr, nc] != labels[i]:
                is_boundary = True
                break
    if is_boundary:
        boundary_mask[i] = True
    else:
        interior_mask[i] = True

if interior_mask.any() and boundary_mask.any():
    ratio = np.mean(uncertainty[boundary_mask]) / np.mean(uncertainty[interior_mask])
    print(f"  Boundary/Interior uncertainty ratio: {ratio:.2f} (expected: > 2.0)")

# Sub-metric C: Correlation between uncertainty and reconstruction error
recon_errors = recon_error_map[valid_pixel_mask]
finite = np.isfinite(recon_errors) & np.isfinite(uncertainty)
if finite.sum() > 100:
    corr_unc_recon = np.corrcoef(uncertainty[finite], recon_errors[finite])[0, 1]
    print(f"  Pearson(uncertainty, recon_error): {corr_unc_recon:.3f} (expected: 0.3–0.6)")
```

### Publication Figures

Generate **6 publication-quality figures**:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Use the dark theme already set up in existing stages
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9', 'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
})

# Figure 1: Smoothed mineral classification map + uncertainty map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# ... colormap with discrete colors for each cluster ...

# Figure 2: N_endmember-panel abundance maps grid
fig, axes = plt.subplots(4, 6, figsize=(20, 14))  # adjust grid size
# Each panel: one endmember's abundance map

# Figure 3: Uncertainty validation scatter plots (3 sub-metrics)

# Figure 4: SPM vs cluster bar chart comparison

# Figure 5: Cluster mean spectra vs library reference spectra overlaid

# Figure 6: Reconstruction error map
```

### Validation JSON Report

```python
report = {
    'scene': sname,
    'n_clusters': n_clusters,
    'n_noise_pixels': int((labels == -1).sum()),
    'n_endmembers': N_ENDMEMBERS,
    'dbcv_score': float(dbcv),
    'tier1_spm_accuracy': validation_results,
    'tier2_library_match': cluster_mineral_matches,
    'tier3_uncertainty': {
        'spearman_spm_unc': float(corr_spm_unc) if 'corr_spm_unc' in dir() else None,
        'boundary_interior_ratio': float(ratio) if 'ratio' in dir() else None,
        'pearson_unc_recon': float(corr_unc_recon) if 'corr_unc_recon' in dir() else None,
    },
    'physical_validity': {
        'abundances_positive': True,
        'row_sums_one': True,
        'mean_max_abundance': float(mean_max),
    },
    'overall_verdict': 'STRONG' if dbcv > 0.3 else ('MODERATE' if dbcv > 0.1 else 'WEAK'),
    'endmember_names': endmember_names,
}

with open(os.path.join(OUTPUT_DIR, 'mineral_identification_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
```

---

## File & Directory Conventions

### Output Directory Structure

```
D:\NEW CROSS MISSION\Data\Scene 12\Crism\mineral_output\
├── lib_spectra_438_cr_l2.npy       ← Stage 1: Library preprocessed
├── lib_spectra_438_cr.npy          ← Stage 1: Library CR (not L2)
├── lib_spm.npy                     ← Stage 1: Library SPM values
├── lib_names.json                  ← Stage 1: Mineral name list
├── cube_cr.npy                     ← Stage 2: Full scene CR cube
├── cube_cr_l2.npy                  ← Stage 2: Full scene CR+L2 cube
├── spm_map.npy                     ← Stage 2: Full scene SPM map
├── encoder_B438.pth                ← Stage 3: Trained encoder model
├── latent_vectors.npy              ← Stage 3: All valid pixel latent vectors
├── latent_coords.npy               ← Stage 3: (row, col) for each latent vector
├── cluster_labels.npy              ← Stage 4: HDBSCAN labels
├── cluster_uncertainty.npy         ← Stage 4: Per-pixel uncertainty
├── label_map.npy                   ← Stage 4: 2D label map (H, W)
├── uncertainty_map.npy             ← Stage 4: 2D uncertainty map (H, W)
├── cluster_names.json              ← Stage 4: Cluster mineral assignments
├── abundance_map.npy               ← Stage 5: (H, W, N_endmembers)
├── recon_error_map.npy             ← Stage 5: Reconstruction error map
├── dominant_map.npy                ← Stage 5: Dominant mineral per pixel
├── smoothed_label_map.npy          ← Stage 6: Spatially smoothed labels
├── smoothed_abundance_map.npy      ← Stage 6: Smoothed abundances
├── fig1_classification.png         ← Stage 7: Classification + uncertainty
├── fig2_abundance_panels.png       ← Stage 7: Per-endmember maps
├── fig3_uncertainty_validation.png ← Stage 7: Uncertainty scatter plots
├── fig4_spm_comparison.png         ← Stage 7: SPM bar chart
├── fig5_spectra_comparison.png     ← Stage 7: Cluster vs library spectra
├── fig6_recon_error.png            ← Stage 7: Reconstruction error
└── mineral_identification_report.json  ← Stage 7: Full validation report
```

### Integration with Run_pipeline.py

The orchestrator already expects `Stage5_minerals.py` with a `run_mineral_identification()` function:

```python
# In Run_pipeline.py (already exists):
if 5 in stages:
    from Stage5_minerals import run_mineral_identification
    summary = run_mineral_identification()
```

So the main file should expose:
```python
def run_mineral_identification(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """Runs all 7 internal stages of mineral identification."""
    # Stage 0: Setup
    # Stage 1: Library preprocessing
    # Stage 2: Scene CR + SPM
    # Stage 3: Encoder training
    # Stage 4: HDBSCAN clustering
    # Stage 5: Neural unmixing
    # Stage 6: Spatial smoothing
    # Stage 7: Validation + figures
    return summary_dict
```

---

## Dependency List

```
# Already in Requirements.txt:
numpy
scipy
matplotlib
pandas
torch

# NEW dependencies to add:
hdbscan>=0.8.33
scikit-image>=0.19.0     # for SLIC superpixels
scikit-learn>=1.0         # for various utilities
```

Install command:
```bash
pip install hdbscan scikit-image scikit-learn
```

---

## Critical Implementation Warnings

### 1. CRISM ≠ M³ — Key Differences

| Property | M³ (Moon) | CRISM (Mars) |
|----------|-----------|--------------|
| Bands | 85 | 438 (L-sensor) |
| Wavelength range | 0.46–2.97 µm | 1.028–3.920 µm |
| FWHM | 20–30 nm | ~6.55 nm |
| Spatial resolution | 140 m/pixel | 18 m/pixel |
| Target body | Moon | Mars |
| Atmosphere | None | CO₂ atmosphere |
| Bad bands | Few (>2.80 µm) | Many (atmospheric, filter boundary) |
| Library | RELAB | USGS splib07 (CRISM-convolved) |
| Target minerals | 12 (lunar) | 212+ (martian) |

### 2. Do NOT Apply SRF Convolution

The USGS `ASCIIdata_splib07b_cvCRISMjMTR3` spectra are **already convolved to CRISM bandpass**. Applying Gaussian SRF convolution again would double-convolve them, producing incorrect results.

### 3. Handle the 489 → 438 Channel Mismatch

The library has 489 channels (S+L sensor joined). The scene cube has 438 bands (L-sensor only). You MUST subset the library to match the scene's wavelength range.

### 4. Bad-Band Handling in SPM Computation

Many SPM center/shoulder wavelengths fall in or near bad-band regions. When computing SPM indices, ALWAYS check that the nearest band is (a) not in a bad region and (b) within 30 nm of the target wavelength. Return NaN if either check fails.

### 5. Memory Management

- The 438-band cube is 5× larger per pixel than 85-band M³
- Process continuum removal in spatial blocks (64–128 rows)
- Process encoder inference in batches of 2048–4096
- HDBSCAN requires full latent array in RAM (~N_valid × 64 × 4 bytes)

### 6. L2 Normalization After Continuum Removal

Both library and scene spectra MUST undergo:
1. NaN interpolation
2. Convex hull continuum removal
3. Bad-band NaN restoration
4. L2 normalization

In that exact order. Skipping or reordering any step makes the comparison invalid.

### 7. Encoder Input Shape

```python
# Input to encoder: (batch, 1, 438) — NOT (batch, 438)
# The Conv1d layers expect (batch, channels, length)
x = spectrum.unsqueeze(1)  # Add channel dimension
```

### 8. Softmax for Abundance Constraints

The unmixer output MUST use `F.softmax(logits, dim=1)` — NOT sigmoid. Softmax simultaneously enforces:
- Non-negativity (all fractions ≥ 0)
- Sum-to-one (fractions sum to 1.0)
These are the two physical requirements for valid linear mixture fractions.

---

## Rare Mineral Detection Strategy

### The Challenge

Common minerals (pyroxene, plagioclase, olivine) dominate most pixels. Rare minerals (e.g., prehnite, serpentine, opal, carbonates, zeolites, sulfides) occupy very few pixels and can be overwhelmed by noise or majority-class bias.

### Strategy 1: HDBSCAN's Noise Category as a Rare-Mineral Reservoir

Pixels labeled as "noise" (label = -1) by HDBSCAN are not in any dense cluster. Many of these are **genuinely different** spectra — potentially rare minerals. 

**Implementation**:
```python
noise_pixels = valid_coords_all[labels == -1]
noise_spectra = cube_cr[noise_pixels[:, 0], noise_pixels[:, 1], :]

# Run spectral angle matching against ALL 1,276 library spectra
for i, spec in enumerate(noise_spectra):
    best_angle = np.pi
    best_match = "Unknown"
    for li, lib_spec in enumerate(lib_spectra_cr):
        angle = spectral_angle(spec, lib_spec)
        if angle < best_angle:
            best_angle = angle
            best_match = lib_names[li]
    if best_angle < 0.20:  # Strong match
        rare_detections.append({
            'pixel': noise_pixels[i].tolist(),
            'mineral': best_match,
            'spectral_angle': float(best_angle),
            'confidence': 'HIGH' if best_angle < 0.10 else 'MODERATE'
        })
```

### Strategy 2: Abundance-Based Rare Mineral Detection

After unmixing, any pixel where a non-dominant endmember has abundance > 10% may contain a rare mineral as a secondary component.

```python
# For each pixel, check if any non-dominant endmember has significant abundance
for r in range(H):
    for c in range(W):
        if not valid_pixel_mask[r, c]:
            continue
        abund = abundance_map[r, c, :]
        dominant = np.argmax(abund)
        for em_idx in range(N_ENDMEMBERS):
            if em_idx == dominant:
                continue
            if abund[em_idx] > 0.10:  # > 10% of pixel
                minor_detections.append({
                    'pixel': [int(r), int(c)],
                    'mineral': endmember_names[em_idx],
                    'abundance': float(abund[em_idx]),
                    'dominant': endmember_names[dominant],
                })
```

### Strategy 3: Per-Pixel Full Library Search (Top-K Matches)

For the most comprehensive rare mineral identification, compute spectral angle to **every** library spectrum for **every** pixel. This is expensive but exhaustive.

```python
# Optimized: Use matrix operations
lib_matrix = np.nan_to_num(lib_spectra_cr_l2)  # (N_lib, 438)

# For each pixel's CR+L2 spectrum, compute cosine similarity to all library entries
# cosine_sim = dot(pixel, lib) / (|pixel| * |lib|)  — already L2-normalized, so just dot product
for batch_start in range(0, N_valid, 1000):
    batch_end = min(batch_start + 1000, N_valid)
    coords = valid_coords_all[batch_start:batch_end]
    specs = np.nan_to_num(cube_cr_l2[coords[:, 0], coords[:, 1], :])  # (batch, 438)

    # Cosine similarity matrix
    sims = specs @ lib_matrix.T  # (batch, N_lib)

    # Top-3 matches per pixel
    top3 = np.argsort(sims, axis=1)[:, -3:][:, ::-1]
    top3_scores = np.take_along_axis(sims, top3, axis=1)

    # Store results
    for i in range(len(coords)):
        pixel_matches[tuple(coords[i])] = [
            {'mineral': lib_names[top3[i, k]], 'cos_sim': float(top3_scores[i, k])}
            for k in range(3)
        ]
```

### Strategy 4: Sub-Clustering Noise Pixels

Re-run HDBSCAN on noise pixels with **much lower** `min_cluster_size`:

```python
noise_latent = latent_vectors[labels == -1]
if len(noise_latent) > 50:
    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(5, len(noise_latent) // 100),
        min_samples=3,
        metric='euclidean',
    )
    sub_labels = sub_clusterer.fit_predict(noise_latent.astype(np.float64))
    n_sub = sub_labels.max() + 1
    print(f"  Sub-clustering found {n_sub} rare mineral groups in noise pixels")

    # Name each sub-cluster by library matching
    for sc in range(n_sub):
        sc_mask = sub_labels == sc
        sc_specs = noise_spectra_cr[sc_mask]
        mean_spec = np.nanmean(sc_specs, axis=0)
        # Match to library
        ...
```

---

## The Correct Mental Model for the Agent

> **CRITICAL INSTRUCTION**: This is a single continuous pipeline that runs in order from Stage 0 through Stage 7 without branching or baseline-first logic. **Every stage runs every time.** The stages are:

| Stage | Purpose | Depends On |
|-------|---------|-----------|
| 0 | Load denoised cube + metadata | Stage 4 output (_S4.npy) |
| 1 | Preprocess USGS library to match CRISM | Cube wavelengths + bad masks |
| 2 | CR + SPM computation on full scene | Stage 0 cube + Stage 1 library |
| 3 | Train contrastive encoder (self-supervised) | Stage 2 scene data + Stage 1 library |
| 4 | HDBSCAN clustering on encoder output | Stage 3 latent vectors |
| 5 | Neural unmixing for abundance estimation | Stage 3 encoder + Stage 4 clusters |
| 6 | Spatial smoothing via superpixels | Stage 4 labels + Stage 5 abundances |
| 7 | Validation + figures + report | All prior stages |

**No stage should be skipped. No "baseline first" logic. No branching.**

### Key Differences from M³ Methodology to Remember

1. **438 bands, not 85** — affects all Conv1d kernel sizes and linear layer dimensions
2. **USGS CRISM-convolved library** — already at CRISM spectral resolution, NO SRF convolution needed
3. **489 → 438 channel subsetting** — library has S+L joined, scene has L only
4. **212 unique minerals, not 12** — need dynamic endmember count, not fixed 12
5. **Mars atmosphere** — many more bad bands than Moon, affects SPM computation
6. **12 SPM indices, not 8** — more diagnostic parameters available for Mars
7. **18 m/pixel** — higher spatial resolution means less sub-pixel mixing than M³
8. **Physical I/F range** — clamped to [-0.05, 1.30] by existing pipeline
