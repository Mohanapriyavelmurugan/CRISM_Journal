# Mineral ID Fixes (Full-Height FRT) — Update

**Date:** 2026-10-04  
**Scenes:** `D:\CRISM scenes\scenes\Scene 1..5` (full-height L-sensor TRR3)

## Issues addressed in `Stage5_minerals.py`

| Issue | Fix |
|------|-----|
| Tiny 15×64 footprint | Stage0 loads new full-height FRT `.img`+`.lbl` (420–540 × 640 × 438); warns if H&lt;100 |
| SPM NaNs (BD1900/BD2210/SINDEX2) | Relaxed `spm_bad_mask` allows Viviano-Beck diagnostic bands |
| Artifact SPMs | `sanitize_spm_map` disables scene-wide ice / all-negative OLINDEX3/LCP |
| Library→Olivine collapse | Mars-prior library matching on good bands |
| SPM vs library conflict | Hybrid `authoritative_label` policy in `cluster_names.json` |
| Exotic endmembers | Mars-prior farthest-point + cluster-seeded endmembers |
| Smoothing kills minorities | Protected minority clusters in SLIC vote |
| Weak uncertainty | Composite: 0.45 HDBSCAN + 0.35 (1−max abund) + 0.20 recon error |
| Noisy rare detection | Stricter Strategy 2 (rare list only, abund&gt;0.25); Strategy 1 Mars-prior/rare |
| No external GT | Report marks `labels_provisional=true` + disclaimer |

## Also for large cubes

- Vectorized SPM maps
- Fast max-filter continuum when &gt;20k valid pixels
- HDBSCAN fit on ≤40k subsample + `approximate_predict`
- Lower default batches/epochs for 4 GB GPU

## How to run

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"

# Scene 1 (default)
python run_mineral_fullfrt.py --scene 1

# Or all five
python run_mineral_fullfrt.py --all

# Direct
$env:CRISM_DATA_DIR = "D:\CRISM scenes\scenes\Scene 1"
python Stage5_minerals.py
```

**Note:** If `ml_output/*_S4.npy` is absent, Stage5 loads the TRR3 `.img` directly (with a warning). For best science, run Stages 1–4 on these scenes first, then Stage 5.

## SPM source fix (2026-10-04)

Stage 3/4 blanks atmospheric caution bands to NaN in `*_S4.npy`, which made BD1900/BD2210/SINDEX2 etc. print `unavailable`.

**Fix:** Stage 5 now computes Viviano-Beck SPM maps from the original TRR3 `.img` (caution bands intact), while continuum removal / encoder / unmixing still use denoised S4.

## Post-run science fixes (2026-10-04, Scene 1 WEAK review)

Core methodology unchanged (Stages 0–7, USGS CRISM-convolved library, Viviano-Beck SPM on I/F, contrastive encoder, HDBSCAN, softmax ASC, hybrid SPM/library labels, SLIC).

| Issue from Scene 1 run | Fix (still in `Stage5_minerals.py`) |
|------------------------|-------------------------------------|
| SLIC filled 90% noise → cluster 0 | Smooth **only** pixels with label ≥ 0; noise stays −1 |
| Softmax collapsed to ~95% Olivine | Unique-family FPS + min SAM between EMs; diversity + use losses |
| Duplicate Olivine/Kieserite EMs | `mineral_family()` uniqueness in endmember selection |
| HDBSCAN 2 clusters / 69% noise | Latent+SPM features; smaller `min_cluster_size`; leaf retry |
| Flat BD1900 naming “Hydrated” | Sanitize scene-wide hydration SPMs |
| OLINDEX3 dead on IMG vs OK on S4 | Mafic SPM fallback recompute on S4 I/F |
| Library SAM tied all clusters to Olivine | SPM-first when SAM ≥ 0.15; avoid same-family rematch |
| Kieserite rare FPs under pure Olivine | Strategy2: abund≥0.30, max_dom&lt;0.85, SPM co-support |
| Uncertainty ceiling 0.85 everywhere | Soft noise floor `NOISE_UNC_FLOOR=0.70` |

## Run-2 science fixes (2026-10-05)

Core Stage 0–7 methodology preserved. Addresses WEAK full-FRT diagnostics:

| Issue | Scientific fix |
|-------|----------------|
| CR/atm artifact (~1.55 µm) | Classic CRISM **bland-region spectral ratio** before SPM + CR |
| Library SAM stuck ~0.19 | Windowed SAM excluding artifact bands |
| Leaf HDBSCAN micro-stripes (1.4% coverage) | **EOM-only**, larger `min_cluster_size`, stripe geometry filter, optional UMAP, same-SPM-family merge |
| Softmax Olivine collapse / dead EMs | SPM-hot library seeds; **freeze E to USGS library**; prune unused EMs |
| OLINDEX3 dead | Recompute on ratioed S4; Stage4-style 3-pt diagnostic fallback |
| SPM naming without scene support | Require hotspot overlap (scene p80) for SPM group names |
| SPM ALL NAN / unavailable after bland | **Bug:** S4 bland (caution bands NaN) was applied to IMG SPM cube. **Fix:** separate bland for IMG (SPM) vs S4 (CR) |
| HDBSCAN AttributeError / hang (numba) | Windows numba parfor hang during HDBSCAN JIT. **Fix:** `NUMBA_NUM_THREADS=1`, `algorithm=generic` first, fit≤20k, sklearn fallback; UMAP off by default |

## Run-3 science fixes (2026-10-05, Scene 1 WEAK → address coverage/labels)

Core Stage 0–7 methodology preserved (USGS CRISM-convolved library, Viviano-Beck SPM,
bland ratio, contrastive encoder, HDBSCAN EOM, softmax ASC with frozen E, hybrid labels, SLIC).

| Issue from Scene 1 (94% noise, dual-Pyroxene, EM collapse) | Fix |
|-----------------------------------------------------------|-----|
| Vertical pushbroom striping in labels | **Column destripe** (additive col-median residual) before bland ratio |
| Edge columns saturated uncertainty | Mask ±`EDGE_COL_MARGIN` samples from `valid_pixel_mask` |
| Encoder trained on bland dust | SPM-gated training (ice SPM excluded; ≤15% bland fill) |
| BD1500/BD1435 dominate cluster feats | Ice SPM weight 0.05 in latent+SPM fusion; mild xy compactness |
| HDBSCAN ~94% noise | Smaller EOM `min_cluster_size`; retry if noise>75%; **SPM-active noise rescue** to nearest centroid |
| Both clusters labeled Pyroxene (SAM~0.01) | SAM top-5 + margin; SPM wins on suspicious tight SAM / family disagree |
| Softmax ~89% one EM; Alunite/Siderite without SPM | SPM-co-supported EM pool; stronger use/dom losses; report dead EMs |
| Windows `OSError 22` on `cube_cr.npy` | `safe_np_save` (temp → delete → rename) |

Re-run:

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
$env:CRISM_USE_UMAP = "0"
python run_mineral_fullfrt.py --scene 1
```

## Run-4 science fixes (2026-10-05) — prove minerals present

Core methodology unchanged. Restores diagnostic bands and adds evidence package.

| Fix | Change |
|-----|--------|
| **A** | CR/encoder/SAM on bland-ratioed **TRR3 IMG** with `mineral_bad_mask` keeping 1.05/1.9/2.2 µm |
| **B** | Adaptive SPM gates: `max(Viviano_thr, p90)` + component ≥50; ice excluded |
| **C** | SPM-first labels when SAM top-2 margin is tiny |
| **D** | HDBSCAN only on SPM-active domain (**K still free**); bland stays −1 |
| **E** | Interpolate ~1.55 µm CR notch; SAM excludes notch only |
| **F** | `mineral_proof.json` + `fig7_mineral_proof.png` → report `minerals_present` |
