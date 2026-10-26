# CRISM Mineral Identification — Status Report

**Project:** CRISM Journal / `cv-crism`  
**Pipeline:** Stage 5 mineral identification (`Stage5_minerals.py`)  
**Primary test scene:** Scene 1 — `FRT0000BF9E` (full-height L-sensor TRR3)  
**Date:** 5 October 2026  

---

## 1. Goal

### What we want

Identify **which minerals are really present** in each CRISM scene, and **where** they occur, in a way that is:

1. **Scientifically defensible** — based on established CRISM practice (summary parameters, spectral library matching, continuum removal), not guesswork.  
2. **Automatic and scalable** — works on full-height FRT cubes without hand-picking ROIs for every scene.  
3. **Honest about uncertainty** — when evidence is weak, the system should say so (provisional labels, noise, “not present”) rather than invent minerals.  
4. **Traceable** — every claimed mineral should be backed by measurable evidence (SPM hotspot, diagnostic band depths, library match), saved in reports and figures.

### Success looks like

For a scene such as Scene 1, we can state clearly, for example:

> “Olivine is present in spatially coherent regions; evidence = adaptive OLINDEX3 hotspot + continuum-removed absorption near 1 µm + library support. Hydrated clays / carbonates are not supported at detection thresholds.”

…and produce maps and a `mineral_proof.json` that a reviewer can check without re-running the whole analysis by hand.

---

## 2. How we identify minerals (methodology)

We **have not changed the core scientific methodology**. The pipeline is a fixed linear sequence. What we have refined is *data hygiene*, *gates*, and *evidence reporting* around that core.

### 2.1 End-to-end pipeline (Stages 0–7)

| Stage | Name | Role |
|------:|------|------|
| 0 | Setup & load | Load full-height CRISM L-sensor cube (prefer denoised S4; keep original TRR3 `.img` for SPM). Build bad-band and valid-pixel masks. |
| 1 | USGS library | Load CRISM-convolved USGS spectral library (`splib07b_cvCRISMjMTR3`). **No extra SRF.** Continuum-remove library spectra to match the scene. |
| 2 | Continuum + SPM | Bland-region spectral ratio → Viviano–Beck summary parameters (SPM) → continuum removal (CR) for ML/library. |
| 3 | Encoder | 1D CNN contrastive encoder (NT-Xent + SPM anchor) learns a latent embedding of spectra. |
| 4 | Clustering | **HDBSCAN** (EOM) finds a **data-driven** number of spectral units (K is not fixed). |
| 5 | Unmixing | Softmax abundance network with **library endmembers frozen** (ASC-style abundances ≥0, sum≈1). |
| 6 | Smoothing | SLIC spatial vote **only on clustered pixels**; noise stays unlabeled (−1). |
| 7 | Validate & prove | Figures, JSON report, and mineral-presence proof package. |

### 2.2 Scientific building blocks (what each tool is for)

```
Ratioed I/F  →  Viviano–Beck SPM     →  “Where might mineral X be?”
             →  Continuum removal    →  “What does the absorption shape look like?”
             →  USGS library SAM     →  “Which library mineral matches that shape?”
             →  Encoder + HDBSCAN    →  “Which pixels form coherent spectral units?”
             →  Softmax unmixing     →  “Rough mixture fractions under ASC.”
```

| Method | Purpose | What it is *not* |
|--------|---------|------------------|
| **Bland-region ratio** | Suppress atmosphere / continuum so mineral bands stand out (classic CRISM practice). | Not a mineral classifier by itself. |
| **Viviano–Beck SPM** (OLINDEX3, HCPINDEX2, BD1900_2, …) | Fast, peer-reviewed indices tied to mineral *groups*. | Not a specimen-level name (e.g. “Forsterite Fo70”). |
| **Continuum removal (CR)** | Isolate absorption features for matching and learning. | Useless if diagnostic wavelengths were blanked. |
| **Library SAM** | Match mean unit spectrum to USGS CRISM-convolved specimens. | Misleading if all candidates have nearly the same angle. |
| **HDBSCAN** | Discover how many distinct spectral *units* exist (dynamic K). | Does not name minerals; labeling comes after. |
| **Neural unmixing** | Soft abundances with physical ASC constraints. | Can collapse to one endmember if the scene is bland. |

### 2.3 Labeling policy (hybrid)

For each HDBSCAN cluster we combine:

1. **SPM group** — which summary parameter is enriched in that cluster (e.g. Olivine via OLINDEX3).  
2. **Library match** — best USGS spectrum by spectral angle (with top-5 and margin).  
3. **Authoritative label** — hybrid rule: strong, discriminative library match can win; if SAM is tied or conflicts with SPM, **SPM group is preferred**.

Names are marked **provisional** without an external published geologic map for that FRT.

### 2.4 Important design choice: “noise”

In this pipeline, **noise (label −1)** means:

> “HDBSCAN did not assign this pixel to a dense spectral unit.”

It usually includes spectrally bland dust/continuum, edge columns, and pixels outside the SPM-active domain. It does **not** mean “delete the data” or “sensor failure only.”

---

## 3. What we have done so far

### 3.1 Infrastructure

- Full-height FRT support (e.g. ~450 × 640 × 438), not the old tiny subset footprint.  
- Runner: `run_mineral_fullfrt.py --scene 1` (and `--all`).  
- Outputs under each scene’s `mineral_output/` (maps, figures, JSON reports).  
- Windows-safe large `.npy` saves (`safe_np_save`) to avoid overwrite crashes.  
- GPU path (e.g. GTX 1650) with reduced batch/fit sizes for stability.

### 3.2 Science-preserving fixes (methodology kept)

Work focused on making the **same** methodology *work correctly* on real full FRTs:

| Theme | What we implemented |
|-------|---------------------|
| **SPM source** | Compute Viviano–Beck indices on original TRR3 `.img` (caution bands intact); S4 alone blanked BD1900/BD2210/etc. |
| **Bland ratio** | Separate bland spectra for IMG (SPM) vs S4 where needed; then ratio before SPM/CR. |
| **Column striping** | Additive column-median destripe before bland ratio. |
| **Edge columns** | Mask ±20 samples (bad optics / high uncertainty). |
| **Diagnostic bands (critical)** | Continuum removal / encoder / library SAM now use **bland-ratioed IMG** with a mineral bad-mask that **keeps** ~1.05, ~1.9, ~2.2 µm (previously blanked by the Stage 3/4 encoding mask). |
| **1.55 µm artifact** | Interpolate notch in CR; exclude from SAM windows only. |
| **Adaptive SPM gates** | Detection = `max(Viviano threshold, scene p90)` + connected component ≥ 50 pixels; ice indices excluded from the active domain. |
| **Clustering domain** | HDBSCAN runs on the **SPM-active** pixel set only; **K remains free**. Bland continuum is not forced into clusters. |
| **Label honesty** | Top-5 SAM + margin; SPM-first when matches are tied; rare-mineral Strategy-2 requires SPM co-support. |
| **Unmixing** | Freeze library endmembers; seed only SPM-supported families; use/diversity regularizers; report near-dead endmembers. |
| **Proof package** | `mineral_proof.json` + `fig7_mineral_proof.png` + report field `minerals_present`. |

### 3.3 What Scene 1 taught us (evidence-based)

From quantitative audits of Scene 1 products (not assumptions):

- **OLINDEX3** is the strongest spatially coherent SPM signal (high spatial autocorrelation; little variance explained by column striping).  
- At a fixed threshold of 0.02, more than half the scene can look “olivine-positive” — so **scene-adaptive** thresholds are required.  
- **Library SAM on the old CR cube was non-discriminative**: olivine-hot, bland, and other masks all matched with nearly the same tiny angles, because diagnostic wavelengths were NaN.  
- **Clays / carbonates / sulfates / ice** did not pass adaptive gates as scene-supported minerals in the audited run.  
- Pipeline verdict was often **WEAK** (low DBCV, high bland fraction) — expected for a spectrally muted FRT until diagnostics and gates were fixed.

### 3.4 Current deliverables (per scene)

Typical files in `mineral_output/`:

| File | Content |
|------|---------|
| `spm_map.npy`, `spm_gate_info.json` | Summary parameters + adaptive gates |
| `label_map.npy`, `cluster_names.json` | Units + hybrid mineral labels |
| `abundance_map.npy`, `endmember_names.json` | Soft unmixing |
| `fig1`–`fig7` | Classification, abundances, SPM, spectra, proof |
| `mineral_identification_report.json` | Full run summary + methodology |
| `mineral_proof.json` | Per-mineral present / not present with evidence |

---

## 4. What we are still trying to achieve

### 4.1 Near-term (same methodology)

1. **Prove presence per scene** — After the latest re-run, confirm that `minerals_present` (e.g. Olivine on Scene 1) is backed by non-empty evidence in `mineral_proof.json` (adaptive hotspot + CR depth and/or library).  
2. **Stable, reviewable maps** — Cluster maps that follow geologic units more than detector stripes.  
3. **Discriminative library matches** — With diagnostic bands restored, SAM top-1 should separate olivine from pyroxene/quartz when the mineral is real.  
4. **Run Scenes 2–5** with the same pipeline and compare which mineral groups appear.  
5. **External check when available** — Correlate our OLINDEX/HCP/LCP maps with CRISM summary products (MTRDR/TER) or published maps for the same observation ID.

### 4.2 Longer-term (still within the same scientific frame)

- Stronger pushbroom destreak if residual column structure remains after destripe.  
- Optional use of VNIR (S-sensor) where Fe mineralogy needs the &lt;1 µm region.  
- Tighter rare-mineral claims only when both SPM and library support exist.  
- Human-readable “mineral card” per scene (one page: present / absent / uncertain).

---

## 5. Issues we still face

| Issue | Why it matters | Status |
|-------|----------------|--------|
| **Spectrally bland FRTs** | Many pixels are dust/continuum; HDBSCAN correctly leaves them as noise → sparse maps look “empty” even when one mineral is real. | Expected; mitigated by SPM-active domain. |
| **Residual striping** | Pushbroom artifacts can still form vertical speckles in cluster maps. | Partially mitigated (destripe + stripe filter); may need stronger correction. |
| **SAM ambiguity on muted spectra** | Even with restored bands, top library candidates can be close; margin checks and SPM-first labeling are required. | Mitigated in labeling policy; must verify on new run. |
| **Unmixing collapse** | Softmax can still assign ~one endmember everywhere if contrast is low. | Regularizers + SPM-supported EMs; abundances are supporting, not primary proof. |
| **No published unit map for every FRT** | We cannot cite a paper that already mapped `FRT0000BF9E` mineral units for every scene. | Proof package + optional PDS summary products as external check. |
| **Compute / Windows** | Full FRT + HDBSCAN is heavy; file locks and numba issues appeared earlier. | Mostly mitigated (`safe_np_save`, generic HDBSCAN, fit caps). |
| **Verdict “WEAK”** | Clustering quality metrics (e.g. DBCV) stay low on bland scenes. | Does not by itself mean “no minerals”; use `mineral_proof` for presence. |

---

## 6. How to read results (practical)

When a run finishes, check in this order:

1. **`mineral_proof.json`** — Which mineral *groups* are `present: true`, and the evidence list.  
2. **`mineral_identification_report.json` → `minerals_present`** — Short list for the scene.  
3. **`fig7_mineral_proof.png`** — OLINDEX (or other) map, adaptive hotspot, clusters.  
4. **`cluster_names.json`** — Hybrid labels per cluster (provisional names).  
5. **`fig1_classification.png`** — Spatial pattern (expect sparse units + bland as black/noise).

**Do not** treat abundance maps alone as proof of a mineral. Prefer SPM + diagnostic CR + library evidence in the proof file.

---

## 7. How to run

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:PYTHONIOENCODING = "utf-8"
$env:CRISM_USE_UMAP = "0"
python run_mineral_fullfrt.py --scene 1
```

Outputs: `D:\CRISM scenes\scenes\Scene 1\mineral_output\`

---

## 8. One-paragraph summary

We are building an automatic CRISM mineral-identification pipeline that follows standard hyperspectral practice: bland-ratioed I/F, Viviano–Beck summary parameters, continuum removal, USGS CRISM-convolved library matching, contrastive encoding, density-based clustering (dynamic number of units), and constrained unmixing. The goal is not to maximize colorful pixels, but to state **which minerals are actually supported by the data**, with uncertainty and proof artifacts. Core methodology is fixed; recent work removed blockers (blanked diagnostic bands, bland-domain clustering, non-discriminative SAM, missing evidence reports) so that claims such as “olivine is present in Scene 1” can be demonstrated with measurable SPM, spectral, and library evidence rather than assumed from a single map color.

---

## 9. Related documents

| Document | Role |
|----------|------|
| `CRISM_MINERAL_IDENTIFICATION_GUIDE.md` | Scientific guide / intended method |
| `MINERAL_IDENTIFICATION_IMPLEMENTATION.md` | Implementation notes |
| `MINERAL_ID_FIXES.md` | Chronological fix log (Runs 1–4) |
| `Stage5_minerals.py` | Executable pipeline |

---

*End of report.*
