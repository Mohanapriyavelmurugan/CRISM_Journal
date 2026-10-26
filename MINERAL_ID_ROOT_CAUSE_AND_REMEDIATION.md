# CRISM Mineral Identification — Root-Cause Audit & Remediation Guideline

**Scope:** `Stage5_minerals.py`, `run_mineral_fullfrt.py`, `Stage4_denoising.py`, `Stage3_physics.py`, `crism_utils.py`
**Audited scene:** Scene 1 — `FRT0000BF9E_07_IF165L_TRR3` (450 × 640 × 438)
**Audited run:** `mineral_output/` products dated 5 Oct 2026
**Date:** 5 October 2026

> **Bottom line.** The scientific methodology is sound and does not need to change. The
> implementation is reading the data cube on a **wrong wavelength axis** — the band order
> is reversed relative to what `crism_utils.WAVES_L` assumes. Every Viviano–Beck index,
> every band mask, every library match and every atmospheric correction in Stages 3–5 is
> therefore evaluated at the wrong wavelength. This single defect explains essentially all
> of the observed symptoms, including the result that all ten mineral groups were reported
> `PRESENT` in one FRT. Separately, the denoised S4 cube is **not currently fit** for
> mineral identification and is in fact **not being used** for it.

---

## 1. What was verified (no assumptions)

Every claim below is backed by a measurement on the actual files, not by reading comments.
Audit scripts written for this review, all read-only with respect to the pipeline:

| Script | What it establishes |
|---|---|
| `audit_raw_read.py` | The `.img` reader is correct; cube bytes match an independent BIL read exactly |
| `audit_spectrum_shape.py` | The scene median spectrum is non-physical on the current axis |
| `audit_wavelength_order.py` | The band→wavelength mapping is reversed; recovers dispersion from CO₂ anchors |
| `audit_band_mapping_damage.py` | Independent confirmation + per-index wavelength error table |
| `audit_denoised_for_mineralid.py` | S4 fitness: NaN coverage, noise reduction, bias, stripe suppression |
| `audit_plot_wavelength_proof.py` | Figure `mineral_output/fig_wavelength_axis_proof.png` |

---

## 2. Root cause — the wavelength axis is reversed

### 2.1 Where it comes from

The TRR3 label carries **no `BAND_BIN_CENTER`**. CRISM TRR3 stores wavelengths in a
separate calibration product, named in the label:

```
MRO:WAVELENGTH_FILE_NAME = "CDR410803692813_WA0000000L_3.IMG"
```

That CDR is not present in the scene directory. `parse_pds3_label()` therefore returns
`wavelengths = None`, and `get_sensor_waves()` silently falls back to a hard-coded table:

```45:60:cv-crism/crism_utils.py
    # Anchor points: (band_index_0based, wavelength_um)
    anchors = [
        (0,   1.0285), (55,  1.1805), (107, 1.3330),
        (160, 1.5065), (213, 1.6935), (260, 1.8520),
        (307, 2.0150), (354, 2.1860), (380, 2.2700),
        (407, 2.4950), (437, 3.9200),
    ]
```

This table is **piecewise-linear interpolation between eleven hand-guessed anchors**. It is
ascending, and it implies a dispersion that varies from 2.8 nm/band to 47.5 nm/band. Real
CRISM L dispersion is uniform at ≈6.55 nm/band — independently confirmed from the USGS
CRISM-convolved library on disk (`s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt`:
413 L channels, 1.0036–3.8968 µm, median spacing 6.610 nm). So the fallback table is wrong
in both **direction** and **dispersion**.

### 2.2 The evidence that the order is reversed

**(a) The median spectrum is impossible as plotted, and textbook-correct when reversed.**

![wavelength axis proof](D:\CRISM%20scenes\scenes\Scene%201\mineral_output\fig_wavelength_axis_proof.png)

On the current axis, median I/F falls from 0.174 at "1.15 µm" to **0.005 at "1.55 µm"** — a
factor of 35 — then jumps straight back to 0.154. No Mars surface or atmospheric process
does that. Reversed, the same data is a standard CRISM L spectrum: flat ≈0.195 from
1.0–1.9 µm, the sharp 2.01 µm CO₂ doublet, a 2.1–2.6 µm plateau, the saturated 2.7 µm CO₂
band, the dark 2.8–3.2 µm region, and the thermal upturn toward 3.9 µm.

**(b) A two-point fit on CO₂ recovers the correct detector endpoints by itself.**

Anchoring only the two deepest minima to the known CO₂ band centres (2.70 µm and 2.01 µm)
yields dispersion **−6.765 nm/band**, band 0 = **3.931 µm**, band 437 = **0.975 µm**. The
magnitude matches the true CRISM L dispersion (6.55–6.61 nm), and the endpoints match the
L detector range (≈1.0–3.92 µm) — neither of which was an input to the fit.

**(c) Every instrument-flagged invalid band lands on a physically expected location — but
only under the reversed mapping.**

| Invalid bands (<1% finite) | Reversed axis says | Current axis says |
|---|---|---|
| 0–3 | 3.911–3.931 µm → above long-λ cut-off | 1.028–1.037 µm |
| 173–178 | 2.727–2.761 µm → **saturated 2.70 µm CO₂ core** | 1.552–1.570 µm |
| 436–437 | 0.975–0.982 µm → below the 1.0 µm cut-on | 3.872–3.920 µm |

Under the reversed mapping all three runs are exactly where CRISM data *must* be invalid.
Under the current mapping all three are at wavelengths with no reason to be blank.

This also resolves a long-standing mystery in the code. The "**1.55 µm artifact notch**"
that `suppress_cr_artifact_notch()` (Fix E) was written to interpolate across **is the
2.70 µm CO₂ band**. There is no 1.55 µm artifact.

**(d) Secondary confirmations.** Bands 130–160 → 3.05–2.85 µm with I/F 0.025→0.017 (the real
3 µm hydration/low-reflectance region). Bands 50→4 → 3.59–3.90 µm with I/F rising
0.163→0.178 (the thermal upturn). The label's own
`MRO:IKF_START_CHANNEL = (0, 179)` filter-region split sits precisely at the saturated CO₂
core, which is where a calibration pipeline would naturally split.

### 2.3 What each index has actually been measuring

Using the recovered axis, here is the wavelength each Viviano–Beck index currently samples:

| Index | Intended centre | Band used | **Actually sampled** | Error |
|---|---|---|---|---|
| OLINDEX3 | 1.330 (left 1.054, right 1.750) | 106 / 16 / 230 | **3.214 / 3.823 / 2.375** | up to **+2.77 µm** |
| LCPINDEX2 | 1.810 | 247 | **2.260** | +0.45 |
| HCPINDEX2 | 2.120 | 336 | **1.658** | −0.46 |
| BD1900_2 | 1.930 | 289 | **1.976** (inside the 2.01 µm CO₂ complex) | +0.05 |
| BD2100_2 | 2.132 | 339 | **1.638** | −0.49 |
| BD2210_2 | 2.210 | 377 | **1.381** | −0.83 |
| BD2290 | 2.290 | 382 | **1.347** | −0.94 |
| BD2300 | 2.295 | 383 | **1.340** | −0.96 |
| BD2500_2 | 2.500 | 407 | **1.178** | −1.32 |
| SINDEX2 | 2.100 | 330 | **1.699** | −0.40 |

Not one index samples its intended band. Two consequences are worth naming explicitly
because they match the symptoms recorded in `MINERAL_ID_STATUS_REPORT.md`:

- **"OLINDEX3 is the strongest, most spatially coherent signal, and >50% of the scene is
  olivine-positive."** It is computed from bands at **3.82, 3.21 and 2.38 µm** — i.e. it is
  measuring the 3 µm hydration band against the thermal tail. That is a strong, broad,
  spatially coherent signal over essentially all of Mars. The "olivine map" is a
  dust-hydration/thermal-slope map.
- **"BD1900 hydration passes everywhere."** Its centre band falls in the **2.01 µm
  atmospheric CO₂** complex. The "hydration" detection is atmospheric CO₂.

### 2.4 It also invalidates the library match

`stage1_library()` places each USGS spectrum onto the scene band grid by nearest wavelength
using the same bad `waves` array:

```1592:1600:cv-crism/Stage5_minerals.py
    lib_to_scene_map: List[Optional[int]] = []
    for sw in waves:
        closest = int(np.argmin(np.abs(lib_waves_489 - sw)))
```

So library array index *b* holds reflectance at `w_pipe[b]` (ascending) while scene index *b*
holds data at `w_rev[b]` (descending). The spectral angle is being computed between a
library spectrum and an approximately **mirror-imaged** scene spectrum. This is the real
reason SAM was "non-discriminative": in the audited run, clusters 0, 1 and 2 all matched
within 0.0015–0.0022 rad of their second-best candidate, with Siderite, Olivine, Diopside
and Quartz separated by under 0.003 rad. All three clusters fell back to
`label_basis: "spm_first_tied_sam"`, so the library contributed nothing to any label.

### 2.5 Stages 3 and 4 are affected too

`Stage3_physics.py` calls `get_sensor_waves` / `build_bad_mask` and places its atmospheric
anchor windows by wavelength (lines 334–364), so the "multiplicative anchor" correction
anchored on the wrong bands. `Stage4_denoising.py` uses `build_bad_mask(waves)` to decide
which bands to denoise. So **S3 and S4 are both built on the wrong axis**, and their
reported gains (S3 +6.3% SNR, S4 +33% SNR, "OLINDEX3 preserved to 0.7%") are not evidence
of anything mineralogical.

---

## 3. Is the denoised cube ready for mineral identification?

**No — and it is also not currently being used.** Three separate findings.

### 3.1 It is not used

Despite `cube_source` being S4, the continuum-removal cube — which feeds the encoder,
HDBSCAN, library SAM, unmixing and the proof package — is built from the **raw TRR3 `.img`**:

```1851:1854:cv-crism/Stage5_minerals.py
    # Fix A: continuum removal on bland-ratioed IMG (same cube as SPM)
    # so 1.05 / 1.9 / 2.2 µm diagnostics remain finite for encoder + library SAM.
    cube_for_cr = cube_spm
    print("  CR source: bland-ratioed TRR3 IMG (diagnostic bands restored)")
```

The report confirms it: `"cr_source": "bland_ratioed_IMG_diagnostic_bands"`. The S4-derived
`cube_ratio` is referenced in exactly two places (lines 1786 and 1797), both inside a mafic
fallback that only triggers when an SPM index is unusable. In the audited run every index
was `ok`, so **the fallback never fired and S4 contributed no pixel values to any mineral
product.** Stages 1–4 effectively ran for nothing.

### 3.2 It cannot be used as-is: 134 bands are all-NaN

Stage 4 skips every band in `build_bad_mask`, and those bands end up NaN in S4. Measured on
Scene 1 (percent of valid pixels finite):

| Pipeline label | Band | finite in IMG | **finite in S4** |
|---|---|---|---|
| OLINDEX3 RB1054 | 9 | 100.0% | **0.0%** |
| LCPINDEX2 centre | 248 | 100.0% | **0.0%** |
| BD1900 centre | 282 | 100.0% | **0.0%** |
| BD2210 centre | 361 | 100.0% | **0.0%** |

Overall S4 is finite on 67.3% of channels versus 96.4% for the raw IMG; 138 of 438 bands are
bit-identical to S3 (untouched) and 134 are NaN. This is precisely why someone previously
routed CR back to the raw IMG — a reasonable workaround for a real blocker, but it meant
the denoiser became decorative. **Note that after the wavelength fix the mask moves**: the
currently-blanked ranges (bands 248–288, 342–376) are truly 2.03–2.29 µm and 1.38–1.63 µm,
i.e. Stage 4 has been discarding real 2.1/2.2/2.3 µm mineral territory while leaving the
genuine CO₂ cores in.

### 3.3 Two quantitative concerns about the denoising itself

**Over-aggressive destriping.** Column-mean structure (after removing a smooth cubic
cross-track trend, so genuine illumination gradients are not counted) collapses by roughly
three orders of magnitude:

| Band | stripe metric IMG | stripe metric S4 |
|---|---|---|
| 1.210 (pipeline label) | 3.85e-03 | **2.58e-06** |
| 1.330 | 1.96e-03 | **1.46e-06** |
| 2.290 | 5.18e-03 | **7.87e-07** |

`classical_destripe()` subtracts `col_mean − scene_mean` for **every** column of **every**
good band, which forces all 640 column means to be identical. Any geologic unit that is
spatially extended in the along-track direction has its contrast removed along with the
stripe. Stage 2 identified only **7** stripe-affected bands, yet the correction is applied to
all ~304 good bands. `Stage5_minerals.column_destripe()` then applies a second, similar
correction on top.

**Wavelength-dependent multiplicative bias.** S4 is systematically darker than the IMG, and
not uniformly so: −8.8% at band 65, −11.9% at band 106, −6.8% at band 230, −5.9% at band
336. A band depth is a ratio, so a differential gain error of this size directly changes
every index. Meanwhile the actual noise suppression is modest — spectral roughness only
falls ~15% (5.04e-04 → 4.27e-04 at band 65) and spatial roughness ~10%. The headline "+33%
SNR" is dominated by destriping, not by denoising.

---

## 4. Issues independent of the wavelength bug

These will still be wrong after the axis is fixed, so they need separate attention.

### I-1. Scene CR and library CR use different algorithms (breaks SAM)

Library spectra use a global convex upper hull:

```376:380:cv-crism/Stage5_minerals.py
def continuum_removal(spectrum: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """
    Modified upper convex hull continuum removal.
    Traces the upper envelope by steepest-slope walk from left to right.
    """
```

The scene uses a 21-band moving maximum whenever there are >20 000 valid pixels, which is
always true for a full FRT:

```1878:1881:cv-crism/Stage5_minerals.py
            env = maximum_filter1d(np.nan_to_num(flat, nan=0.0), size=21, axis=1, mode="nearest")
            env = np.maximum(env, np.nan_to_num(flat, nan=0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                cr = flat / np.maximum(env, 1e-8)
```

A 21-channel moving maximum tracks the spectrum closely, so it normalises away broad
absorptions and drives CR toward 1.0 with only narrow residual dips. Comparing that against
hull-normalised library spectra is comparing two different quantities. **This alone makes
every spectral angle in the pipeline uninterpretable**, and it is a sufficient explanation
for the clustering of all candidates within 0.003 rad.

### I-2. The detection gate is tautological — it cannot report absence

```530:532:cv-crism/Stage5_minerals.py
        p_adapt = float(np.percentile(vals, percentile))
        thr_abs = float(SPM_DETECT_THRESH[name])
        thr = max(thr_abs, p_adapt)
```

With `percentile = 90`, whenever the scene p90 exceeds the absolute threshold the gate
admits **exactly the top 10% of pixels by construction**, and `SPM_MIN_COMPONENT = 50` is
then trivially satisfied. In the audited run `n_raw = 26 388` for BD1900_2, OLINDEX3 and
LCPINDEX2 — identical, because 26 388 is precisely 10% of the finite pixels. A percentile
gate has no null hypothesis and can never return "absent".

The presence rule compounds this:

```3780:3783:cv-crism/Stage5_minerals.py
            entry["present"] = (
                n_hot >= SPM_MIN_COMPONENT and len(supporting) >= 1
            )
```

`supporting` includes the string `overlaps_clusters=[...]`, and clusters are *defined* on the
SPM-active domain, which is the union of the hotspots. So overlap is guaranteed. The result
is circular: hotspot exists (guaranteed) **and** overlaps a cluster (guaranteed) ⇒ present.
Hence **all ten mineral groups reported `PRESENT` for one FRT**. Seven of the ten
(Monohydrated Sulfates, Fe/Mg-OH Phyllosilicates, Fe/Mg Phyllosilicates, Carbonates,
Low-Ca Pyroxene, High-Ca Pyroxene, Sulfates) carry **no spectral evidence at all** — their
entire evidence list is the hotspot plus the cluster overlap. Only Hydrated Minerals,
Al-OH Clays and Olivine have even one spectral term, and those terms are themselves
measured at the wrong wavelengths.

### I-3. No significance test; thresholds below the noise floor

CR depth thresholds of 0.005 are used as evidence (`d2.21 = 0.0051` was accepted as proof of
Al-OH clays). With scene SNR ~11–15 and uncorrected atmospheric and column residuals, a
0.005 band depth is not distinguishable from systematic residual. `depths_bland` is computed
for a bland reference but is only compared against for OLINDEX3; every `BD*` index skips the
hot-versus-bland contrast entirely.

### I-4. SPM formulas are simplified, not the published Viviano–Beck forms

`SPM_INDICES` defines every index as a single 3-point continuum-interpolated band depth.
The published Viviano–Beck (2014) forms are multi-channel weighted sums for OLINDEX3,
LCPINDEX2, HCPINDEX2 and BD2300; BD1900_2 and BD2210_2 average two centres; and SINDEX2 is
a *convexity* measure rather than a band depth. The implemented OLINDEX3,

```
(0.1·R1054 + 0.1·R1211 + 0.4·R1329 + 0.4·R1474) / R1750 − 1
```

is a ratio of reflectances, whereas published OLINDEX3 is a weighted sum of *relative band
depths* against a continuum anchored at longer wavelengths. Every formula and every entry in
`SPM_DETECT_THRESH` (currently annotated "guide", not sourced) should be checked line by line
against Table 2 of the paper. I have deliberately not written specific coefficients here
because they must come from the paper, not from recollection.

### I-5. Ratio reference is scene-wide, not column-local

`estimate_bland_spectrum()` returns a single spectrum for the whole scene. Standard CRISM
practice ratios each spectrum to a bland reference drawn from the **same detector columns**,
because the dominant residuals are column-dependent. A scene-wide denominator cannot cancel
them, which is part of why residual striping keeps reappearing downstream.

### I-6. Continuum removal applied to an already-ratioed cube

`cube_cr` is the hull/envelope normalisation of `cube_spm`, which is itself `I/F ÷ bland`.
A ratio spectrum is already continuum-suppressed; normalising it again and then comparing to
hull-normalised library **reflectance** mixes physical quantities. Pick one convention —
either ratio-to-bland for detection and hull-CR of *unratioed* I/F for library matching, or
hull-CR on both sides — and apply it consistently.

### I-7. Clustering quality is not being treated as a blocker

`dbcv_score = −0.192` (negative means the partition is worse than no structure),
233 108 of 288 000 pixels are noise (81%), and only 3 clusters emerge. The run is labelled
`overall_verdict: "WEAK"` but still emits ten confident mineral claims. Cluster 0 is named
"Low-Ca Pyroxene" while its best library match is Siderite.

### I-8. Minor but worth fixing

- `MARS_PRIOR_MINERALS` includes 50 names and `prior_only=True` restricts SAM to them; this
  is a defensible Mars prior but it should be recorded in the report as a constraint, since
  it makes "best match" conditional on the prior.
- The ROWNUM table (438 selected detector rows, 0–445, monotonic) is parsed by none of the
  stages. It is the authoritative statement of which detector rows are present and should be
  used to index into the wavelength CDR.
- `np.random` is used in `stage2` without a local seed, so training-pixel selection is not
  reproducible across runs even though `_set_seed` exists.

---

## 5. Remediation plan (ordered — do not reorder)

The ordering matters: fixing anything downstream before the axis is pointless, because every
downstream mask, index and match is keyed on wavelength.

### Step 0 — Freeze and quarantine the current results

Move `Scene 1/mineral_output/` to `mineral_output_PREFIX_WRONG_AXIS/`. Mark the ten
`minerals_present` entries and `MINERAL_ID_STATUS_REPORT.md` §3.3 as retracted. Do not reuse
`cube_cr.npy`, `spm_map.npy`, `lib_spectra_438_cr.npy` or `latent_vectors.npy`.

### Step 1 — Establish an authoritative wavelength table

This is the only fix that must be right; everything else depends on it.

**Chosen approach — anchor on the USGS table already on disk.** It needs no download, and a
feasibility check confirms it is accurate enough:

`s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt` contains 413 L channels spanning
1.00364–3.89676 µm. Those 413 values lie on a **uniform 439-channel grid**,
`1.00364 + 0.0066101·ch`, with max deviation 3.3 nm and rms 1.5 nm. The two gaps in the USGS
set are exactly the channels CRISM drops: 1.0102→1.0472 µm (37 nm ≈ 5 channels, near the
cut-on) and 2.6549→2.8004 µm (145 nm ≈ 22 channels, the saturated CO₂ core). 413 + 5 + 22 ≈
439, and our cube has 438 bands — so the TRR3 L set *is* this grid.

Procedure:

1. Take the dispersion and zero point from the USGS table: step = **6.6101 nm**, reference
   **1.00364 µm**.
2. Build the axis **descending** in band index: `w[band] = w_437 + 0.0066101·(437 − band)`.
3. Solve the single remaining unknown (the integer band offset `w_437`) by least squares
   against the hard physical anchors measured in this audit: the saturated-CO₂ invalid run
   (bands 173–178), the cut-on invalid run (bands 436–437), the cut-off invalid run
   (bands 0–3), and the 2.01 µm CO₂ minimum (band ≈ 284). These are sharp, so the offset
   resolves to a single channel.
4. Assert monotonic, span ≈1.00–3.92 µm, then **cache to disk** beside the scene so no run
   can ever silently fall back again.

**Do not ship the quick 2-point empirical fit used in this audit.** `w = 3.9312 −
0.006765·band` gives band 437 = 0.975 µm, whereas the USGS-implied grid gives ≈1.0036 µm —
a disagreement of about **6 channels (~40 nm)**. That is fine for identifying the reversal,
which is all it was used for, but it is too coarse for band-depth work. Quoted wavelengths
elsewhere in this document carry that ~40 nm caveat; the *reversal itself* is independent of
it and is not in doubt.

**Later upgrade (optional, not blocking).** The authoritative product is the WA CDR named in
the label, `CDR410803692813_WA0000000L_3.IMG`, from the PDS Geosciences Node. It gives a
wavelength per *(column, band)* pair, since CRISM wavelengths drift slightly across the
640-column detector. Adopt it if cross-track consistency of narrow features (e.g. 2.21 vs
2.23 µm clay discrimination) becomes the limiting factor. Index it using the **ROWNUM table**
already present in the `.img` (438 entries, values 0–445, monotonic).

**Then make the failure loud, not silent:**

```python
# crism_utils.get_sensor_waves
if label_waves is None:
    raise ValueError(
        "No wavelength table for this cube. Supply the WA CDR or a cached axis; "
        "refusing to guess (see MINERAL_ID_ROOT_CAUSE_AND_REMEDIATION.md §5 Step 1)."
    )
```

Delete `_build_L_wavelengths()`. A fabricated axis must never be reachable.

### Step 2 — Add a wavelength assertion gate to every stage

Before any science runs, assert against physics that cannot be coincidental:

- the deepest median absorption is at 2.70 ± 0.03 µm;
- a local minimum exists at 2.01 ± 0.02 µm;
- the all-NaN runs fall at λ < 1.01 µm, λ > 3.90 µm, and 2.72–2.77 µm;
- median I/F is monotonically rising between 3.5 and 3.9 µm.

Fail the run if any check fails. This is cheap and would have caught the defect immediately.

### Step 3 — Rebuild all wavelength-keyed constants

With the true axis, regenerate `BAD_BAND_REGIONS_L`, `SPM_HARD_EXCLUDE`,
`MINERAL_CR_EXCLUDE` and `SAM_EXCLUDE_WINDOWS`. Specifically:

- **Delete the "1.55 µm artifact" machinery** (`suppress_cr_artifact_notch`,
  `SAM_EXCLUDE_WINDOWS = [(1.450, 1.650)]`). There is no such artifact.
- **Mask the real atmospheric cores**: ≈1.95–2.07 µm (CO₂) and ≈2.70–2.78 µm (saturated CO₂).
- **Keep the mineral diagnostics unmasked**: 1.0–1.9, 2.1–2.6, and 2.8–3.0 µm.
- Re-derive the EDA dead-band list, which was also computed on the wrong axis.

### Step 4 — Re-run Stages 1–4 and fix the denoiser before using S4

1. Re-run Stage 1 (EDA), Stage 2 (noise), Stage 3 (physics) and Stage 4 on the correct axis.
2. **Restrict destriping to the bands that are actually striped.** Stage 2 found 7. Apply
   `classical_destripe` only to those, and remove the second pass in
   `Stage5_minerals.column_destripe` so the correction is applied once, in one place.
3. **Replace mean-equalisation with a high-pass column correction.** Estimate the column
   offset from the residual after removing a smooth (e.g. order-3) cross-track polynomial, so
   genuine broad cross-track geology survives. Cap the correction and log how much signal
   was removed per band.
4. **Preserve diagnostic bands through Stage 4.** Denoise them rather than skipping them, or
   at minimum pass them through unchanged instead of writing NaN. The acceptance criterion is
   that S4 finite coverage matches the IMG to within the genuinely invalid bands only
   (i.e. ~96%, not 67%).
5. **Constrain the gain.** Add a check that the per-band ratio of S4 to IMG medians is flat
   to within ~1% across good bands, so band depths are not distorted. The current −6% to
   −12% wavelength-dependent darkening fails this.
6. Re-validate band-depth preservation using the **corrected** index definitions.

Only when (4) and (5) pass should `cube_for_cr` be switched to the S4-derived
`cube_ratio`. Keep the raw IMG as the comparison baseline, and report both.

### Step 5 — Make continuum removal consistent

Use **one** continuum algorithm for scene and library. Recommendation: the global convex
upper hull (`continuum_removal`) on both sides, since it is the established convention and
is already correct for the library. If full-FRT speed is the obstacle, vectorise the hull
(monotone-chain over the spectral axis, batched across pixels) rather than substituting a
moving maximum. Add a unit test asserting that a library spectrum pushed through the *scene*
CR path matches the same spectrum through the *library* CR path to within a small tolerance —
that test fails today and is the cheapest possible guard against this class of bug.

Also resolve I-6: decide explicitly whether library matching operates on hull-CR of unratioed
I/F or on hull-CR of ratioed spectra, document the choice, and make both sides match.

### Step 6 — Replace the detection gate with a real statistical test

Keep the methodology — SPM hotspot, diagnostic depth, library support — but make each leg
falsifiable.

1. **Absolute thresholds from the paper**, not scene percentiles. The percentile may be used
   to *rank* candidates, never to decide presence.
2. **Null distribution from a bland reference population.** Detection requires the hot-region
   band depth to exceed the bland-region depth by at least *k*·σ, where σ is estimated from
   the spatial scatter of the index over bland pixels. Report the effect size and *k*.
3. **Noise-floor gate.** Estimate per-band uncertainty from the data (e.g. along-track second
   differences, as used in the audit scripts) and propagate it into a band-depth uncertainty.
   Reject any claimed depth below ~3σ. This retires the 0.005 thresholds.
4. **Require real spectral evidence.** Remove `overlaps_clusters` from `supporting` — it is
   circular. Presence should require the SPM hotspot **and** a significant diagnostic
   absorption at the correct wavelength **and** a library match whose margin over the
   next-best *different mineral family* exceeds `SAM_MARGIN_STRONG`.
5. **Spatial coherence beyond component size.** Require the hotspot to be spatially clustered
   relative to a permutation null (e.g. Moran's I on the index map), which rejects stripe-like
   and salt-and-pepper false positives.
6. **Allow and expect "absent".** A healthy FRT result is 1–3 mineral groups present and the
   rest explicitly absent with a reported upper limit on band depth. Ten of ten present is
   the signature of a broken gate.

### Step 7 — Correct the index definitions

Transcribe OLINDEX3, LCPINDEX2, HCPINDEX2, BD2300, BD1900_2, BD2210_2 and SINDEX2 from
Viviano–Beck (2014) Table 2 verbatim, including the multi-channel weights, the correct
continuum anchors, and the sign convention for SINDEX2. Add a regression test that computes
each index on a USGS library spectrum of the mineral it is designed to detect and asserts a
positive response, plus a negative control on a spectrally bland spectrum. Cite the paper's
threshold values in `SPM_DETECT_THRESH` with a source comment.

### Step 8 — Fix the ratio reference

Replace the single scene-wide bland spectrum with a **column-local** denominator: for each
detector column (or a small window of columns), build a bland reference from the blandest
pixels in that same column window. This is standard CRISM practice and removes
column-dependent residuals that a global denominator cannot.

### Step 9 — Gate the report on clustering quality

Treat `dbcv_score < 0` or noise fraction > 70% as a hard failure for cluster-derived labels.
Under that condition, still emit the SPM/CR/library evidence, but suppress
`cluster_names.json`-derived mineral names and mark them unavailable rather than printing a
name whose own library match disagrees with it.

### Step 10 — External validation

Once Scene 1 produces a short, defensible list, compare the OLINDEX3 / LCPINDEX2 /
HCPINDEX2 / BD2300 maps against the CRISM MTRDR summary products for `FRT0000BF9E` from the
PDS. That is the decisive independent check, and it is available for this observation.

---

## 6. Acceptance tests

Do not declare any of this fixed until all of the following pass on Scene 1.

| # | Test | Pass criterion |
|---|---|---|
| 1 | Wavelength axis sourced | Axis built from the USGS 6.6101 nm grid, descending, offset solved against the physical anchors; cached to disk; `_build_L_wavelengths()` deleted |
| 2 | Physics assertions | Deepest absorption 2.70 ± 0.03 µm; local min 2.01 ± 0.02 µm; invalid runs at the three expected locations |
| 3 | Median spectrum | Matches the lower panel of `fig_wavelength_axis_proof.png` qualitatively |
| 4 | CR consistency | Library spectrum through scene-CR path ≈ library-CR path |
| 5 | SAM discrimination | On a synthetic cube built from known library endmembers, top-1 recovers the correct mineral with margin > `SAM_MARGIN_STRONG` |
| 6 | Index sanity | Each index responds positively on its target library mineral and ~0 on a bland spectrum |
| 7 | S4 coverage | Finite-band coverage within ~1% of the raw IMG outside genuinely invalid bands |
| 8 | S4 radiometry | Per-band S4/IMG median ratio flat to ~1% across good bands |
| 9 | Destripe restraint | Column correction applied only to Stage-2-identified striped bands; removed signal logged |
| 10 | Gate can say no | On a deliberately bland synthetic scene, zero minerals reported present |
| 11 | Scene 1 result | A short list (expect 1–3 groups) with σ-backed evidence, remainder explicitly absent with upper limits |
| 12 | External agreement | Spatial correlation with PDS MTRDR summary products for `FRT0000BF9E` reported |

---

## 7. What is *not* wrong

Worth stating, because the remediation should not throw this away:

- The **methodology** is right: bland-region ratio → Viviano–Beck summary parameters →
  continuum removal → USGS CRISM-convolved library matching → contrastive encoding →
  density-based clustering with free *K* → ASC-constrained unmixing, with provisional
  labelling and an evidence package. That chain is standard, defensible practice.
- The **`.img` reader is correct** — it matches an independent BIL read bit for bit, including
  the ROWNUM suffix handling.
- The **library handling is correct in principle**: the USGS `cvCRISMjMTR3` set is already
  CRISM-convolved and the code correctly refrains from applying an SRF.
- The **engineering scaffolding** is sound: full-FRT support, `safe_np_save`, the HDBSCAN
  backend fallbacks, the Windows/numba workarounds, the figure and report generation.
- The **instinct behind "Fix A"** — that diagnostic bands were being blanked — was correct.
  The diagnosis was just applied to the wrong wavelengths.

The work to date built a usable framework. What it needs is a correct wavelength axis, a
consistent continuum convention, and a detection gate that is capable of returning "no".

---

## 8. Decisions and remaining questions

**Decided.** The wavelength axis will be rebuilt from the USGS table already on disk
(§5 Step 1): descending, 6.6101 nm/band, integer offset solved against the saturated-CO₂ and
detector cut-on/cut-off anchors. No download required. The WA CDR remains an optional later
upgrade for per-column precision.

**Status.** This document is the deliverable; no pipeline code has been modified. The only
files added are the read-only audit scripts listed in §1 and the proof figure.

**Still open.**

1. **Scope of the re-run** — Scene 1 alone first as a validation case, or all five scenes
   once Steps 1–3 are in?
2. **Per-column axis** — worth adopting only if narrow-feature discrimination turns out to be
   the limiting factor; defer until Step 10 (external comparison against MTRDR) shows whether
   it matters.
3. **Index transcription (Step 7)** — this needs Viviano–Beck (2014) Table 2 in hand.
   Coefficients should be transcribed from the paper rather than reconstructed, so someone
   with the PDF needs to do or check that step.
