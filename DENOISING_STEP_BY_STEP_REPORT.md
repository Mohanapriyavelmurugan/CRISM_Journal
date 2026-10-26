# CRISM Denoising Pipeline: Step-by-Step Simple Report

This report explains how the CRISM denoising pipeline works in simple language.

The goal of denoising is to take the raw CRISM image cube and reduce unwanted noise while keeping the real mineral and surface information safe.

Think of the CRISM data as a stack of many images. Each image in the stack is taken at a different wavelength, like looking at Mars through many different color filters. A single pixel is not just one value; it has a full spectrum across many wavelengths.

---

## 1. Input Data

The pipeline starts from a folder such as:

```text
D:\NEW CROSS MISSION\Data\Scene 12\Crism
```

Inside that folder, it expects CRISM files such as:

```text
scene_name.img
scene_name.lbl
```

or sometimes:

```text
scene_name.img
scene_name.lbl.txt
```

The `.img` file contains the actual image cube values.

The `.lbl` file contains the instructions for reading the `.img` file, such as image size, number of bands, data type, and wavelength information.

The pipeline uses both files together.

---

## 2. Overall Pipeline Command

The command used for Scene 12 is:

```powershell
cd "D:\CRISM Journal\cv-crism"
$env:CRISM_DATA_DIR = "D:\NEW CROSS MISSION\Data\Scene 12\Crism"
python Run_pipeline.py --stages 1,2,3,4
```

This runs four stages:

```text
Stage 1: EDA / basic checking
Stage 2: Noise characterization
Stage 3: Physics-based correction
Stage 4: Machine-learning denoising
```

The final denoised output is saved in:

```text
D:\NEW CROSS MISSION\Data\Scene 12\Crism\ml_output
```

---

## 3. Stage 1: Basic Data Checking

Script used:

```text
Stage1_eda.py
```

Output folder:

```text
eda_output
```

### What This Stage Does

This stage opens the CRISM cube and studies it before doing any correction.

It checks each wavelength band and asks simple questions:

- Is this band valid?
- Is the signal too weak?
- Is the band unusually noisy?
- Is the band dead or almost constant?
- Are there invalid values?

### Simple Example

Imagine you have a book with hundreds of pages, but some pages are blank, some are smudged, and some are readable.

Stage 1 is like quickly checking every page and marking:

- readable pages
- damaged pages
- suspicious pages

It does not fix the data yet. It only studies and summarizes it.

### Important Files Created

```text
eda_output\eda_summary.json
eda_output\eda_summary.csv
eda_output\<scene_name>_band_stats.png
```

The `.json` and `.csv` files store the summary.

The `.png` plot shows band statistics visually.

---

## 4. Stage 2: Noise Characterization

Script used:

```text
Stage2_noise.py
```

Output folder:

```text
noise_output
```

### What This Stage Does

This stage identifies what kind of noise is present in the cube.

It does not fully remove the noise yet. It makes a noise map that tells later stages what needs to be corrected.

The main output is:

```text
noise_output\noise_map.json
```

This file is very important because Stage 3 and Stage 4 use it.

### Types of Noise Detected

#### N1: Stripe Noise

Stripe noise appears as vertical or column-like lines in the image.

Simple explanation:

Some detector columns may behave slightly differently from others. This creates repeated vertical striping.

The code detects this by checking whether column averages are unusually different from each other.

#### N2: Gaussian Random Noise

This is random grainy noise.

Simple explanation:

It is like static in an old TV image. The signal is still there, but it is mixed with random shaking or grain.

The code estimates signal-to-noise ratio for each band. Bands with poor SNR are marked as noisy.

#### N3: Spectral Spikes

These are sudden sharp jumps in one or a few bands.

Simple explanation:

Imagine a smooth line graph where one point suddenly shoots upward. That point is likely not real geology; it may be a detector artifact or cosmic-ray-like spike.

#### N4: Thermal Noise

This mostly affects longer wavelengths.

Simple explanation:

At longer wavelengths, heat from the surface or instrument can add extra signal that does not represent minerals.

#### N5: Saturation

Saturation happens when values are clipped at the top.

Simple explanation:

It is like taking a photo of something too bright. The bright region becomes flat white, and the real details are lost.

### Important Files Created

```text
noise_output\noise_map.json
noise_output\<scene_name>_noise.csv
noise_output\<scene_name>_snr.png
```

The most important file is:

```text
noise_map.json
```

It tells the next stages which bands have stripes, random noise, spikes, thermal problems, or saturation.

---

## 5. Stage 3: Physics-Based Correction

Script used:

```text
Stage3_physics.py
```

Output folder:

```text
physics_output
```

### What This Stage Does

This stage fixes problems that can be corrected using simple physical or mathematical rules.

It mainly handles:

- spikes
- thermal excess
- saturation
- illumination differences
- dust or slope-like effects

It does not mainly remove stripe noise or random Gaussian noise. Those are left for Stage 4.

### Step 3.1: Illumination Correction

Sometimes one row or region of the image looks brighter or darker because of lighting geometry.

Simple explanation:

Imagine photographing a page where one side is in shadow. The writing is still there, but brightness is uneven.

The pipeline uses row-median normalization to reduce this uneven lighting.

### Step 3.2: Spike Correction

If one pixel or band suddenly has an unrealistic sharp value, the pipeline compares it with nearby values.

If it looks like a spike, it replaces it using a median-filtered value.

Simple explanation:

If one number in a smooth list is clearly wrong, replace it with a value similar to its neighbors.

### Step 3.3: Saturation Correction

If a band is saturated, the value may be too high and clipped.

The pipeline tries to repair it using nearby spectral bands.

Simple explanation:

If one page in a book is damaged, estimate its content from the page before and after it.

### Step 3.4: Thermal Correction

For long wavelengths, heat can add unwanted signal.

The pipeline uses a Planck-style thermal model to estimate and subtract the heat contribution.

Simple explanation:

It tries to separate "light reflected from the surface" from "extra signal caused by heat."

### Step 3.5: Atmospheric / Dust Slope Correction

Dust or atmosphere can tilt the spectrum.

The pipeline applies an anchor-based correction to reduce broad slope effects.

Simple explanation:

If the whole spectrum is leaning upward or downward because of dust, this stage tries to straighten it.

### Important Files Created

```text
physics_output\<scene_name>_S3.npy
physics_output\stage3_log.json
```

The `_S3.npy` file is the Stage 3 corrected cube.

This is cleaner than the original input, but it may still have stripe noise and random noise.

---

## 6. Stage 4: Machine-Learning Denoising

Script used:

```text
Stage4_denoising.py
```

Output folder:

```text
ml_output
```

This is the main denoising stage.

It reads:

```text
physics_output\<scene_name>_S3.npy
noise_output\noise_map.json
```

It writes:

```text
ml_output\<scene_name>_S4.npy
ml_output\stage4_validation.json
```

The `_S4.npy` file is the final denoised cube.

---

## 7. Stage 4 Part A: Stripe Denoising

Noise type handled:

```text
N1 stripe noise
```

### What Is Stripe Noise?

Stripe noise appears as repeated vertical lines or column patterns.

Simple explanation:

Imagine scanning a document with a scanner where one scanner strip is slightly darker. Every image will show a faint vertical line.

### Step 1: Classical Destriping

The pipeline first uses a simple correction.

For each band:

1. It calculates the average value of each column.
2. It calculates the average value of the whole scene.
3. If a column is too bright or too dark compared with the scene, it adjusts that column.

Simple explanation:

If column 20 is always brighter than the rest, reduce column 20 slightly.

This removes the main stripe pattern.

### Step 2: Residual CNN Correction

After classical destriping, small leftover stripe patterns may remain.

The pipeline trains a small 1D CNN model on the same scene.

CNN means convolutional neural network.

Simple explanation:

The model learns what the leftover stripe pattern looks like and removes it.

This is self-supervised. That means the model does not need a separate clean image. It learns from the scene itself.

### How It Learns Without a Clean Image

The pipeline creates training examples from the scene:

- "clean-like" data comes from the classically destriped cube
- "noisy" data comes from the original noisy profile
- the model learns how to move from noisy column profiles to cleaner column profiles

Simple explanation:

It compares the striped version and the less-striped version of the same data, then learns the difference.

---

## 8. Stage 4 Part B: Gaussian Random Noise Denoising

Noise type handled:

```text
N2 Gaussian random noise
```

### What Is Gaussian Noise?

This is grainy, random noise spread across the image or spectrum.

Simple explanation:

It is like random static. It does not form neat stripes, but it makes the signal rough.

### Spectral DAE

The pipeline uses a spectral DAE.

DAE means denoising autoencoder.

This model works along the spectral direction. That means it studies the wavelength curve of each pixel.

### Simple Explanation

Each pixel has a spectrum, like a line graph across wavelengths.

Real mineral spectra are usually somewhat smooth and structured.

Random noise jumps around.

The model learns the smooth structure and removes the random jumping.

### Noise2Void / Blind-Spot Idea

The pipeline uses a blind-spot method.

During training, it hides some values from the spectrum and asks the model to guess them from neighboring wavelength values.

Simple example:

Suppose the real sequence is:

```text
10, 11, 12, 13, 14
```

If the model sees:

```text
10, 11, hidden, 13, 14
```

it should guess:

```text
12
```

Because the model is not allowed to see the hidden value, it cannot simply memorize the noise. It must learn the real shape of the spectrum.

This helps remove random noise while preserving real mineral absorption features.

---

## 9. Why the Pipeline Uses Two Denoising Methods

Stripe noise and random noise are different problems.

Stripe noise has a spatial pattern:

```text
vertical columns
```

Gaussian noise has a random spectral/spatial pattern:

```text
grainy variation
```

So the pipeline uses two different tools:

```text
Stripe noise     -> classical destriping + 1D CNN
Gaussian noise   -> spectral denoising autoencoder
```

This is better than using one generic filter for everything.

---

## 10. Validation After Denoising

After Stage 4, the pipeline checks whether denoising improved the cube safely.

It calculates:

- SNR before and after
- SAM spectral angle
- mineral band-depth preservation

### SNR

SNR means signal-to-noise ratio.

Higher SNR usually means the useful signal is clearer compared with noise.

### SAM

SAM means spectral angle mapper.

It checks whether the spectral shape changed too much.

Simple explanation:

If denoising makes the spectrum cleaner but changes its shape badly, that is dangerous. The mineral information could be distorted.

### Band-Depth Preservation

Mineral detection often depends on absorption dips at certain wavelengths.

The pipeline checks important mineral features such as:

```text
BD2300
BD1900
OLINDEX3
BD1500
```

Simple explanation:

If a real mineral dip existed before denoising, the pipeline tries to make sure it is still there after denoising.

---

## 11. Safety Behavior

The pipeline has safety checks.

If denoising looks good, it saves the denoised result.

If denoising is questionable, it blends the denoised cube with the Stage 3 physics cube.

If denoising fails validation, it rejects the ML output and keeps the Stage 3 physics output instead.

Simple explanation:

The pipeline does not blindly trust the machine-learning result. It checks whether the output still makes scientific sense.

---

## 12. Final Outputs

After running:

```powershell
python Run_pipeline.py --stages 1,2,3,4
```

you should get:

```text
eda_output
noise_output
physics_output
ml_output
```

The most important final file is:

```text
ml_output\<scene_name>_S4.npy
```

This is the final denoised cube.

The validation report is:

```text
ml_output\stage4_validation.json
```

This tells whether denoising passed, needed review, or failed.

---

## 13. Short Summary

In simple words:

1. The pipeline reads the CRISM `.img` and `.lbl` files.
2. It checks which bands are usable.
3. It identifies different kinds of noise.
4. It fixes physical problems like spikes, saturation, thermal signal, and illumination.
5. It removes stripe noise using column correction and a small CNN.
6. It removes random spectral noise using a denoising autoencoder.
7. It checks that mineral signatures were not damaged.
8. It saves the final denoised cube as `_S4.npy`.

The final result is a cleaner CRISM cube that should preserve the real spectral information needed for mineral analysis.

