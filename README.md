# Robust Fetal ECG Detection from Abdominal Recordings

## Overview

This repository provides a custom implementation of the sequential analysis method detailed by Martens et al. (2007) for non-invasive fetal electrocardiogram (fECG) extraction. The primary challenge this project addresses is isolating the weak fetal cardiac signal from maternal abdominal recordings, which are heavily contaminated by maternal ECG (mECG), power-line interference, and baseline drift.

## Clinical Context

Fetal cardiac monitoring is essential for evaluating fetal health and detecting distress. While Doppler ultrasound is widely used, non-invasive abdominal fECG provides several unique benefits:
- **Continuous and ambulatory monitoring** without frequent probe repositioning.
- **Detailed morphological analysis** of cardiac cycles (P, QRS, and T waves) enabling deeper physiological insights.
- **Accurate Fetal Heart Rate (FHR)** variability assessment.

## Pipeline Architecture

This project explicitly avoids "black-box" approaches, implementing a strictly controlled, 5-stage sequential filtering pipeline based on established signal processing techniques:

```
Raw Abdominal Signal (1000 Hz)
    ↓
1. Baseline Wander Removal
    ↓
2. Power-Line Interference Cancellation
    ↓
3. Anti-Aliased Upsampling (2000 Hz)
    ↓
4. Maternal ECG Cancellation (MECG)
    ↓
5. Fetal ECG Extraction & Synchronous Averaging
```

### 1. Baseline Wander Remover

**Challenge:** Patient movement and respiration induce low-frequency baseline drift that corrupts the fECG spectrum.

**Implementation:**
- High-pass FIR filter (3 Hz cutoff). Tap count auto-scales with the sampling rate to preserve filter sharpness — equivalent to 1000 taps at 400 Hz, which works out to ~2500 taps for this project's native 1000 Hz data.
- Designed strictly using the **Window Method** (Hamming window).
- Applied via zero-phase filtering (`scipy.signal.filtfilt`) to prevent any non-linear phase distortion of the QRS complexes.

### 2. Power-Line Interference Canceller

**Challenge:** 50 Hz power-line noise and its harmonics frequently corrupt clinical recordings.

**Implementation:**
- Adaptive noise cancellation via a configurable Phase-Locked Loop (PLL), each tracked component with its own independently tracked amplitude and phase. Defaults to standard single-tone cancellation of the 50 Hz mains fundamental only; can be configured to additionally cancel a chosen number of harmonics (e.g. 100/150/200 Hz), up to a 200 Hz limit. The main analysis notebook configures 4-harmonic cancellation (fundamental + 100/150/200 Hz) for all reported results.
- Features an **amplitude-based blocking mechanism** that suspends filter adaptation during high-energy QRS complexes, strictly protecting the cardiac morphology from being filtered out.

### 3. Signal Upsampling

**Challenge:** Accurate removal of the maternal ECG requires sub-millisecond precision alignment.

**Implementation:**
- The signal is upsampled from its native rate (1000 Hz for this dataset) to 2000 Hz.
- Relies on polyphase filtering (`resample_poly`) to ensure strict anti-aliasing while increasing temporal resolution for optimal template matching.

### 4. Maternal ECG (MECG) Canceller

**Challenge:** The maternal heartbeat dominates the abdominal recording and must be precisely subtracted without damaging the underlying fetal signal.

**Implementation:**
- **Channel Combination:** Principal Component Analysis (PCA) extracts the dominant maternal cardiac axis.
- **QRS Detection:** Employs a **Matched Filter (Cross-correlation)** to detect R-peaks.
- **Robust Template Generation:** Calculates a moving average of the last 10 maternal beats, implementing a trimming technique (discarding maximum and minimum amplitude beats) to reject outliers and prevent fetal QRS contamination.
- **Subtraction:** Segments the template into P, QRS, and T waves, fitting each independently to the raw signal via **Ridge-regularised least squares** (a small Tikhonov penalty is added to the normal equations for numerical stability).

### 5. Fetal ECG Extractor

**Challenge:** The residual signal contains the isolated but noisy fECG, requiring precise detection and enhancement.

**Implementation:**
- **Fetal QRS Detection:** Re-applies the Matched Filter technique on the MECG-free residual to locate fetal R-peaks.
- **Synchronous Averaging:** Computes the FHR and performs ensemble averaging over 150 consecutive fetal beats to dramatically increase the Signal-to-Noise Ratio (SNR) and reveal the clean fetal cardiac morphology.

## Project Structure

```
BSP_project/
├── data/                       # Raw PhysioNet records, organised as data/set_a/ with matching .fqrs ground-truth annotations
│                               #   (data/set_b/ also ships in this repo but has no .fqrs annotations, so the pipeline doesn't use it)
├── notebooks/                  # Jupyter notebooks for interactive analysis
│   └── main_analysis.ipynb     # Pipeline execution and validation
├── results/                    # Auto-generated figures (git-ignored; re-created by running the notebook)
└── src/                        # Core Python modules
    ├── filtering.py            # FIR baseline and adaptive 50Hz filters
    ├── preprocessing.py        # Upsampling routines
    ├── mecg_canceller.py       # Maternal ECG detection and least squares subtraction
    ├── fecg_extractor.py       # Fetal QRS detection and synchronous averaging
    ├── utils.py                # FHR/reliability/success-rate metrics and ground-truth validation
    └── plotting.py             # Shared matplotlib visualisations for the notebook
```

## Validation & Evaluation

The `main_analysis.ipynb` notebook includes advanced validation steps:
- **Welch's Periodogram:** Compares the Power Spectral Density (PSD) before and after filtering.
- **ICA Benchmarking:** Uses `FastICA` (from `scikit-learn`) as a baseline BSS method to demonstrate the robustness of the sequential approach in noisy environments.
- **Ground-Truth Validation:** Matches detected fetal QRS locations against the reference `.fqrs` annotations shipped with the dataset to compute real precision, recall and F1.

## Dependencies

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `wfdb`
- `pandas`
- `jupyter`

## Usage

1. Place your raw `.dat`/`.hea` files, plus their matching `.fqrs` ground-truth annotation files, into `data/set_a/` (this project targets the PhysioNet/CinC Challenge 2013 dataset layout, not the separate NIFECGDB database — check [physionet.org](https://physionet.org) for the exact record set you need).
2. Open `notebooks/main_analysis.ipynb`.
3. Run the pipeline cells sequentially to process the signals, extract the fECG, and visualise the results.

## References

Martens, S. M. M., Rabotti, C., Mischi, M., & Sluijter, R. J. (2007). *A robust fetal ECG detection method for abdominal recordings*. Physiological Measurement, 28(4), 373-388.
Xiao, Y., Lu, Y., Liu, M., Zeng, R., & Bai, J. (2022). A deep feature fusion network for fetal state assessment. Frontiers in Physiology, 13, 969052.
