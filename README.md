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
Raw Abdominal Signal (400 Hz)
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
- High-pass FIR filter (1000 taps, 3 Hz cutoff).
- Designed strictly using the **Window Method** (Hamming/Blackman).
- Applied via zero-phase filtering (`scipy.signal.filtfilt`) to prevent any non-linear phase distortion of the QRS complexes.

### 2. Power-Line Interference Canceller

**Challenge:** 50 Hz power-line noise and its harmonics frequently corrupt clinical recordings.

**Implementation:**
- Adaptive noise cancellation targeting 50 Hz and its first 3 harmonics.
- Features an **amplitude-based blocking mechanism** that suspends filter adaptation during high-energy QRS complexes, strictly protecting the cardiac morphology from being filtered out.

### 3. Signal Upsampling

**Challenge:** Accurate removal of the maternal ECG requires sub-millisecond precision alignment.

**Implementation:**
- The signal is upsampled from 400 Hz to 2000 Hz.
- Relies on polyphase filtering (`resample_poly`) to ensure strict anti-aliasing while increasing temporal resolution for optimal template matching.

### 4. Maternal ECG (MECG) Canceller

**Challenge:** The maternal heartbeat dominates the abdominal recording and must be precisely subtracted without damaging the underlying fetal signal.

**Implementation:**
- **Channel Combination:** Principal Component Analysis (PCA) extracts the dominant maternal cardiac axis.
- **QRS Detection:** Employs a **Matched Filter (Cross-correlation)** to detect R-peaks, explicitly avoiding Hilbert Transform envelopes.
- **Robust Template Generation:** Calculates a moving average of the last 10 maternal beats, implementing a trimming technique (discarding maximum and minimum amplitude beats) to reject outliers and prevent fetal QRS contamination.
- **Subtraction:** Segments the template into P, QRS, and T waves, fitting them to the raw signal using **Standard Least Squares** (Moore-Penrose pseudo-inverse). No Ridge Regression or regularization is used, strictly adhering to standard OLS.

### 5. Fetal ECG Extractor

**Challenge:** The residual signal contains the isolated but noisy fECG, requiring precise detection and enhancement.

**Implementation:**
- **Fetal QRS Detection:** Re-applies the Matched Filter technique on the MECG-free residual to locate fetal R-peaks.
- **Synchronous Averaging:** Computes the FHR and performs ensemble averaging over 150 consecutive fetal beats to dramatically increase the Signal-to-Noise Ratio (SNR) and reveal the clean fetal cardiac morphology.

## Project Structure

```
BSP_project/
├── data/           # Raw PhysioNet datasets (e.g., NIFECGDB)
├── docs/           # Reference papers and documentation
├── notebooks/      # Jupyter notebooks for interactive analysis
│   └── main_analysis.ipynb # Pipeline execution and validation
├── results/        # Saved figures, PSDs, and extracted metrics
└── src/            # Core Python modules
    ├── filtering.py       # FIR baseline and adaptive 50Hz filters
    ├── preprocessing.py   # Upsampling routines
    ├── mecg_canceller.py  # Maternal ECG detection and least squares subtraction
    └── fecg_extractor.py  # Fetal QRS detection and synchronous averaging
```

## Validation & Evaluation

The `main_analysis.ipynb` notebook includes advanced validation steps:
- **Welch's Periodogram:** Compares the Power Spectral Density (PSD) before and after filtering.
- **ICA Benchmarking:** Uses `FastICA` (from `scikit-learn`) as a baseline BSS method to demonstrate the superior robustness of our sequential approach in noisy environments, as claimed by the reference paper.
- **Signal Quality:** Implements Sample Entropy (SampEn) to quantitatively evaluate the complexity and quality of the extracted fECG.

## Dependencies

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `wfdb`
- `pandas`
- `jupyter`

## Usage

1. Place your raw `.dat` and `.hea` files from the [PhysioNet Non-Invasive Fetal ECG Database](https://physionet.org/content/nifecgdb/) into the `data/` directory.
2. Open `notebooks/main_analysis.ipynb`.
3. Run the pipeline cells sequentially to process the signals, extract the fECG, and visualize the results.

## References

Martens, S. M. M., Rabotti, C., Mischi, M., & Sluijter, R. J. (2007). *A robust fetal ECG detection method for abdominal recordings*. Physiological Measurement, 28(4), 373-388.
