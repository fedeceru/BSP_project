import os
import numpy as np
import wfdb
from typing import Tuple
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

FHR_MIN_BPM_bradycardia = 78 # 1.3Hz * 60 -- Bradycardia
FHR_MAX_BPM_tachycardia = 198.0 # 3.3Hz * 60 -- Tachycardia

FHR_MIN_BPM_std = 120 # 2Hz * 60 -- Physiological lower bound
FHR_MAX_BPM_std = 162 # 2.7Hz * 60 -- Physiological upper bound

def _fhr_bounds(std: bool) -> Tuple[float, float]:
    if std:
        return FHR_MIN_BPM_std, FHR_MAX_BPM_std
    return FHR_MIN_BPM_bradycardia, FHR_MAX_BPM_tachycardia

def compute_fhr_series(fetal_peaks: np.ndarray, fs: float, std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Fetal Heart Rate (FHR) in beats per minute (bpm) based on detected peaks, paired
    with the time (in seconds, at the end of each RR interval) of each value. Filters out values
    outside the [FHR_MIN_BPM, FHR_MAX_BPM] physiological bounds, keeping times and values aligned
    via an explicit mask (rather than assuming filtered-out points form a contiguous prefix).

    Args:
        fetal_peaks (np.ndarray): Array of detected/annotated fetal R-peak indices.
        fs (float): Sampling frequency of the signal the peaks were detected on, in Hz.
        std (bool): Whether to use the standard physiological bounds instead of the
            wider bradycardia/tachycardia bounds.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (times_sec, fhr_bpm), same length, only physiologically
        valid points included.
    """
    if len(fetal_peaks) < 2:
        return np.array([]), np.array([])

    times_sec = fetal_peaks[1:] / fs
    rr_intervals_sec = np.diff(fetal_peaks) / fs
    fhr_bpm = 60.0 / rr_intervals_sec

    fhr_min, fhr_max = _fhr_bounds(std)
    valid = (fhr_bpm >= fhr_min) & (fhr_bpm <= fhr_max)
    return times_sec[valid], fhr_bpm[valid]

def compute_fhr(fetal_peaks: np.ndarray, fs: float, std: bool = True) -> np.ndarray:
    """
    Calculates Fetal Heart Rate (FHR) in beats per minute (bpm) based on detected peaks.
    Filters out values outside the [FHR_MIN_BPM, FHR_MAX_BPM] physiological bounds.

    Args:
        fetal_peaks (np.ndarray): Array of detected fetal R-peak indices.
        fs (float): Sampling frequency of the signal the peaks were detected on, in Hz.
        std (bool): Whether to use the standard physiological bounds instead of the
            wider bradycardia/tachycardia bounds.

    Returns:
        np.ndarray: Array of valid FHR values in bpm.
    """
    _, fhr_bpm = compute_fhr_series(fetal_peaks, fs, std)
    return fhr_bpm

def compute_fhr_coverage(fetal_peaks: np.ndarray, fs: float, total_duration_sec: float,
                         std: bool = True) -> float:
    """
    Fraction of the recording covered by physiologically valid, RR-interval-derived FHR
    estimates. Sums the duration of every in-bounds RR interval and divides by the recording
    length, so a trace that is only sparsely/occasionally valid scores low even if it contains
    a couple of coincidentally in-range beats.

    Args:
        fetal_peaks (np.ndarray): Detected fetal R-peak indices.
        fs (float): Sampling frequency of the signal the peaks were detected on, in Hz.
        total_duration_sec (float): Duration of the recording/segment being evaluated, in seconds.
        std (bool): Whether to use the standard physiological bounds instead of the wider
            bradycardia/tachycardia bounds. Should match the `std` value used for `compute_fhr`.

    Returns:
        float: Coverage in [0, 1].
    """
    if total_duration_sec <= 0:
        return 0.0

    _, fhr_bpm = compute_fhr_series(fetal_peaks, fs, std)
    if len(fhr_bpm) == 0:
        return 0.0

    covered_sec = np.sum(60.0 / fhr_bpm)
    return min(covered_sec / total_duration_sec, 1.0)

def compute_fhr_mae_rmse(gt_times: np.ndarray, gt_fhr: np.ndarray,
                         est_times: np.ndarray, est_fhr: np.ndarray,
                         total_duration_sec: float, grid_step_sec: float = 1.0
                         ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Mean absolute error (MAE) and root-mean-square error (RMSE), in bpm, between an estimated
    FHR trace and a ground-truth FHR trace. Detected and ground-truth peaks generally do not
    occur at the same instants, so both traces are linearly interpolated onto a common regular
    time grid before differencing; no extrapolation is performed, so only the time range covered
    by both traces (their overlap) contributes to the comparison.

    Args:
        gt_times (np.ndarray): Time (s) of each ground-truth FHR value, as returned by
            compute_fhr_series on the `.fqrs` annotations.
        gt_fhr (np.ndarray): Ground-truth FHR values in bpm.
        est_times (np.ndarray): Time (s) of each estimated FHR value.
        est_fhr (np.ndarray): Estimated FHR values in bpm.
        total_duration_sec (float): Duration of the recording/segment being evaluated, in seconds.
        grid_step_sec (float): Spacing of the common comparison grid, in seconds.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray]: (mae_bpm, rmse_bpm, overlap_fraction,
        gt_on_grid, est_on_grid). overlap_fraction is the share of the grid actually covered by
        both traces (low overlap means the error estimate rests on little data). gt_on_grid /
        est_on_grid are the paired interpolated values within the overlap (e.g. for an
        estimated-vs-ground-truth scatter plot); empty if there is no overlap, in which case the
        scalar metrics are NaN.
    """
    if len(gt_times) < 2 or len(est_times) < 2 or total_duration_sec <= 0:
        return np.nan, np.nan, 0.0, np.array([]), np.array([])

    grid = np.arange(0.0, total_duration_sec, grid_step_sec)

    overlap_start = max(gt_times[0], est_times[0])
    overlap_end = min(gt_times[-1], est_times[-1])
    in_overlap = (grid >= overlap_start) & (grid <= overlap_end)

    if not np.any(in_overlap):
        return np.nan, np.nan, 0.0, np.array([]), np.array([])

    gt_on_grid = np.interp(grid[in_overlap], gt_times, gt_fhr)
    est_on_grid = np.interp(grid[in_overlap], est_times, est_fhr)

    diff = est_on_grid - gt_on_grid
    mae_bpm = np.mean(np.abs(diff))
    rmse_bpm = np.sqrt(np.mean(diff ** 2))
    overlap_fraction = np.sum(in_overlap) / len(grid)

    return mae_bpm, rmse_bpm, overlap_fraction, gt_on_grid, est_on_grid

def compute_fhr_reliability(fhr_values: np.ndarray, fhr_times: np.ndarray,
                            block_size_sec: float = 10.0, outlier_threshold_bpm: float = 10.0) -> float:
    """
    Calculates the FHR detection reliability:
    1 minus the ratio between the number of outliers and the total number of
    points in the FHR trace. A point is an outlier if it deviates more than
    `outlier_threshold_bpm` from the median FHR calculated over its own
    `block_size_sec`-second block.

    Args:
        fhr_values (np.ndarray): FHR values in bpm, as returned by compute_fhr.
        fhr_times (np.ndarray): Time (in seconds) associated with each FHR value.
        block_size_sec (float): Block duration in seconds. Defaults to 10.0.
        outlier_threshold_bpm (float): Outlier deviation threshold in bpm. Defaults to 10.0.

    Returns:
        float: Reliability in [0, 1], or NaN if the trace is empty (undefined).
    """
    if len(fhr_values) == 0:
        return np.nan

    fhr_values = np.asarray(fhr_values)
    block_idx = (np.asarray(fhr_times) // block_size_sec).astype(int)

    is_outlier = np.zeros(len(fhr_values), dtype=bool)
    for b in np.unique(block_idx):
        mask = block_idx == b
        block_median = np.median(fhr_values[mask])
        is_outlier[mask] = np.abs(fhr_values[mask] - block_median) > outlier_threshold_bpm

    return 1.0 - (np.sum(is_outlier) / len(fhr_values))

def compute_success_rate(feasible_flags) -> float:
    """
    Calculates the FHR detection success rate:
    the percentage of patients for which a physiologically feasible FHR trace was found.

    Args:
        feasible_flags: Boolean array or Series, one entry per patient.

    Returns:
        float: Success rate as a percentage in [0, 100].
    """
    return np.mean(feasible_flags) * 100

def load_ground_truth_peaks(data_dir: str, record_name: str, target_fs: float) -> np.ndarray:
    """
    Loads the reference fetal QRS annotations (.fqrs) for a record and rescales
    them from their native sampling rate to target_fs, so they align sample-for-sample
    with signals resampled to target_fs.

    Args:
        data_dir (str): Directory containing the record's .fqrs annotation file.
        record_name (str): Record name (without extension).
        target_fs (float): Sampling frequency to rescale the annotation sample indices to.

    Returns:
        np.ndarray: Ground-truth peak indices at target_fs.
    """
    ann = wfdb.rdann(os.path.join(data_dir, record_name), 'fqrs')
    return np.round(ann.sample * (target_fs / ann.fs)).astype(int)

def match_peaks_to_gt(detected: np.ndarray, gt_peaks: np.ndarray, tolerance: int) -> Tuple[int, int, int]:
    """
    Greedily matches detected peaks to ground-truth peaks within a sample tolerance.

    Args:
        detected (np.ndarray): Detected peak indices.
        gt_peaks (np.ndarray): Ground-truth peak indices.
        tolerance (int): Maximum distance, in samples, for a detected peak to count as a match.

    Returns:
        Tuple[int, int, int]: (true positives, false positives, false negatives).
    """
    matched_gt = np.zeros(len(gt_peaks), dtype=bool)
    tp = 0
    for d in detected:
        if len(gt_peaks) == 0:
            break
        diffs = np.abs(gt_peaks - d)
        j = np.argmin(diffs)
        if diffs[j] <= tolerance and not matched_gt[j]:
            matched_gt[j] = True
            tp += 1
    fn = len(gt_peaks) - tp
    fp = len(detected) - tp
    return tp, fp, fn

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculates precision, recall and F1 score from true/false positive/negative counts.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        Tuple[float, float, float]: (precision, recall, f1), NaN where undefined.
    """
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
    return precision, recall, f1

def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, ci: float = 0.95,
                     random_state: int = None) -> Tuple[float, float, float, float]:
    """
    Spearman's rank correlation coefficient between x and y, with a percentile bootstrap
    confidence interval obtained by resampling the paired (x, y) observations with replacement.

    Args:
        x (np.ndarray): First variable (e.g. mean SNR/SIR in dB), one value per record.
        y (np.ndarray): Second variable (e.g. reliability), same length as x. NaNs in either
            array are dropped pairwise before correlating.
        n_boot (int): Number of bootstrap resamples.
        ci (float): Confidence level, e.g. 0.95 for a 95% CI.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[float, float, float, float]: (rho, p_value, ci_low, ci_high). All NaN if fewer
        than 3 valid paired points remain after dropping NaNs.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan

    rho, p_value = spearmanr(x, y)

    rng = np.random.default_rng(random_state)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rhos[i] = spearmanr(x[idx], y[idx])[0]

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.nanpercentile(boot_rhos, [100 * alpha, 100 * (1 - alpha)])
    return rho, p_value, ci_low, ci_high

def fit_reliability_collapse_point(x: np.ndarray, y: np.ndarray, reliability_threshold: float = 0.5
                                   ) -> Tuple[float, bool]:
    """
    Locates an empirical "reliability collapse point" along x (e.g. mean SNR or SIR, in dB):
    the x value at which a 1D logistic classifier predicts a 50% chance of reliability falling
    below `reliability_threshold`. Reliability is first binarized (>= threshold vs. below) and
    a logistic regression is fit against x; the collapse point is where the fitted probability
    crosses 0.5, i.e. -intercept / coefficient.

    Args:
        x (np.ndarray): Predictor values (e.g. mean SNR or SIR in dB), one per record.
        y (np.ndarray): Reliability values in [0, 1], one per record.
        reliability_threshold (float): Reliability level below which a record counts as
            "collapsed".

    Returns:
        Tuple[float, bool]: (collapse_point_db, converged). (NaN, False) if there are fewer
        than 2 records in either class (the classifier would be degenerate) or the fitted
        coefficient is zero.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    labels = (y >= reliability_threshold).astype(int)

    if len(np.unique(labels)) < 2 or np.min(np.bincount(labels)) < 2:
        return np.nan, False

    clf = LogisticRegression()
    clf.fit(x.reshape(-1, 1), labels)
    coef, intercept = clf.coef_[0, 0], clf.intercept_[0]
    if coef == 0:
        return np.nan, False
    return -intercept / coef, True

def compute_snr_sir(s4: np.ndarray, s5: np.ndarray, s6: np.ndarray, peaks: np.ndarray, fs: float,
                    window_size_sec: float = 0.25, num_beats: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the per-channel SNR and SIR of the abdominal FECG. The FECG
    power PF is estimated from S6, the synchronously averaged fetal beat.
    The MECG power PM is estimated from the difference
    between S4 and S5 (the maternal component the MECG canceller removed),
    over the whole signal. The noise power PN is estimated from the residual
    between each individual fetal beat window in S5 and the averaged template
    S6, since averaging over many beats cancels out anything not correlated
    with the fetal beat, leaving essentially noise.

    Args:
        s4 (np.ndarray): Multi-channel signal before MECG cancellation (n_samples, n_channels).
        s5 (np.ndarray): Multi-channel signal after MECG cancellation (n_samples, n_channels).
        s6 (np.ndarray): Synchronously averaged fetal beat, as returned by
            FECGExtractor.synchronous_averaging(s5, peaks, window_size_sec, num_beats).
        peaks (np.ndarray): Fetal QRS peak indices used to build s6.
        fs (float): Sampling frequency in Hz.
        window_size_sec (float): Beat-extraction window; must match the one used to build s6.
        num_beats (int): Number of trailing beats averaged; must match the one used to build s6.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (snr_db, sir_db), one value per channel.
    """
    half_window = int((window_size_sec / 2) * fs)
    selected_peaks = peaks[-num_beats:] if len(peaks) > num_beats else peaks

    noise_segments = []
    for p in selected_peaks:
        start = p - half_window
        end = p + half_window
        if start >= 0 and end <= len(s5):
            noise_segments.append(s5[start:end, :] - s6)

    p_f = np.mean(s6 ** 2, axis=0)
    p_m = np.mean((s4 - s5) ** 2, axis=0)

    if noise_segments:
        p_n = np.mean(np.concatenate(noise_segments, axis=0) ** 2, axis=0)
    else:
        p_n = np.full(s6.shape[1], np.nan)

    snr_db = 10 * np.log10(p_f / p_n)
    sir_db = 10 * np.log10(p_f / p_m)

    return snr_db, sir_db
