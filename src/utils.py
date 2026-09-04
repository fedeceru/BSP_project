import os
import numpy as np
import wfdb
from typing import Tuple

def compute_fhr(fetal_peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Calculates Fetal Heart Rate (FHR) in beats per minute (bpm) based on detected peaks.
    Filters out non-physiological values outside the ~78 bpm (1.3 Hz) to ~198 bpm (3.3 Hz) bounds.

    Args:
        fetal_peaks (np.ndarray): Array of detected fetal R-peak indices.
        fs (float): Sampling frequency of the signal the peaks were detected on, in Hz.

    Returns:
        np.ndarray: Array of valid FHR values in bpm.
    """
    if len(fetal_peaks) < 2:
        return np.array([])

    rr_intervals_sec = np.diff(fetal_peaks) / fs

    # Filter RR intervals based on physiological constraints (1.3 Hz to 3.3 Hz -> 0.77s to 0.303s)
    valid_rr = rr_intervals_sec[(rr_intervals_sec > 0.303) & (rr_intervals_sec < 0.77)]

    if len(valid_rr) == 0:
        return np.array([])

    return 60.0 / valid_rr

def compute_fhr_reliability(fhr_values: np.ndarray, fhr_times: np.ndarray,
                            block_size_sec: float = 10.0, outlier_threshold_bpm: float = 10.0) -> float:
    """
    Calculates the FHR detection reliability as defined in section 2.4.1:
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
    Calculates the FHR detection success rate as defined in section 2.4.1:
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

def compute_snr_sir(s4: np.ndarray, s5: np.ndarray, s6: np.ndarray, peaks: np.ndarray, fs: float,
                    window_size_sec: float = 0.25, num_beats: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the per-channel SNR and SIR of the abdominal FECG as defined in
    section 2.4.2. The FECG power PF is estimated from S6, the synchronously
    averaged fetal beat. The MECG power PM is estimated from the difference
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
