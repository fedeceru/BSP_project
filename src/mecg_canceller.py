import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks, correlate
from sklearn.decomposition import PCA
from typing import Tuple, List, Optional

class MECGCanceller:
    """
    Maternal ECG (MECG) Canceller class.
    Extracts and subtracts maternal QRS complexes from abdominal ECG signals
    using an adaptive template subtraction method with Ridge Regression and
    robust template estimation.
    """

    def __init__(self, fs: float = 2000.0, win_pre: float = 0.25, win_post: float = 0.45,
                 n_avg: int = 10, qrs_win: float = 0.05, lam: float = 1e-3):
        """
        Constructor for MECGCanceller.

        Args:
            fs (float): Sampling frequency of the signal in Hz. Defaults to 2000.0.
            win_pre (float): Window size in seconds before the R-peak. Defaults to 0.25.
            win_post (float): Window size in seconds after the R-peak. Defaults to 0.45.
            n_avg (int): Number of historical beats to average for the template. Defaults to 10.
            qrs_win (float): Half-window size in seconds to isolate the QRS complex. Defaults to 0.05.
            lam (float): Ridge regression regularisation parameter (Tikhonov penalty). Defaults to 1e-3.
        """
        self.fs = fs
        self.win_pre = win_pre
        self.win_post = win_post
        self.n_avg = n_avg
        self.qrs_win = qrs_win
        self.lam = lam

    def _bandpass_detect(self, X: np.ndarray) -> np.ndarray:
        """
        Applies a bandpass filter to enhance QRS complexes for detection.

        Args:
            X (np.ndarray): Multi-channel signal matrix of shape (n_samples, n_channels).

        Returns:
            np.ndarray: Bandpass filtered signal matrix.
        """
        b, a = butter(3, [5/(self.fs/2), 40/(self.fs/2)], btype='band')
        return filtfilt(b, a, X, axis=0)

    def enhance_maternal_qrs(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Enhances the maternal QRS complexes using a multi-channel approach:
        1. Bandpass filter to remove baseline wander and high-frequency noise.
        2. Channel variance normalization.
        3. PCA to extract the first principal component (maximum variance = max SNR).

        Args:
            signal_matrix (np.ndarray): Multi-channel signal matrix.

        Returns:
            np.ndarray: A 1D array representing the enhanced maternal QRS signal.
        """
        # Step 1: Bandpass filter to isolate the QRS frequency band
        bp = self._bandpass_detect(signal_matrix)

        # Step 2: Channel variance normalization
        # Center the data by subtracting the mean
        centred = bp - np.mean(bp, axis=0)
        # Calculate standard deviation for each channel
        std_dev = np.std(centred, axis=0)
        # Avoid division by zero in case of a completely flat channel
        std_dev[std_dev == 0] = 1.0
        # Normalize the variance (standard deviation = 1 for all channels)
        normalized = centred / std_dev

        # Step 3: PCA to extract the main component
        pca_signal = PCA(n_components=1).fit_transform(normalized).flatten()

        # Ensure a consistent polarity so that maternal R-peaks are always
        # positive maxima (the PCA sign is otherwise arbitrary).
        if np.max(pca_signal) < np.abs(np.min(pca_signal)):
            pca_signal = -pca_signal

        return pca_signal

    def detect_maternal_qrs(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Detects maternal QRS complexes via multi-channel enhancement followed by
        cross-correlation with a QRS template.

        Args:
            signal_matrix (np.ndarray): Multi-channel signal matrix.

        Returns:
            np.ndarray: Array of refined R-peak indices.
        """
        pca_signal = self.enhance_maternal_qrs(signal_matrix)

        # Find rough peaks to build an initial template
        # We use a simple threshold on the squared signal to find high-energy regions
        squared_signal = pca_signal ** 2
        thresh = np.mean(squared_signal) + 2 * np.std(squared_signal)
        min_dist = int(0.35 * self.fs)
        rough_peaks, _ = find_peaks(squared_signal, height=thresh, distance=min_dist)

        # Build a median template from rough peaks
        window_sec = 0.08
        w_half = int((window_sec * self.fs) / 2)

        segments = []
        for p in rough_peaks:
            if p - w_half < 0 or p + w_half >= len(pca_signal):
                continue
            segments.append(pca_signal[p - w_half : p + w_half])

        if not segments:
            return rough_peaks

        maternal_template = np.median(np.array(segments), axis=0)

        # Perform cross-correlation with the template
        cc = correlate(pca_signal, maternal_template, mode='same')

        # Detect true peaks by finding the maxima of the cross-correlation
        cc_thresh = np.mean(cc) + 1.5 * np.std(cc)
        final_peaks, _ = find_peaks(cc, height=cc_thresh, distance=min_dist)

        # Refine each peak to the exact local maximum of the enhanced signal,
        # since the cross-correlation peak can be off by a few samples.
        refine_half_win = max(1, w_half)
        refined_peaks = []
        for p in final_peaks:
            start = max(0, p - refine_half_win)
            end = min(len(pca_signal), p + refine_half_win + 1)
            refined_peaks.append(start + int(np.argmax(pca_signal[start:end])))

        return np.array(refined_peaks, dtype=int)

    def _get_segments(self, win_len: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Calculates index boundaries to split a heartbeat window into P, QRS, and T segments.

        Args:
            win_len (int): Total length of the heartbeat window in samples.

        Returns:
            Tuple containing three tuples of (start_idx, end_idx) for P, QRS, and T waves respectively.
        """
        center = int(self.win_pre * self.fs)
        qrs_rad = int(self.qrs_win * self.fs)

        qrs_start = max(0, center - qrs_rad)
        qrs_end = min(win_len, center + qrs_rad)

        p_start = 0
        p_end = qrs_start

        t_start = qrs_end
        t_end = win_len

        return (p_start, p_end), (qrs_start, qrs_end), (t_start, t_end)

    def _extract_beat(self, sig: np.ndarray, peak: int, win_len: int, n_samples: int) -> Optional[np.ndarray]:
        """
        Extracts a signal segment around a detected peak.

        Args:
            sig (np.ndarray): Single-channel signal array.
            peak (int): Index of the R-peak.
            win_len (int): Total length of the window to extract.
            n_samples (int): Total length of the signal (used for boundary checking).

        Returns:
            Optional[np.ndarray]: Extracted heartbeat segment, or None if out of bounds.
        """
        start = peak - int(self.win_pre * self.fs)
        end = start + win_len
        
        if start < 0 or end > n_samples:
            return None
        
        return sig[start:end]

    def _compute_robust_template(self, beats: List[np.ndarray]) -> np.ndarray:
        """
        Computes a robust template by averaging recent beats, discarding the 
        maximum and minimum amplitude beats to avoid fetal contamination or artifacts.

        Args:
            beats (List[np.ndarray]): List of recently extracted heartbeat segments.

        Returns:
            np.ndarray: Robust averaged heartbeat template.
        """
        beats_arr = np.array(beats)
        n_beats = len(beats_arr)
        
        # If not enough history, fallback to standard mean
        if n_beats < 3:
            return np.mean(beats_arr, axis=0)
            
        # Calculate peak-to-peak amplitude for each beat
        amps = np.ptp(beats_arr, axis=1)
        
        max_idx = int(np.argmax(amps))
        min_idx = int(np.argmin(amps))
        
        # Keep beats excluding max and min
        keep_indices = [i for i in range(n_beats) if i != max_idx and i != min_idx]
        robust_beats = beats_arr[keep_indices]
        
        return np.mean(robust_beats, axis=0)

    def apply(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Applies the complete MECG cancellation pipeline on a multi-channel signal.
        Pads the signal, detects peaks, computes robust adaptive templates, models
        P-QRS-T complexes via Ridge Regression, and subtracts them.

        Args:
            signal_matrix (np.ndarray): Original multi-channel abdominal ECG (n_samples, n_channels).

        Returns:
            np.ndarray: Fetal ECG signal matrix (MECG removed) of the same shape as input.
        """
        n_samples, n_channels = signal_matrix.shape
        
        # Padding to safely process beats near the edges
        pad_pre = int(self.win_pre * self.fs) + 100
        pad_post = int(self.win_post * self.fs) + 100
        sig_padded = np.pad(signal_matrix, ((pad_pre, pad_post), (0, 0)), mode='edge')
        
        n_samples_pad = sig_padded.shape[0]
        fetal_ecg_padded = np.zeros_like(sig_padded)

        m_peaks = self.detect_maternal_qrs(sig_padded)
        win_len = int((self.win_pre + self.win_post) * self.fs)
        
        for ch in range(n_channels):
            sig = sig_padded[:, ch]
            residue = sig.copy()
            
            beat_buffer = []
            
            for peak in m_peaks:
                curr = self._extract_beat(sig, peak, win_len, n_samples_pad)
                if curr is None:
                    continue

                # The template is built only from beats already in the buffer,
                # keeping it independent of the current beat being cancelled.
                # With no prior beats yet, the current beat is used as its own
                # template so that this first occurrence still gets subtracted.
                avg_template = self._compute_robust_template(beat_buffer) if beat_buffer else curr

                # Segment the template for independent scaling
                M = np.zeros((win_len, 3))
                (p_s, p_e), (q_s, q_e), (t_s, t_e) = self._get_segments(win_len)

                M[p_s:p_e, 0] = avg_template[p_s:p_e]
                M[q_s:q_e, 1] = avg_template[q_s:q_e]
                M[t_s:t_e, 2] = avg_template[t_s:t_e]

                try:
                    # Ridge Regression (Tikhonov Regularisation)
                    A = M.T @ M + self.lam * np.eye(3)
                    b = M.T @ curr
                    coeffs = np.linalg.solve(A, b)
                    fitted_mecg = M @ coeffs
                except np.linalg.LinAlgError:
                    # Fallback straight subtraction if inversion fails
                    fitted_mecg = avg_template

                # Subtraction
                start = peak - int(self.win_pre * self.fs)
                end = start + win_len
                residue[start:end] = curr - fitted_mecg

                beat_buffer.append(curr)
                if len(beat_buffer) > self.n_avg:
                    beat_buffer.pop(0)

            fetal_ecg_padded[:, ch] = residue

        # Remove padding to return original dimensions
        fetal_ecg_final = fetal_ecg_padded[pad_pre:-pad_post, :]
        
        return fetal_ecg_final
