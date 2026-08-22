import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, List

class MECGCanceller:
    def __init__(self, fs: float = 2000.0):
        self.fs = fs

    def extract_principal_component(self, channels: np.ndarray) -> np.ndarray:
        """
        Extract the principal component from multi-channel data using PCA.
        channels: shape (n_channels, n_samples)
        """
        if channels.ndim != 2:
            raise ValueError(f"Expected 2D array for channels, got shape {channels.shape}")
            
        n_channels, n_samples = channels.shape
        if n_channels == 1:
            return channels[0] # Nothing to PCA
            
        pca = PCA(n_components=1)
        # PCA expects (n_samples, n_features)
        pc = pca.fit_transform(channels.T)
        return pc.flatten()

    def detect_qrs_matched_filter(self, signal: np.ndarray, template_width_sec: float = 0.1) -> np.ndarray:
        """
        Detect QRS complexes using a Matched Filter (Cross-correlation) without Hilbert Transform.
        """
        if len(signal) == 0:
            return np.array([])
            
        width_samples = int(template_width_sec * self.fs)
        
        # Heuristic to find an initial good template: find the absolute maximum in early signal
        search_window = min(len(signal), int(3 * self.fs))
        if search_window == 0:
            return np.array([])
            
        initial_peak = int(np.argmax(np.abs(signal[:search_window])))
        half_width = width_samples // 2
        
        # Ensure bounds
        start_idx = max(0, initial_peak - half_width)
        end_idx = min(len(signal), initial_peak + half_width)
        template = signal[start_idx:end_idx]
        
        if len(template) == 0:
            return np.array([])
        
        # Cross-correlate (Matched filter)
        cross_corr = np.correlate(signal, template, mode='same')
        
        # Find peaks in cross-correlation
        threshold = 0.6 * np.max(cross_corr)
        peaks = self._find_peaks(cross_corr, threshold, min_dist=int(0.5 * self.fs))
        
        return peaks

    def _find_peaks(self, signal: np.ndarray, threshold: float, min_dist: int) -> np.ndarray:
        """Helper to find local maxima above a threshold with a minimum distance."""
        peaks = []
        n = len(signal)
        i = 0
        while i < n:
            if signal[i] > threshold:
                window_end = min(i + min_dist, n)
                peak_idx = i + np.argmax(signal[i:window_end])
                peaks.append(peak_idx)
                i = peak_idx + min_dist
            else:
                i += 1
        return np.array(peaks)

    def compute_robust_template(self, signal: np.ndarray, peaks: np.ndarray, current_peak_idx: int, window_size_sec: float = 0.6) -> Optional[np.ndarray]:
        """
        Compute a robust template by averaging the last 10 beats,
        excluding the max and min amplitude beats (trimming/clipping).
        """
        if current_peak_idx < 10 or current_peak_idx > len(peaks):
            return None # Not enough history
            
        history_peaks = peaks[current_peak_idx-10:current_peak_idx]
        half_window = int((window_size_sec / 2) * self.fs)
        
        beats = []
        for p in history_peaks:
            start = p - half_window
            end = p + half_window
            if start >= 0 and end < len(signal):
                beats.append(signal[start:end])
                
        if len(beats) < 10:
            return None
            
        beats = np.array(beats)
        
        # Calculate peak-to-peak amplitude for each beat
        amps = np.ptp(beats, axis=1)
        
        # Find indices of max and min amplitude beats
        max_idx = int(np.argmax(amps))
        min_idx = int(np.argmin(amps))
        
        # Exclude max and min to avoid fetal contamination/outliers
        keep_indices = [i for i in range(10) if i != max_idx and i != min_idx]
        robust_beats = beats[keep_indices]
        
        # Average the remaining beats
        template = np.mean(robust_beats, axis=0)
        return template

    def subtract_mecg_least_squares(self, signal: np.ndarray, peaks: np.ndarray, window_size_sec: float = 0.6) -> np.ndarray:
        """
        Model P, QRS, T waves using Standard Least Squares (Moore-Penrose pseudo-inverse)
        and subtract from the original signal. No Ridge Regression used.
        """
        residual = signal.copy()
        
        if len(peaks) < 10:
            return residual # Cannot apply robust cancellation without history
            
        half_window = int((window_size_sec / 2) * self.fs)
        
        for i in range(10, len(peaks)):
            p = peaks[i]
            start = p - half_window
            end = p + half_window
            
            if start < 0 or end > len(signal):
                continue
                
            template = self.compute_robust_template(signal, peaks, i, window_size_sec)
            if template is None or len(template) != (end - start):
                continue
                
            # Segment the template into P, QRS, T
            # Approximate indices for a 0.6s window centered on R peak
            p_end = int(0.4 * len(template))
            qrs_end = int(0.6 * len(template))
            
            T_P = np.zeros_like(template)
            T_P[:p_end] = template[:p_end]
            
            T_QRS = np.zeros_like(template)
            T_QRS[p_end:qrs_end] = template[p_end:qrs_end]
            
            T_T = np.zeros_like(template)
            T_T[qrs_end:] = template[qrs_end:]
            
            # Create design matrix M (columns are P, QRS, T templates)
            M = np.column_stack([T_P, T_QRS, T_T])
            
            # Extract actual segment
            y = signal[start:end]
            
            # Standard Least Squares (Moore-Penrose Pseudo-Inverse)
            try:
                # a = (M^T M)^-1 M^T y = pinv(M) y
                M_pinv = np.linalg.pinv(M)
                a = np.dot(M_pinv, y)
                
                # Reconstruct fitted MECG complex
                fitted_mecg = np.dot(M, a)
                
                # Subtract
                residual[start:end] -= fitted_mecg
            except np.linalg.LinAlgError:
                # Fallback to straight subtraction if SVD fails (very rare)
                residual[start:end] -= template
            
        return residual
