import numpy as np
from typing import Optional

class FECGExtractor:
    def __init__(self, fs: float = 2000.0):
        self.fs = fs

    def detect_fetal_qrs(self, residual_signal: np.ndarray, template_width_sec: float = 0.05) -> np.ndarray:
        """
        Detect fetal QRS complexes using Cross-correlation / Matched Filter on the residual.
        """
        if len(residual_signal) == 0:
            return np.array([])
            
        width_samples = int(template_width_sec * self.fs)
        
        # Simple extraction for initial fetal template
        search_window = min(len(residual_signal), int(3 * self.fs))
        if search_window == 0:
            return np.array([])
            
        initial_peak = int(np.argmax(np.abs(residual_signal[:search_window])))
        half_width = width_samples // 2
        
        start_idx = max(0, initial_peak - half_width)
        end_idx = min(len(residual_signal), initial_peak + half_width)
        template = residual_signal[start_idx:end_idx]
        
        if len(template) == 0:
            return np.array([])
            
        cross_corr = np.correlate(residual_signal, template, mode='same')
        
        # FHR can be up to 180-200 bpm -> 3-3.3 Hz -> ~0.3s min dist
        threshold = 0.5 * np.max(cross_corr)
        min_dist = int(0.25 * self.fs)
        
        peaks = self._find_peaks(cross_corr, threshold, min_dist)
        return peaks

    def _find_peaks(self, signal: np.ndarray, threshold: float, min_dist: int) -> np.ndarray:
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

    def compute_fhr(self, fetal_peaks: np.ndarray) -> np.ndarray:
        """
        Calculate Fetal Heart Rate (FHR) in beats per minute (bpm).
        """
        if len(fetal_peaks) < 2:
            return np.array([])
            
        rr_intervals_sec = np.diff(fetal_peaks) / self.fs
        
        # Filter out invalid RR intervals (e.g. 0s or very large)
        valid_rr = rr_intervals_sec[rr_intervals_sec > 0.1]
        
        if len(valid_rr) == 0:
            return np.array([])
            
        fhr = 60.0 / valid_rr
        return fhr

    def synchronous_averaging(self, signal: np.ndarray, peaks: np.ndarray, num_beats: int = 150, window_size_sec: float = 0.4) -> Optional[np.ndarray]:
        """
        Average a set number of fetal beats (default 150) to improve Signal-to-Noise Ratio.
        """
        if len(peaks) == 0:
            return None
            
        half_window = int((window_size_sec / 2) * self.fs)
        beats = []
        
        # Take up to the last num_beats
        selected_peaks = peaks[-num_beats:] if len(peaks) > num_beats else peaks
        
        for p in selected_peaks:
            start = p - half_window
            end = p + half_window
            if start >= 0 and end <= len(signal):
                beats.append(signal[start:end])
                
        if not beats:
            return None
            
        beats_array = np.array(beats)
        fecg_template = np.mean(beats_array, axis=0)
        return fecg_template
