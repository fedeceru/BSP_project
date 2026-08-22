import numpy as np
from scipy import signal
from typing import Union

class BaselineWanderRemover:
    def __init__(self, num_taps: int = 1000, cutoff_hz: float = 3.0, fs: float = 400.0, window: str = 'hamming'):
        """
        Baseline Wander Remover using FIR high-pass filter designed with Window method.
        """
        self.num_taps = num_taps
        self.cutoff_hz = cutoff_hz
        self.fs = fs
        self.window = window
        self.taps = self._design_filter()

    def _design_filter(self) -> np.ndarray:
        """
        Design the FIR high-pass filter using the window method.
        """
        if self.cutoff_hz >= self.fs / 2:
            raise ValueError("Cutoff frequency must be less than Nyquist frequency.")
            
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff_hz / nyq
        
        if self.num_taps % 2 == 0:
            self.num_taps += 1
        
        # Design high-pass FIR filter using the window method
        taps = signal.firwin(self.num_taps, normal_cutoff, pass_zero=False, window=self.window)
        return taps

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase filtering to the data to prevent phase distortion.
        Assumes data is a 1D numpy array or 2D array (channels x samples).
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
            
        if data.size == 0:
            raise ValueError("Data cannot be empty.")
            
        # filtfilt operates along axis=-1 by default, so it natively handles 2D arrays efficiently
        return signal.filtfilt(self.taps, 1.0, data, axis=-1)


class PowerLineCanceller:
    def __init__(self, fs: float = 400.0, f0: float = 50.0, mu: float = 0.01, harmonics: int = 3, qrs_threshold_ratio: float = 2.5):
        """
        Adaptive power-line interference canceller with amplitude-based blocking.
        """
        self.fs = fs
        self.f0 = f0
        self.mu = mu
        self.harmonics = harmonics
        self.qrs_threshold_ratio = qrs_threshold_ratio

    def apply(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Apply the adaptive filter to cancel 50Hz and its harmonics.
        Uses a blocking mechanism to suspend adaptation during QRS complexes.
        """
        if ecg_signal.ndim != 1:
            raise ValueError("PowerLineCanceller currently supports 1D array per call. Please iterate over channels.")
            
        N = len(ecg_signal)
        n = np.arange(N)
        
        # Calculate a robust threshold for QRS detection (to block adaptation)
        std_val = np.std(ecg_signal)
        mean_val = np.mean(ecg_signal)
        threshold = mean_val + self.qrs_threshold_ratio * std_val
        
        # Generate reference signals (sine and cosine for 50Hz and harmonics)
        ref_signals = []
        for h in range(1, self.harmonics + 1):
            freq = h * self.f0
            if freq < self.fs / 2: # Keep below Nyquist
                ref_signals.append(np.cos(2 * np.pi * freq * n / self.fs))
                ref_signals.append(np.sin(2 * np.pi * freq * n / self.fs))
                
        if not ref_signals:
            return ecg_signal.copy()
            
        ref_signals = np.array(ref_signals)
        num_refs = ref_signals.shape[0]
        
        weights = np.zeros(num_refs)
        clean_signal = np.zeros(N)
        
        # Sequential LMS adaptation (cannot be fully vectorized over time)
        for i in range(N):
            x_vec = ref_signals[:, i]
            interference_est = np.dot(weights, x_vec)
            error = ecg_signal[i] - interference_est
            clean_signal[i] = error
            
            # Blocking mechanism: if error is too large, we might be in a QRS complex.
            # Suspend adaptation to protect the QRS complex.
            if abs(error) < threshold:
                weights = weights + 2 * self.mu * error * x_vec
                
        return clean_signal
