import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple

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
        if self.cutoff_hz >= self.fs / 2:
            raise ValueError("Cutoff frequency must be less than Nyquist frequency.")

        # Derive the normalized cutoff frequency for the FIR filter design            
        nyq = 0.5 * self.fs 
        normal_cutoff = self.cutoff_hz / nyq

        # Ensure the number of taps is odd for a Type I FIR filter to have a zero phase response. 
        # Type I filter allows for a non-zero amplitude at the Nyquist frequency (fs/2), making it possible to create high-pass filters. 
        if self.num_taps % 2 == 0:
            self.num_taps += 1

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
            
        # filtfilt applies a linear digital filter twice, once forward and once backwards.
        # The combined filter has zero phase and a filter order twice that of the original.
        return signal.filtfilt(self.taps, 1.0, data, axis=-1)


class AdaptivePLICanceller:
    def __init__(self, fs: float = 400.0, f_line: float = 50.0):
        """
        Adaptive mains interference canceller tracking amplitude, frequency, 
        and phase with a rolling-window blocking mechanism to protect QRS complexes.
        """
        self.fs = fs
        self.f_line = f_line 
        self.w_n = 2 * np.pi * f_line / fs # Natural frequency of the mains interference in radians/sample
        
        tau = 0.13 # Time constant for amplitude adaptation
        self.K_a = 1.0 / (fs * tau) # Adaptation gain for amplitude

        # PLL parameters for frequency and phase adaptation
        zeta = 1.0 # Goldilocks damping ratio for the PLL, without oscillations or sluggishness
        ratio_wn_wp = 0.04 # consider only slow frequency variations in the mains frequency
        omega_n = self.w_n * ratio_wn_wp  

        # Adaptation gains for frequency and phase
        self.K_dw = omega_n ** 2 
        self.K_phi = 2 * zeta * omega_n 

        # Design a IIR high-pass filter of 2nd order to remove baseline wander and low-frequency noise
        cutoff_hz = 80.0
        nyq = 0.5 * fs 
        b, a = signal.butter(2, cutoff_hz / nyq, btype='high')

        w, h = self._freqz_scalar(b, a, f_line, fs) # it return a complex number
        gain_at_50 = np.abs(h) # ignoring the phase response, we only need the magnitude response at 50 Hz 

        # Normalize the filter coefficients to ensure unity gain at 50 Hz, keeping cutoff frequency intact at 80 Hz
        self.b_err = b / gain_at_50
        self.a_err = a

    @staticmethod
    def _lfilter_step(b: np.ndarray, a: np.ndarray, x: float, zi: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Performs a single IIR filter step whilst retaining the state zi.

        Returns:
            y: The filtered output for the current input sample.
            zi: The updated filter state for the next input sample.
        """
        y = b[0] * x + zi[0]
        for i in range(len(zi) - 1):
            zi[i] = b[i+1] * x - a[i+1] * y + zi[i+1]
        zi[-1] = b[-1] * x - a[-1] * y
        return y, zi

    @staticmethod
    def _freqz_scalar(b: np.ndarray, a: np.ndarray, f: float, fs: float) -> Tuple[float, complex]:
        """
        Calculates the frequency response at a single frequency f.
        Simulates the behavior of scipy.signal.freqz but for a single frequency point, without having to compute the full frequency response.

        Returns:
            w: The normalized frequency in radians/sample.
            h: The complex frequency response at the specified frequency.
        """
        w = 2 * np.pi * f / fs # Convert frequency to radians/sample
        zm1 = np.exp(-1j * w) # Compute z^-1 for the given frequency (50 Hz)
        # Apply the filter coefficients to compute the frequency response at that frequency
        num = np.polyval(b, zm1) # Evaluate the numerator polynomial at z^-1 
        den = np.polyval(a, zm1) # Evaluate the denominator polynomial at z^-1
        return w, num / den 
        
    def _apply_comb_filter_and_detect_blocking(self, d_k: np.ndarray) -> np.ndarray:
        """
        Uses a comb filter to estimate the signal energy without the mains   
        interference, deciding when to block the adaptation.
        It obtains that by calculating the difference between the current sample and the sample one period ago (1/f_line seconds).
        Then, it computes the rolling standard deviation of this difference signal to estimate the noise level.
        A threshold is set at sqrt(2) times the rolling standard deviation, and if the absolute value of the difference signal exceeds this threshold, 
        it indicates a potential QRS complex or other significant event, and adaptation is blocked for a short period around that sample.
        
        Returns:
            blocking_mask: A boolean array indicating where adaptation should be blocked.
        """
        beta = int(round(self.fs / self.f_line))
        
        pad = np.zeros(beta) 
        d_shifted = np.concatenate((pad, d_k[:-beta])) 
        d_H = d_k - d_shifted # Compute the difference signal
        
        win_len = int(self.fs)
        d_H_series = pd.Series(d_H)
        # Compute the rolling standard deviation of the difference signal to estimate the noise level
        sigma = d_H_series.rolling(window=win_len, center=True).std().fillna(0).values
        
        # Set the threshold for blocking adaptation based on the estimated noise level
        chi = np.sqrt(2) * sigma
        
        raw_mask = np.abs(d_H) > chi
        expansion = int(0.05 * self.fs)
        blocking_mask = np.convolve(raw_mask.astype(int), np.ones(2 * expansion + 1), mode='same') > 0
        
        return blocking_mask

    def apply(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Executes the adaptive cancellation loop sample-by-sample.
        1) It computes the reference signals (sine and cosine) based on the current phase estimate.
        2) It calculates the error signal by subtracting the estimated interference from the actual ECG signal.
        3) It applies the high-pass filter to both the error and reference signals to remove low-frequency noise and baseline wander.
        4) It updates the adaptive parameters (amplitude, frequency, and phase) based on the filtered signals, but only if the current sample is not 
        within a blocking region (i.e., not part of a QRS complex or other significant event).
        5) It ensures that the amplitude remains non-negative and that the frequency adaptation is constrained within a reasonable range to prevent instability.
        
        Returns:
            e_output: The error signal after adaptive cancellation, which should have reduced mains interference.
        """
        if not isinstance(ecg_signal, np.ndarray):
            raise TypeError("Data must be a numpy array.")
            
        if ecg_signal.ndim != 1:
            raise ValueError("AdaptivePLICanceller currently supports 1D array per call. Please iterate over channels.")
            
        n_samples = len(ecg_signal)
        e_output = np.zeros(n_samples)
        
        # Initialize adaptive parameters
        theta_a = 0.0 # AMPLITUDE
        theta_phi = 0.0 # PHASE
        theta_dw = 0.0 # FREQUENCY
        
        # Initialize filter states for the error and reference signals
        zi_e = np.zeros(max(len(self.a_err), len(self.b_err)) - 1)
        zi_y_sin = np.zeros_like(zi_e) 
        zi_y_cos = np.zeros_like(zi_e) 
        
        blocking_mask = self._apply_comb_filter_and_detect_blocking(ecg_signal)
        
        for k in range(n_samples):
            arg = self.w_n * k + theta_phi # Compute the argument for the sine and cosine functions based on the current sample index and phase estimate
            ref_sin = np.sin(arg)
            ref_cos = np.cos(arg)
            
            x_hat = theta_a * ref_sin # Compute the estimated interference based on the current amplitude and reference sine signal
            d_val = ecg_signal[k]
            e_val = d_val - x_hat 
            e_output[k] = e_val
            
            e_w, zi_e = self._lfilter_step(self.b_err, self.a_err, e_val, zi_e)
            
            y_sin_w, zi_y_sin = self._lfilter_step(self.b_err, self.a_err, ref_sin, zi_y_sin)
            y_cos_w, zi_y_cos = self._lfilter_step(self.b_err, self.a_err, ref_cos, zi_y_cos)
            
            if not blocking_mask[k]:
                alpha = 1.0 / theta_a if theta_a > 1e-6 else 1.0

                # Derivative of the error with respect to the adaptive parameters
                # If e_w it's similar to the sine, it means the amplitude is too low, so we need to increase it.
                # If e_w is similar to the cosine, it means the phase is off, so we need to adjust it.
                eta_a = e_w * y_sin_w 
                eta_phi = e_w * (alpha * y_cos_w)

                # Update amplitude 
                theta_a_next = theta_a + self.K_a * eta_a
                
                if theta_a_next < 0:    
                    theta_a_next = 0.0

                # Update frequency
                theta_dw_next = theta_dw + self.K_dw * eta_phi
                
                max_dw = 2 * np.pi * 4.0 / self.fs
                theta_dw_next = np.clip(theta_dw_next, -max_dw, max_dw)

                # Update phase
                theta_phi_next = theta_phi + self.K_phi * eta_phi + theta_dw
                
                theta_a = theta_a_next
                theta_dw = theta_dw_next
                theta_phi = theta_phi_next
                
        return e_output