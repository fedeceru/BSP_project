import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple

class BaselineWanderRemover:
    """
    Baseline Wander Remover class.
    Implements a zero-phase high-pass FIR filter designed with the window method
    to eliminate low-frequency baseline wander from ECG signals without introducing
    phase distortion.
    """
    def __init__(self, num_taps: int = 1000, cutoff_hz: float = 3.0, fs: float = 400.0, window: str = 'hamming'):
        """
        Constructor for BaselineWanderRemover.

        Args:
            num_taps (int): Number of filter coefficients. Defaults to 1000.
            cutoff_hz (float): Cutoff frequency of the high-pass filter in Hz. Defaults to 3.0.
            fs (float): Sampling frequency of the signal in Hz. Defaults to 400.0.
            window (str): Window function to use for FIR filter design. Defaults to 'hamming'.
        """
        self.num_taps = num_taps
        self.cutoff_hz = cutoff_hz
        self.fs = fs
        self.window = window
        self.taps = self._design_filter()

    def _design_filter(self) -> np.ndarray:
        """
        Designs a high-pass FIR filter using the window method.

        Returns:
            np.ndarray: The filter coefficients (FIR filter taps).
            
        Raises:
            ValueError: If the cutoff frequency exceeds the Nyquist limit.
        """
        if self.cutoff_hz >= self.fs / 2:
            raise ValueError("Cutoff frequency must be less than Nyquist frequency.")

        # Derive the normalised cutoff frequency for the FIR filter design
        nyq = 0.5 * self.fs 
        normal_cutoff = self.cutoff_hz / nyq

        # Ensure the number of taps is odd for a Type I FIR filter to have a zero phase response, 
        # so that the filter does not introduce phase distortion to the ECG signal. 
        # Type I filter allows for a non-zero amplitude at the Nyquist frequency (fs/2), 
        # making QRS complexes more accurately preserved. 
        if self.num_taps % 2 == 0:
            self.num_taps += 1

        # Hamming function is used to reduce the Gibbs phenomenon, which can cause ripples 
        # in the frequency response of the filter.
        taps = signal.firwin(self.num_taps, normal_cutoff, pass_zero=False, window=self.window)
        return taps

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase filtering to the data to prevent phase distortion.
        Assumes data is a 1D numpy array or 2D array (channels x samples).

        Args:
            data (np.ndarray): The raw ECG signal array.

        Returns:
            np.ndarray: The baseline wander removed signal.
            
        Raises:
            TypeError: If input data is not a numpy array.
            ValueError: If input data is empty.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
            
        if data.size == 0:
            raise ValueError("Data cannot be empty.")
            
        # filtfilt applies a linear digital filter twice, once forward and once backwards.
        # By doing so we go from a linear-phase filter to a zero-phase filter.
        return signal.filtfilt(self.taps, 1.0, data, axis=-1)


class AdaptivePLICanceller:
    """
    Adaptive Power Line Interference (PLI) Canceller class.
    Removes power-line interference (e.g., 50 Hz mains noise) from ECG signals 
    using a Phase-Locked Loop (PLL) and error-isolation mechanisms.
    """
    def __init__(self, fs: float = 400.0, f_line: float = 50.0):
        """
        Constructor for AdaptivePLICanceller.
        
        Configures the Phase-Locked Loop (PLL) and error-isolation mechanisms 
        through three core components:
        
        1. Amplitude Tracking: Sets the adaptation gain (K_a) using a 0.13s time constant 
           to smoothly track fluctuations in the interference volume.
        2. Frequency/Phase Tracking (PLL): Configures a critically damped (zeta = 1.0) 
           control loop. The gains (K_dw, K_phi) are tuned to strictly track slow, 
           natural drifts in the mains frequency without oscillating.
        3. Error Isolation Filter: Designs an internal 2nd-order 80 Hz IIR high-pass filter. 
           The coefficients (b_err, a_err) are mathematically normalised to guarantee exactly 
           unity gain (1.0) at the nominal mains frequency (f_line). This acts as a visor, 
           hiding low-frequency cardiac components from the learning algorithm to ensure 
           stable gradient calculations.

        Args:
            fs (float): Sampling frequency of the signal in Hz. Defaults to 400.0.
            f_line (float): Nominal power-line frequency in Hz. Defaults to 50.0.
        """
        self.fs = fs
        self.f_line = f_line 
        # Natural frequency of the mains interference in radians/sample
        self.w_n = 2 * np.pi * f_line / fs 
        
        # Time constant for amplitude adaptation
        tau = 0.13 
        # Adaptation gain for amplitude
        self.K_a = 1.0 / (fs * tau) 

        # PLL parameters for frequency and phase adaptation
        # Goldilocks damping ratio for the PLL, without oscillations or sluggishness
        zeta = 1.0 
        # Consider only slow frequency variations in the mains frequency
        ratio_wn_wp = 0.04 
        omega_n = self.w_n * ratio_wn_wp  

        # Adaptation gains for frequency and phase
        self.K_dw = omega_n ** 2 
        self.K_phi = 2 * zeta * omega_n 

        # Design a 2nd order IIR high-pass filter to remove baseline wander and low-frequency noise
        cutoff_hz = 80.0
        nyq = 0.5 * fs 
        b, a = signal.butter(2, cutoff_hz / nyq, btype='high')

        # Complex number representing frequency response at f_line
        w, h = self._freqz_scalar(b, a, f_line, fs) 
        # Ignoring the phase response, we only need the magnitude response at 50 Hz
        gain_at_50 = np.abs(h)  

        # Normalise the filter coefficients to ensure unity gain at 50 Hz,
        # keeping cutoff frequency intact at 80 Hz
        self.b_err = b / gain_at_50
        self.a_err = a

    @staticmethod
    def _lfilter_step(b: np.ndarray, a: np.ndarray, x: float, zi: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Performs a single IIR filter step whilst retaining the state zi.

        Args:
            b (np.ndarray): Numerator coefficients of the filter.
            a (np.ndarray): Denominator coefficients of the filter.
            x (float): Current input sample.
            zi (np.ndarray): Current state of the filter.

        Returns:
            Tuple[float, np.ndarray]: 
                - y: The filtered output for the current input sample.
                - zi: The updated filter state for the next input sample.
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
        Simulates the behaviour of scipy.signal.freqz but for a single frequency point,
        without having to compute the full frequency response array.

        Args:
            b (np.ndarray): Numerator coefficients.
            a (np.ndarray): Denominator coefficients.
            f (float): Frequency point of interest in Hz.
            fs (float): Sampling frequency in Hz.

        Returns:
            Tuple[float, complex]:
                - w: The normalised frequency in radians/sample.
                - h: The complex frequency response at the specified frequency.
        """
        w = 2 * np.pi * f / fs # Convert frequency to radians/sample
        zm1 = np.exp(-1j * w)  # Compute z^-1 for the given frequency
        # Apply the filter coefficients to compute the frequency response
        num = np.polyval(b, zm1) # Evaluate the numerator polynomial at z^-1 
        den = np.polyval(a, zm1) # Evaluate the denominator polynomial at z^-1
        return w, num / den 
        
    def _apply_comb_filter_and_detect_blocking(self, d_k: np.ndarray) -> np.ndarray:
        """
        Uses a comb filter to estimate the signal energy without the mains interference, 
        deciding when to block the adaptation.
        
        Calculates the difference between the current sample and the sample one period 
        ago (1/f_line seconds). It computes the rolling standard deviation of this difference 
        signal to estimate the noise level. A threshold is set at sqrt(2) times the rolling 
        standard deviation. If the absolute value of the difference signal exceeds this threshold, 
        it indicates a potential QRS complex (or significant event), blocking adaptation for a 
        short period around that sample.
        
        Args:
            d_k (np.ndarray): The raw ECG signal array.

        Returns:
            np.ndarray: A boolean mask array indicating where adaptation should be blocked.
        """
        beta = int(round(self.fs / self.f_line))
        
        pad = np.zeros(beta) 
        d_shifted = np.concatenate((pad, d_k[:-beta])) 
        d_H = d_k - d_shifted # Compute the difference signal
        
        win_len = int(self.fs)
        d_H_series = pd.Series(d_H)
        # Compute the rolling standard deviation to estimate the noise level
        sigma = d_H_series.rolling(window=win_len, center=True).std().fillna(0).values
        
        # Set the threshold for blocking adaptation
        chi = np.sqrt(2) * sigma
        
        raw_mask = np.abs(d_H) > chi
        expansion = int(0.05 * self.fs)
        blocking_mask = np.convolve(raw_mask.astype(int), np.ones(2 * expansion + 1), mode='same') > 0
        
        return blocking_mask

    def apply(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Executes the adaptive cancellation loop sample-by-sample.

        Algorithm steps:
        1. Computes the reference signals (sine and cosine) based on the current phase estimate.
        2. Calculates the error signal by subtracting the estimated interference from the actual signal.
        3. Applies the high-pass filter to both the error and reference signals to remove low-frequency artifacts.
        4. Updates the adaptive parameters (amplitude, frequency, phase) based on the filtered signals, 
           only if the current sample is outside a blocking region (e.g., QRS complex).
        5. Constrains amplitude to remain non-negative and caps frequency drift to prevent instability.
        
        Args:
            ecg_signal (np.ndarray): 1D array representing a single ECG channel.

        Returns:
            np.ndarray: The error signal acting as the clean output, with reduced mains interference.
            
        Raises:
            TypeError: If input data is not a numpy array.
            ValueError: If input array is not 1-dimensional.
        """
        if not isinstance(ecg_signal, np.ndarray):
            raise TypeError("Data must be a numpy array.")
            
        if ecg_signal.ndim != 1:
            raise ValueError("AdaptivePLICanceller currently supports 1D array per call. Please iterate over channels.")
            
        n_samples = len(ecg_signal)
        e_output = np.zeros(n_samples)
        
        # Initialise adaptive parameters
        theta_a = 0.0 # AMPLITUDE
        theta_phi = 0.0 # PHASE
        theta_dw = 0.0 # FREQUENCY
        
        # Initialise filter states for the error and reference signals
        zi_e = np.zeros(max(len(self.a_err), len(self.b_err)) - 1)
        zi_y_sin = np.zeros_like(zi_e) 
        zi_y_cos = np.zeros_like(zi_e) 
        
        blocking_mask = self._apply_comb_filter_and_detect_blocking(ecg_signal)
        
        for k in range(n_samples):
            # Compute argument for sine and cosine based on sample index and phase estimate
            arg = self.w_n * k + theta_phi 
            ref_sin = np.sin(arg)
            ref_cos = np.cos(arg)
            
            # Estimated interference based on current amplitude
            x_hat = theta_a * ref_sin 
            d_val = ecg_signal[k]
            e_val = d_val - x_hat 
            e_output[k] = e_val
            
            e_w, zi_e = self._lfilter_step(self.b_err, self.a_err, e_val, zi_e)
            
            y_sin_w, zi_y_sin = self._lfilter_step(self.b_err, self.a_err, ref_sin, zi_y_sin)
            y_cos_w, zi_y_cos = self._lfilter_step(self.b_err, self.a_err, ref_cos, zi_y_cos)
            
            if not blocking_mask[k]:
                alpha = 1.0 / theta_a if theta_a > 1e-6 else 1.0

                # Derivative of the error with respect to the adaptive parameters:
                # - If e_w is correlated with the sine, amplitude is too low/high.
                # - If e_w is correlated with the cosine, the phase needs adjustment.
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