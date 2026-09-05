import numpy as np
import pandas as pd
from scipy import signal
from typing import Optional, Tuple

class BaselineWanderRemover:
    """
    Baseline Wander Remover class.
    Implements a zero-phase high-pass FIR filter designed with the window method
    to eliminate low-frequency baseline wander from ECG signals without introducing
    phase distortion.
    """
    def __init__(self, num_taps: Optional[int] = None, cutoff_hz: float = 3.0, fs: float = 1000.0, window: str = 'hamming'):
        """
        Constructor for BaselineWanderRemover.

        Args:
            num_taps (Optional[int]): Number of filter coefficients. If not given, it
                is scaled from fs to keep the same filter duration (and therefore the
                same relative transition bandwidth) as 1000 taps at 400 Hz. Defaults to None.
            cutoff_hz (float): Cutoff frequency of the high-pass filter in Hz. Defaults to 3.0.
            fs (float): Sampling frequency of the signal in Hz. Defaults to 1000.0 (this
                project's standard native/original sampling rate -- pass fs_orig explicitly).
            window (str): Window function to use for FIR filter design. Defaults to 'hamming'.
        """
        self.num_taps = num_taps if num_taps is not None else int(round(1000 * fs / 400))
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
    Removes power-line interference (e.g., 50 Hz mains noise) from ECG signals using a
    Phase-Locked Loop (PLL) and error-isolation mechanisms.

    Configurable between two cancellation strategies via `num_harmonics`:
    - Standard (num_harmonics=1, the default): tracks and cancels only the mains
      fundamental (~50 Hz) with a single PLL.
    - Selective harmonic cancellation (num_harmonics>1): additionally tracks and cancels
      the next (num_harmonics - 1) mains harmonics, each with its own independent PLL, up
      to a hard limit of MAX_HARMONIC_HZ (200 Hz).
    """
    MAX_HARMONIC_HZ = 200.0

    def __init__(self, fs: float = 1000.0, f_line: float = 50.0, num_harmonics: int = 1):
        """
        Constructor for AdaptivePLICanceller.

        Tracks and cancels the mains fundamental plus its next (num_harmonics - 1) harmonics
        (f_line, 2*f_line, ..., num_harmonics*f_line). Each harmonic runs its own independent
        amplitude/frequency/phase-locked loop, each isolated from the others by its own band-pass
        filter (see point 3 below) -- an earlier version derived every harmonic's phase as a multiple
        of one shared fundamental estimate, but even the small residual phase jitter that a single
        tracked tone settles into is amplified by h once multiplied out to a harmonic, which left
        higher harmonics converging to a biased amplitude far below their true value.

        Every harmonic's frequency/phase loop shares the same gains, derived once from the
        fundamental (see point 2 below) rather than scaled up per harmonic -- mains frequency drift is
        one slow physical process shared by every harmonic, not something that gets faster for higher
        ones, and scaling the loop gains up by h was found to make the smallest, highest harmonics
        unstable enough to lose an otherwise good lock. `apply()` also warm-starts each harmonic's
        amplitude and phase from a short direct projection rather than letting every harmonic climb
        from zero -- with a fixed adaptation gain, a small-amplitude+harmonic can need far longer than
        a typical recording to climb from zero even when it would track perfectly well once started
        near its true value.

        Configures the Phase-Locked Loop (PLL) and error-isolation mechanisms
        through three core components, instantiated once per tracked harmonic:

        1. Amplitude Tracking: Sets the adaptation gain (K_a) using a 0.13s time constant
           to smoothly track fluctuations in that harmonic's interference volume. Frequency-independent,
           so shared across every harmonic.
        2. Frequency/Phase Tracking (PLL): Configures a critically damped (zeta = 1.0)
           control loop, sized once from the fundamental's own nominal frequency and shared by every
           harmonic. The gains (K_dw, K_phi) are tuned to strictly track slow, natural drifts in the
           mains frequency without oscillating.
        3. Error Isolation Filters: Designs one internal 2nd-order IIR band-pass filter per tracked
           harmonic, centred on that harmonic's own frequency. Each filter's coefficients are
           mathematically normalised to guarantee exactly unity gain (1.0) at that harmonic's own
           frequency. Being centred (not just high-pass) also keeps neighbouring harmonics, only
           f_line apart, from leaking into each other's isolated error signal. This acts as a visor,
           hiding both low-frequency cardiac content and adjacent harmonics from the learning
           algorithm to ensure stable gradient calculations.

        Args:
            fs (float): Sampling frequency of the signal in Hz. Defaults to 1000.0 (this
                project's standard native/original sampling rate -- pass fs_orig explicitly).
            f_line (float): Nominal power-line frequency in Hz. Defaults to 50.0.
            num_harmonics (int): Number of mains components to track, starting from the fundamental
                (1 = standard single-tone cancellation of the fundamental only, >1 = selective
                harmonic cancellation, additionally tracking the next (num_harmonics - 1) harmonics).
                Defaults to 1 (standard cancellation).

        Raises:
            ValueError: If num_harmonics is less than 1, if the highest tracked harmonic exceeds
                the MAX_HARMONIC_HZ cancellation limit, or if it isn't safely below the Nyquist
                frequency for the given fs.
        """
        if num_harmonics < 1:
            raise ValueError("num_harmonics must be at least 1.")

        if f_line * num_harmonics > self.MAX_HARMONIC_HZ:
            raise ValueError(
                f"Highest tracked harmonic ({f_line * num_harmonics:.1f} Hz) exceeds the "
                f"{self.MAX_HARMONIC_HZ:.0f} Hz cancellation limit. Reduce num_harmonics."
            )

        self.fs = fs
        self.f_line = f_line
        self.num_harmonics = num_harmonics

        nyq = 0.5 * fs
        if f_line * num_harmonics >= 0.9 * nyq:
            raise ValueError(
                f"Highest tracked harmonic ({f_line * num_harmonics:.1f} Hz) is too close to the "
                f"Nyquist frequency ({nyq:.1f} Hz) for fs={fs}. Reduce num_harmonics or increase fs."
            )

        # Time constant for amplitude adaptation -- frequency-independent, shared by every harmonic
        tau = 0.13
        self.K_a = 1.0 / (fs * tau)

        # Goldilocks damping ratio for every PLL, without oscillations or sluggishness
        zeta = 1.0
        # Consider only slow frequency variations relative to the fundamental's own frequency
        ratio_wn_wp = 0.04

        # Frequency/phase-adaptation gains are derived from the fundamental only and then shared by
        # every harmonic, rather than scaled up per harmonic. Mains frequency drift is a single slow
        # physical process shared by every harmonic, not something that itself gets "faster" for
        # higher harmonics -- scaling the loop gains up by h (so each harmonic's phase loop reacts h
        # times faster) makes higher harmonics far more sensitive to noise for no tracking benefit,
        # and was found to destabilise the smallest, highest harmonics enough to erase an otherwise
        # good amplitude/phase estimate.
        omega_n_fundamental = (2 * np.pi * f_line / fs) * ratio_wn_wp
        K_dw_shared = omega_n_fundamental ** 2
        K_phi_shared = 2 * zeta * omega_n_fundamental

        half_bw_hz = 10.0
        self.w_n_list = []   # nominal angular frequency per harmonic, radians/sample
        self.K_dw_list = []  # frequency-adaptation gain per harmonic (shared across harmonics)
        self.K_phi_list = []  # phase-adaptation gain per harmonic (shared across harmonics)
        self.b_err_list = []
        self.a_err_list = []
        for h in range(1, num_harmonics + 1):
            f_h = f_line * h

            # This harmonic's own carrier frequency, radians/sample -- needed to generate the right
            # reference sine/cosine -- but the same shared PLL gains for every harmonic.
            self.w_n_list.append(2 * np.pi * f_h / fs)
            self.K_dw_list.append(K_dw_shared)
            self.K_phi_list.append(K_phi_shared)

            # 2nd order IIR band-pass filter centred on this harmonic's own frequency, renormalised
            # to unity gain there. Being centred (not just high-pass) keeps neighbouring harmonics,
            # only f_line apart, from leaking into this harmonic's isolated error signal.
            low = (f_h - half_bw_hz) / nyq
            high = (f_h + half_bw_hz) / nyq
            b, a = signal.butter(2, [low, high], btype='band')

            # Complex number representing frequency response at this harmonic's frequency
            w, hresp = self._freqz_scalar(b, a, f_h, fs)
            # Ignoring the phase response, we only need the magnitude response at this harmonic
            gain_at_h = np.abs(hresp)
            self.b_err_list.append(b / gain_at_h)
            self.a_err_list.append(a)

        # Kept for backwards compatibility with anything referencing the original single-tone filter
        self.w_n = self.w_n_list[0]
        self.K_dw = self.K_dw_list[0]
        self.K_phi = self.K_phi_list[0]
        self.b_err = self.b_err_list[0]
        self.a_err = self.a_err_list[0]

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
        1. Computes each harmonic's own reference sine/cosine signals from its own independent
           phase estimate.
        2. Sums each harmonic's own amplitude-scaled sine reference into a combined interference
           estimate, then computes one error signal by subtracting it from the actual signal.
        3. Applies each harmonic's own band-pass filter to that same error signal, and to that
           harmonic's own reference signals, to isolate the part of the error relevant to that
           harmonic -- every sample, regardless of blocking, so each filter's internal state stays
           continuous.
        4. Updates each harmonic's amplitude, frequency and phase independently from its own isolated
           error signal, but only if the current sample is outside a blocking region (e.g., QRS
           complex).
        5. Constrains every amplitude to remain non-negative and caps every harmonic's frequency
           drift to prevent instability.

        Args:
            ecg_signal (np.ndarray): 1D array representing a single ECG channel.

        Returns:
            np.ndarray: The error signal acting as the clean output, with reduced mains interference
                and reduced harmonic content.

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
        num_harmonics = self.num_harmonics

        # Initialise adaptive parameters: independent amplitude/phase/frequency per harmonic.
        # Starting every harmonic from zero amplitude works fine for the fundamental (large
        # amplitude, fast to climb via the gradient steps below), but smaller harmonics need many
        # more seconds of gradient descent to climb from zero than a 60s recording provides. Instead,
        # warm-start each harmonic's amplitude and phase from a direct narrowband projection onto a
        # short initial window, then let the per-sample loop below take over for fine tracking and
        # drift-following from that point on.
        warmup_samples = min(int(2.0 * self.fs), n_samples)
        warmup_t = np.arange(warmup_samples) / self.fs
        theta_a = np.zeros(num_harmonics)    # AMPLITUDE
        theta_phi = np.zeros(num_harmonics)  # PHASE
        theta_dw = np.zeros(num_harmonics)   # FREQUENCY
        if warmup_samples > 0:
            warmup_seg = ecg_signal[:warmup_samples]
            for h_idx in range(num_harmonics):
                f_h = self.f_line * (h_idx + 1)
                c_sin = 2.0 * np.mean(warmup_seg * np.sin(2 * np.pi * f_h * warmup_t))
                c_cos = 2.0 * np.mean(warmup_seg * np.cos(2 * np.pi * f_h * warmup_t))
                theta_a[h_idx] = np.hypot(c_sin, c_cos)
                theta_phi[h_idx] = np.arctan2(c_cos, c_sin)

        # Initialise filter states for the error and reference signals, one triple per harmonic
        state_len = max(len(self.a_err_list[0]), len(self.b_err_list[0])) - 1
        zi_e = [np.zeros(state_len) for _ in range(num_harmonics)]
        zi_y_sin = [np.zeros(state_len) for _ in range(num_harmonics)]
        zi_y_cos = [np.zeros(state_len) for _ in range(num_harmonics)]

        blocking_mask = self._apply_comb_filter_and_detect_blocking(ecg_signal)

        # Same absolute drift cap for every harmonic, matching the original single-tone design
        max_dw = 2 * np.pi * 4.0 / self.fs

        for k in range(n_samples):
            ref_sin = np.empty(num_harmonics)
            ref_cos = np.empty(num_harmonics)
            x_hat = 0.0
            for h_idx in range(num_harmonics):
                arg_h = self.w_n_list[h_idx] * k + theta_phi[h_idx]
                ref_sin[h_idx] = np.sin(arg_h)
                ref_cos[h_idx] = np.cos(arg_h)
                # Estimated interference based on this harmonic's current amplitude
                x_hat += theta_a[h_idx] * ref_sin[h_idx]

            d_val = ecg_signal[k]
            e_val = d_val - x_hat
            e_output[k] = e_val

            e_w = np.empty(num_harmonics)
            y_sin_w = np.empty(num_harmonics)
            y_cos_w = np.empty(num_harmonics)
            for h_idx in range(num_harmonics):
                b_h, a_h = self.b_err_list[h_idx], self.a_err_list[h_idx]
                e_w[h_idx], zi_e[h_idx] = self._lfilter_step(b_h, a_h, e_val, zi_e[h_idx])
                y_sin_w[h_idx], zi_y_sin[h_idx] = self._lfilter_step(b_h, a_h, ref_sin[h_idx], zi_y_sin[h_idx])
                y_cos_w[h_idx], zi_y_cos[h_idx] = self._lfilter_step(b_h, a_h, ref_cos[h_idx], zi_y_cos[h_idx])

            if not blocking_mask[k]:
                # Every harmonic adapts its own amplitude, frequency and phase from its own
                # isolated error/reference signals -- if e_w is correlated with that harmonic's
                # sine, its amplitude is too low/high; if correlated with its cosine, its phase
                # needs adjustment.
                for h_idx in range(num_harmonics):
                    alpha = 1.0 / theta_a[h_idx] if theta_a[h_idx] > 1e-6 else 1.0

                    eta_a = e_w[h_idx] * y_sin_w[h_idx]
                    eta_phi = e_w[h_idx] * (alpha * y_cos_w[h_idx])

                    theta_a_next = theta_a[h_idx] + self.K_a * eta_a
                    if theta_a_next < 0:
                        theta_a_next = 0.0

                    theta_dw_next = theta_dw[h_idx] + self.K_dw_list[h_idx] * eta_phi
                    theta_dw_next = np.clip(theta_dw_next, -max_dw, max_dw)

                    theta_phi_next = theta_phi[h_idx] + self.K_phi_list[h_idx] * eta_phi + theta_dw[h_idx]

                    theta_a[h_idx] = theta_a_next
                    theta_dw[h_idx] = theta_dw_next
                    theta_phi[h_idx] = theta_phi_next

        return e_output