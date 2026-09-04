import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List

class FECGExtractor:
    """
    Fetal ECG (FECG) Extractor class.
    Implements the sequential analysis method for detecting fetal QRS complexes
    and extracting the FECG using multi-channel QRS enhancement (PCA) and
    synchronous averaging, as described in the reference paper.
    """

    def __init__(self, fs: float = 2000.0):
        """
        Constructor for FECGExtractor.

        Args:
            fs (float): Sampling frequency of the signal in Hz. Defaults to 2000.0.
                        High sampling frequency is needed for precise alignment.
        """
        self.fs = fs

    def enhance_fetal_qrs(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Enhances the fetal QRS complexes using a multi-channel approach.
        Exploits the large inter-channel correlation of the ECG components by
        normalising the channel variance and extracting the first principal
        component (PCA), which represents the linear combination with the largest SNR.

        Args:
            signal_matrix (np.ndarray): Multi-channel signal matrix (n_samples, n_channels)
                                        after MECG cancellation.

        Returns:
            np.ndarray: A 1D array representing the first principal component (enhanced QRS).
        """
        if signal_matrix.size == 0:
            return np.array([])
            
        # Normalise channel variance
        std_dev = np.std(signal_matrix, axis=0)
        # Avoid division by zero for completely flat channels
        std_dev[std_dev == 0] = 1.0
        normalised_signal = signal_matrix / std_dev

        # Centre the signal
        centred_signal = normalised_signal - np.mean(normalised_signal, axis=0)

        # Perform PCA and get the first principal component
        pca = PCA(n_components=1)
        first_pc = pca.fit_transform(centred_signal).flatten()

        # Ensure a consistent polarity so that fetal R-peaks are always positive
        # maxima (the PCA sign is otherwise arbitrary and depends on the input).
        search_window = min(len(first_pc), int(3 * self.fs))
        if search_window > 0 and np.max(first_pc[:search_window]) < np.abs(np.min(first_pc[:search_window])):
            first_pc = -first_pc

        return first_pc

    def detect_fetal_qrs(self, signal_matrix: np.ndarray, template_width_sec: float = 0.05) -> np.ndarray:
        """
        Detects fetal QRS complexes using multi-channel QRS enhancement followed
        by Cross-correlation / Matched Filter on the enhanced signal.

        Args:
            signal_matrix (np.ndarray): Multi-channel signal matrix (n_samples, n_channels).
            template_width_sec (float): Width of the QRS template in seconds. Defaults to 0.05.

        Returns:
            np.ndarray: Array of detected fetal R-peak indices.
        """
        if len(signal_matrix) == 0:
            return np.array([])
            
        # Step 1: Multi-channel QRS enhancement via PCA
        enhanced_signal = self.enhance_fetal_qrs(signal_matrix)
        
        if len(enhanced_signal) == 0:
            return np.array([])
            
        width_samples = int(template_width_sec * self.fs)

        # Search window for extracting the initial matched filter template
        search_window = min(len(enhanced_signal), int(3 * self.fs))
        if search_window == 0:
            return np.array([])

        initial_peak = int(np.argmax(enhanced_signal[:search_window]))
        half_width = width_samples // 2
        
        start_idx = max(0, initial_peak - half_width)
        end_idx = min(len(enhanced_signal), initial_peak + half_width)
        template = enhanced_signal[start_idx:end_idx]
        
        if len(template) == 0:
            return np.array([])
            
        # Step 2: Cross-correlation for optimal detection
        cross_corr = np.correlate(enhanced_signal, template, mode='same')
        
        # Thresholding and peak extraction
        threshold = 0.5 * np.max(cross_corr)
        # Minimum distance based on physiological limit of 3.3 Hz (fetal tachycardia) -> ~0.3s
        min_dist = int(0.3 * self.fs) 
        
        peaks = self._find_peaks(cross_corr, threshold, min_dist)

        # Step 3: Refine each peak to the exact local maximum of the enhanced
        # signal, since the cross-correlation peak can be off by a few samples.
        refine_half_win = max(1, width_samples // 2)
        refined_peaks = []
        for p in peaks:
            start = max(0, p - refine_half_win)
            end = min(len(enhanced_signal), p + refine_half_win + 1)
            refined_peaks.append(start + int(np.argmax(enhanced_signal[start:end])))

        return np.array(refined_peaks, dtype=int)

    def _find_peaks(self, signal: np.ndarray, threshold: float, min_dist: int) -> np.ndarray:
        """
        Helper method to find peaks in a 1D signal based on a threshold and minimum distance.

        Args:
            signal (np.ndarray): 1D signal (usually cross-correlation output).
            threshold (float): Minimum amplitude to be considered a peak.
            min_dist (int): Minimum distance between consecutive peaks in samples.

        Returns:
            np.ndarray: Array of peak indices.
        """
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
        Calculates Fetal Heart Rate (FHR) in beats per minute (bpm) based on detected peaks.
        Filters out non-physiological values outside the ~78 bpm (1.3 Hz) to ~198 bpm (3.3 Hz) bounds.

        Args:
            fetal_peaks (np.ndarray): Array of detected fetal R-peak indices.

        Returns:
            np.ndarray: Array of valid FHR values in bpm.
        """
        if len(fetal_peaks) < 2:
            return np.array([])
            
        rr_intervals_sec = np.diff(fetal_peaks) / self.fs
        
        # Filter RR intervals based on physiological constraints (1.3 Hz to 3.3 Hz -> 0.25s to 0.77s)
        valid_rr = rr_intervals_sec[(rr_intervals_sec > 0.25) & (rr_intervals_sec < 0.77)]
        
        if len(valid_rr) == 0:
            return np.array([])
            
        fhr = 60.0 / valid_rr
        return fhr

    def synchronous_averaging(self, signal_matrix: np.ndarray, peaks: np.ndarray, 
                              num_beats: int = 150, window_size_sec: float = 0.4) -> Optional[np.ndarray]:
        """
        Averages a set number of fetal beats (default 150) to improve Signal-to-Noise Ratio (SNR).
        Operates on the multi-channel signal to produce an independent FECG complex estimate 
        for each channel.

        Args:
            signal_matrix (np.ndarray): Multi-channel signal matrix (n_samples, n_channels).
            peaks (np.ndarray): Array of detected fetal R-peak indices.
            num_beats (int): Number of consecutive beats to average. Defaults to 150.
            window_size_sec (float): Size of the extraction window around the peak in seconds. Defaults to 0.4.

        Returns:
            Optional[np.ndarray]: Multi-channel averaged FECG template of shape (window_samples, n_channels),
                                  or None if not enough valid peaks exist.
        """
        if len(peaks) == 0 or len(signal_matrix) == 0:
            return None
            
        half_window = int((window_size_sec / 2) * self.fs)
        beats = []
        
        # Take up to the last num_beats
        selected_peaks = peaks[-num_beats:] if len(peaks) > num_beats else peaks
        
        for p in selected_peaks:
            start = p - half_window
            end = p + half_window
            
            # Check boundaries
            if start >= 0 and end <= len(signal_matrix):
                beats.append(signal_matrix[start:end, :])
                
        if not beats:
            return None
            
        beats_array = np.array(beats)
        
        # beats_array shape: (n_beats, window_samples, n_channels)
        # Resulting fecg_template shape: (window_samples, n_channels)
        fecg_template = np.mean(beats_array, axis=0)
        
        return fecg_template