import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List

class FECGExtractor:
    """
    Fetal ECG (FECG) Extractor class.
    Implements the sequential analysis method for detecting fetal QRS complexes
    and extracting the FECG using multi-channel QRS enhancement (PCA) and
    synchronous averaging.
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
        half_width = max(1, width_samples // 2)
        # Minimum distance based on physiological limit of 3.3 Hz (fetal tachycardia) -> ~0.3s
        min_dist = int(0.3 * self.fs)

        # Step 2: Rough peak detection across the whole signal, so the
        # template isn't built from any single beat that might not be
        # representative of the true fetal QRS shape
        squared_signal = enhanced_signal ** 2
        rough_thresh = np.mean(squared_signal) + 2 * np.std(squared_signal)
        rough_peaks = self._find_peaks(squared_signal, rough_thresh, min_dist)

        if len(rough_peaks) == 0:
            return np.array([])

        segments = []
        for p in rough_peaks:
            start = p - half_width
            end = p + half_width
            if start < 0 or end > len(enhanced_signal):
                continue
            segments.append(enhanced_signal[start:end])

        if not segments:
            return rough_peaks

        # A median across many rough peaks is a far more robust template than
        # a single beat, since it isn't thrown off by one noisy or non-fetal peak
        template = np.median(np.array(segments), axis=0)

        # Step 3: Cross-correlation for optimal detection
        cross_corr = np.correlate(enhanced_signal, template, mode='same')
        threshold = np.mean(cross_corr) + 1.5 * np.std(cross_corr)
        peaks = self._find_peaks(cross_corr, threshold, min_dist)

        # Step 4: Refine each peak to the exact local maximum of the enhanced
        # signal, since the cross-correlation peak can be off by a few samples.
        refined_peaks = []
        for p in peaks:
            start = max(0, p - half_width)
            end = min(len(enhanced_signal), p + half_width + 1)
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