import numpy as np
from scipy import signal

class SignalResampler:
    def __init__(self, original_fs: float = 1000.0, target_fs: float = 2000.0):
        """
        Resampler class to upsample the signal with proper anti-aliasing.

        Args:
            original_fs (float): Native sampling frequency of the signal in Hz. Defaults
                to 1000.0 (this project's standard native/original sampling rate -- pass
                fs_orig explicitly).
            target_fs (float): Sampling frequency to upsample to, in Hz. Defaults to 2000.0.
        """
        if original_fs <= 0 or target_fs <= 0:
            raise ValueError("Frequencies must be strictly positive.")
            
        self.original_fs = original_fs
        self.target_fs = target_fs

    def upsample(self, data: np.ndarray) -> np.ndarray:
        """
        Upsample the data from original_fs to target_fs.
        Uses scipy.signal.resample_poly for robust upsampling with a polyphase anti-aliasing filter.
        """
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise ValueError("Data must be a non-empty numpy array.")
            
        if self.original_fs == self.target_fs:
            return data.copy()
            
        up = int(self.target_fs)
        down = int(self.original_fs)
        
        # resample_poly operates along axis=-1 by default, automatically vectorising across channels
        resampled_data = signal.resample_poly(data, up, down, axis=-1)
        return resampled_data
