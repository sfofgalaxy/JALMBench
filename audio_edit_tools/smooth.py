import numpy as np
from scipy import signal

class Smooth:
    """Apply smoothing to audio using Gaussian convolution"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, window_size):
        """
        Smooth audio with 1D Gaussian convolution
        Args:
            audio_data: numpy array of audio data
            window_size: size of the smoothing window range [6, 22]
        Returns:
            smoothed audio data
        """
        # Create Gaussian window
        window = signal.windows.gaussian(window_size, std=window_size/6.0)
        
        # Normalize window
        window = window / np.sum(window)
        
        # Apply convolution
        smoothed_audio = signal.convolve(audio_data, window, mode='same')
        
        return smoothed_audio 