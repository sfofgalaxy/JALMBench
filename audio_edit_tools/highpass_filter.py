from scipy import signal

class HighpassFilter:
    """Apply highpass filter to audio"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, cutoff_ratio):
        """
        Apply highpass filter to remove low frequencies
        Args:
            audio_data: numpy array of audio data
            cutoff_ratio: cutoff frequency ratio (0-1) range [0.1, 0.5]
        Returns:
            filtered audio data
        """
        # Calculate cutoff frequency
        cutoff_freq = cutoff_ratio * (self.sample_rate / 2)
        
        # Design filter
        b, a = signal.butter(5, cutoff_freq / (self.sample_rate / 2), 'highpass')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio 