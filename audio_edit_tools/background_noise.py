import numpy as np
from scipy import signal

class CrowdNoise:
    """Add crowd background noise to audio"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, snr_db):
        """
        Add crowd noise with specified SNR
        Args:
            audio_data: numpy array of audio data
            snr_db: signal-to-noise ratio in dB range [5, 40]
        Returns:
            noisy audio data
        """
        # Generate crowd noise
        noise = np.random.normal(0, 1, len(audio_data))
        b, a = signal.butter(3, [0.1, 0.3], 'bandpass')
        noise = signal.filtfilt(b, a, noise)
        
        # Normalize noise
        noise = noise / np.max(np.abs(noise))
        
        # Calculate signal power
        signal_power = np.mean(audio_data ** 2)
        
        # Calculate noise scale based on SNR
        noise_scale = np.sqrt(signal_power / (10 ** (snr_db / 10)))
        
        # Add scaled noise to signal
        noisy_audio = audio_data + noise_scale * noise
        
        # Normalize to prevent clipping
        max_amp = np.max(np.abs(noisy_audio))
        if max_amp > 1.0:
            noisy_audio = noisy_audio / max_amp
            
        return noisy_audio 