import numpy as np

class GaussianNoise:
    """Add Gaussian noise to audio"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, snr_db):
        """
        Add Gaussian noise with specified SNR
        Args:
            audio_data: numpy array of audio data
            snr_db: signal-to-noise ratio in dB range [5, 40]
        Returns:
            noisy audio data
        """
        # Calculate signal power
        signal_power = np.mean(audio_data ** 2)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio_data))
        
        # Add noise to signal
        noisy_audio = audio_data + noise
        
        # Normalize to prevent clipping
        max_amp = np.max(np.abs(noisy_audio))
        if max_amp > 1.0:
            noisy_audio = noisy_audio / max_amp
            
        return noisy_audio 