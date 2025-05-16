import numpy as np

class Quantization:
    """Quantize audio signal to specified bit levels"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, bit_levels):
        """
        Quantize audio to specified bit levels
        Args:
            audio_data: numpy array of audio data
            bit_levels: number of discrete levels range [4, 64]
        Returns:
            quantized audio data
        """
        # Ensure audio is in [-1, 1] range
        audio_norm = audio_data / np.max(np.abs(audio_data))
        
        # Calculate step size
        steps = 2 ** bit_levels
        step_size = 2.0 / steps
        
        # Quantize the signal
        quantized = np.round(audio_norm / step_size) * step_size
        
        # Clip to [-1, 1]
        quantized = np.clip(quantized, -1.0, 1.0)
        
        return quantized 