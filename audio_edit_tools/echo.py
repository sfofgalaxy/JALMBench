import numpy as np

class Echo:
    """Add echo effect to audio"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, delay_sec, decay=0.5):
        """
        Add echo effect to audio
        Args:
            audio_data: numpy array of audio data
            delay_sec: delay time in seconds range [0.1, 0.9]
            decay: echo decay factor (0-1)
        Returns:
            audio with echo effect
        """
        # Calculate delay in samples
        delay_samples = int(delay_sec * self.sample_rate)
        
        # Create output array
        output = np.zeros(len(audio_data) + delay_samples)
        
        # Add original signal
        output[:len(audio_data)] += audio_data
        
        # Add delayed signal
        output[delay_samples:delay_samples+len(audio_data)] += decay * audio_data
        
        # Trim to original length
        output = output[:len(audio_data)]
        
        # Normalize to prevent clipping
        max_amp = np.max(np.abs(output))
        if max_amp > 1.0:
            output = output / max_amp
            
        return output 