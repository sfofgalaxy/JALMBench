import librosa

class TimeStretch:
    """Time stretch audio using librosa"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, speed_factor):
        """
        Adjust audio playback speed while maintaining pitch
        Args:
            audio_data: numpy array of audio data
            speed_factor: speed factor in range [0.7, 1.5]
        Returns:
            processed audio data
        """
        return librosa.effects.time_stretch(audio_data, rate=speed_factor) 