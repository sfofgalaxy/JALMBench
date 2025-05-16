import ffmpeg
import soundfile as sf
import tempfile
import os

class OpusCodec:
    """Simulate audio compression and decompression using Opus codec"""
    def __init__(self, bitrate=64):
        self.bitrate = bitrate  # kbps

    def process(self, audio_data, sample_rate):
        """
        Input audio data, return audio after Opus encode-decode
        audio_data: numpy array, float32, range [-1, 1]
        sample_rate: sample rate
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_wav:
            sf.write(f_wav.name, audio_data, sample_rate)
            wav_path = f_wav.name
        opus_path = wav_path.replace('.wav', '.opus')
        ffmpeg.input(wav_path).output(opus_path, audio_bitrate=f'{self.bitrate}k', acodec='libopus').run(quiet=True, overwrite_output=True)
        decoded_path = wav_path.replace('.wav', '_decoded.wav')
        ffmpeg.input(opus_path).output(decoded_path, acodec='pcm_s16le').run(quiet=True, overwrite_output=True)
        decoded_audio, _ = sf.read(decoded_path, dtype='float32')
        os.remove(wav_path)
        os.remove(opus_path)
        os.remove(decoded_path)
        return decoded_audio 