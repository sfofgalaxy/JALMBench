from .AudioLM import AudioLM
from .LO.predict import Predictor
import numpy as np
import soundfile as sf
import os
import torch
import whisper
from typing import Optional, Union

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2

def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = os.path.join("AdvWave", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = whisper.load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

class LLaMAOmni(AudioLM):
    def __init__(self, model_cache, vocoder_path, config = {}):
        super().__init__()
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.temperature = config.get('temperature', 0.5)
        self.top_p = config.get('top_p', 1.0)
        self.model_cache = model_cache
        self.vocoder_path = vocoder_path
        self.load_model()
    
    def load_model(self):
        """Initialize the predictor"""
        self.predictor = Predictor()
        self.predictor.setup(self.model_cache, self.vocoder_path)
    
    def process_audio(self, audio_path: str, prompt=None, **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        if audio_path == None:
            # Create a 1-second silent audio with a sample rate of 16000
            sample_rate = 16000
            silent_audio = np.zeros(sample_rate * 1)
            os.makedirs('temp', exist_ok=True)
            audio_path = 'temp/none.wav'
            sf.write(audio_path, silent_audio, sample_rate)

        output = self.predictor.predict(
            input_audio=audio_path,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            output_audio_path=kwargs.get('output_audio_path', None)
        )
        return output
