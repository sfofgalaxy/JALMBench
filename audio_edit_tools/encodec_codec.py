import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

class EncodecCodec:
    """Simulate audio compression and decompression using EnCodec neural codec"""
    def __init__(self, bandwidth=24.0):
        self.bandwidth = bandwidth  # kHz
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.eval()

    def process(self, audio_data, sample_rate):
        """
        Input audio data, return audio after EnCodec encode-decode
        audio_data: numpy array, float32, range [-1, 1]
        sample_rate: sample rate
        """
        wav = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)  # [B, C, T]
        wav = convert_audio(wav, sample_rate, self.model.sample_rate, self.model.channels)
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
            decoded_wav = self.model.decode(encoded_frames)
        decoded_audio = decoded_wav.squeeze().cpu().numpy()
        return decoded_audio 