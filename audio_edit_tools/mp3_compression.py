import os
import tempfile
import numpy as np
import torchaudio
import torch
import soundfile as sf
from scipy import signal

class MP3Compression:
    """Simulate MP3 compression effects using signal processing"""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process(self, audio_data, bitrate):
        """
        Simulate MP3 compression using downsampling and quantization
        Args:
            audio_data: numpy array of audio data
            bitrate: target bitrate in kbps range [8, 40]
        Returns:
            processed audio data with compression artifacts
        """
        # Calculate effective sample rate based on bitrate
        # Lower bitrate = more aggressive compression
        compression_ratio = 1.0 - (bitrate - 8) / (40 - 8) * 0.7  # 0.3 to 1.0
        
        # 1. Apply lowpass filter (MP3 removes high frequencies)
        # Lower bitrate = lower cutoff frequency
        cutoff_ratio = 0.5 - compression_ratio * 0.35  # 0.15 to 0.5
        cutoff_freq = cutoff_ratio * (self.sample_rate / 2)
        
        # Design filter
        b, a = signal.butter(5, cutoff_freq / (self.sample_rate / 2), 'lowpass')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        # 2. Apply quantization (MP3 quantizes frequency domains)
        # Lower bitrate = fewer bits
        bit_depth = int(16 - compression_ratio * 12)  # 4 to 16 bits
        
        # Ensure audio is in [-1, 1] range
        audio_norm = filtered_audio / np.max(np.abs(filtered_audio))
        
        # Calculate quantization steps
        steps = 2 ** bit_depth
        step_size = 2.0 / steps
        
        # Quantize the signal
        quantized = np.round(audio_norm / step_size) * step_size
        
        # 3. Add some artificial compression artifacts
        # More noticeable at lower bitrates
        if bitrate < 24:
            # Add some pre-echo effect (common in MP3)
            pre_echo_size = int(0.01 * self.sample_rate)  # 10ms pre-echo
            pre_echo = np.zeros_like(quantized)
            pre_echo[pre_echo_size:] = quantized[:-pre_echo_size] * 0.1 * compression_ratio
            quantized = quantized + pre_echo
            
            # Add some frequency masking artifacts
            artifacts = np.random.normal(0, 0.001 * compression_ratio, len(quantized))
            b, a = signal.butter(3, 0.1, 'highpass')
            artifacts = signal.filtfilt(b, a, artifacts)
            quantized = quantized + artifacts
        
        # Normalize to prevent clipping
        processed_audio = quantized / np.max(np.abs(quantized))
        
        return processed_audio

    def process_torchaudio(self, audio_data, bitrate):
        """
        Apply MP3 compression and decompression
        Args:
            audio_data: numpy array of audio data
            bitrate: MP3 bitrate in kbps range [8, 40]
        Returns:
            compressed and decompressed audio data
        """
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
            temp_in_path = temp_in.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
            temp_mp3_path = temp_mp3.name
            
        try:
            # Save input audio to temp file
            sf.write(temp_in_path, audio_data, self.sample_rate)
            
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(temp_in_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Save as MP3 with specified compression
            torchaudio.save(
                temp_mp3_path,
                waveform,
                sample_rate,
                format="mp3",
                compression=bitrate  # bitrate in kbps
            )
            
            # Load compressed audio back
            compressed_waveform, _ = torchaudio.load(temp_mp3_path)
            
            # Convert to numpy array
            processed_audio = compressed_waveform[0].numpy()
            
            # Ensure same length as input
            if len(processed_audio) > len(audio_data):
                processed_audio = processed_audio[:len(audio_data)]
            elif len(processed_audio) < len(audio_data):
                pad_length = len(audio_data) - len(processed_audio)
                processed_audio = np.pad(processed_audio, (0, pad_length))
                
            return processed_audio
            
        finally:
            # Clean up temporary files
            for path in [temp_in_path, temp_mp3_path]:
                if os.path.exists(path):
                    os.unlink(path) 