import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os
from TTS.api import TTS
import torch
from typing import List, Dict, Tuple, Optional

class AudioEditingToolbox:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self._init_tts_model()
        
        # Define reference voices for each accent from CREMA-D dataset
        self.accent_references = {
            'african': "attack/AMSE/accent/african_reference.wav",
            'asian': "attack/AMSE/accent/asian_reference.wav",
            'caucasian': "attack/AMSE/accent/caucasian_reference.wav"
        }

    def _init_tts_model(self):
        """Initialize TTS model for voice conversion"""
        try:
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"Warning: Failed to initialize TTS model: {e}")
            self.tts_model = None

    def get_audio_length(self, audio_file: str) -> float:
        """Get the length of audio file in seconds"""
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        return len(audio) / sr

    def tone_adjustment(self, input_file: str, output_file: str, semitones: int):
        """
        Adjust the pitch of audio
        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            semitones: Number of semitones to shift, ∈ {-8, -4, +4, +8}
        """
        audio, sr = librosa.load(input_file, sr=self.sample_rate)
        adjusted_audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=semitones)
        sf.write(output_file, adjusted_audio, self.sample_rate)

    def emphasis(self, input_file: str, output_file: str, segments: List[Tuple[float, float]], amplification_factor: float):
        """
        Increase volume of specific segments to simulate emphasis
        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            segments: List of (start_time, end_time) tuples
            amplification_factor: Volume multiplier, k ∈ {2, 5, 10}
        """
        audio, sr = librosa.load(input_file, sr=self.sample_rate)
        audio_emphasized = audio.copy()

        for start_time, end_time in segments:
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            audio_emphasized[start_idx:end_idx] *= amplification_factor

        # Normalize to prevent clipping
        audio_emphasized = audio_emphasized / np.max(np.abs(audio_emphasized)) * np.max(np.abs(audio))
        sf.write(output_file, audio_emphasized, self.sample_rate)

    def intonation_adjustment(self, input_file: str, output_file: str, interval_type: str = "medium"):
        """
        Implement dynamic pitch modification to simulate natural prosody patterns
        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            interval_type: Type of intervals to use, ∈ {"low", "medium", "high"}
        """
        audio, sr = librosa.load(input_file, sr=self.sample_rate)

        intervals = {
            "low": [0, 2, 4, 6],      # [0,2,4,6]
            "medium": [0, 3, 6, 9],   # [0,3,6,9]
            "high": [0, 4, 8, 12]     # [0,4,8,12]
        }

        selected_intervals = intervals[interval_type]
        num_segments = len(selected_intervals)
        segment_length = len(audio) // num_segments
        modified_audio = np.zeros_like(audio)

        for i, interval in enumerate(selected_intervals):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < num_segments - 1 else len(audio)
            segment = audio[start_idx:end_idx]
            modified_segment = librosa.effects.pitch_shift(segment, sr=self.sample_rate, n_steps=interval)
            modified_audio[start_idx:end_idx] = modified_segment[:len(audio[start_idx:end_idx])]

        sf.write(output_file, modified_audio, self.sample_rate)

    def speed_change(self, input_file: str, output_file: str, speed_factor: float):
        """
        Adjust audio playback rate while maintaining pitch
        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            speed_factor: Speed multiplier, β ∈ {0.5, 1.5}
        """
        audio, sr = librosa.load(input_file, sr=self.sample_rate)
        adjusted_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        sf.write(output_file, adjusted_audio, self.sample_rate)

    def noise_injection(self, input_file: str, output_file: str, noise_type: str = "white", noise_level: float = 0.05):
        """
        Inject background noise into original audio
        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            noise_type: Type of noise, ∈ {"crowd", "machine", "white"}
            noise_level: Noise intensity level
        """
        audio, sr = librosa.load(input_file, sr=self.sample_rate)
        
        # Update noise file paths to be relative to the attack folder
        noise_files = {
            "white": os.path.join("attack", "AMSE", "injection", "white.wav"),
            "crowd": os.path.join("attack", "AMSE", "injection", "crowd.wav"),
            "machine": os.path.join("attack", "AMSE", "injection", "mechanism.wav")
        }

        if noise_type in noise_files and os.path.exists(noise_files[noise_type]):
            # Load noise file
            noise, noise_sr = librosa.load(noise_files[noise_type], sr=self.sample_rate)
            
            # Handle noise length mismatch
            if len(noise) < len(audio):
                repetitions = int(np.ceil(len(audio) / len(noise)))
                noise = np.tile(noise, repetitions)[:len(audio)]
            else:
                noise = noise[:len(audio)]
        else:
            # Fallback to synthetic noise
            print(f"Warning: Noise file {noise_files.get(noise_type, 'unknown')} not found, using synthetic noise.")
            if noise_type == "white":
                noise = np.random.normal(0, 1, len(audio))
            elif noise_type == "crowd":
                noise = np.random.normal(0, 1, len(audio))
                b, a = signal.butter(3, [0.1, 0.3], 'bandpass')
                noise = signal.filtfilt(b, a, noise)
            else:  # machine noise
                noise = np.random.normal(0, 1, len(audio))
                b, a = signal.butter(3, 0.2, 'lowpass')
                noise = signal.filtfilt(b, a, noise)
                # Add harmonic components
                for harmonic in [2, 3, 5]:
                    noise += np.sin(np.linspace(0, harmonic * 2 * np.pi * 10, len(noise))) * 0.2

        # Normalize and add noise
        noise = noise / np.max(np.abs(noise))
        noisy_audio = audio + noise_level * noise
        adjusted_audio = noisy_audio / np.max(np.abs(noisy_audio)) * np.max(np.abs(audio))
        sf.write(output_file, adjusted_audio, self.sample_rate)

    def accent_conversion(self, input_file: str, output_file: str, target_accent: str):
        """
        Convert the accent of the audio using XTTS v2 model
        Args:
            input_file: Input audio file path
            output_file: Output audio file path
            target_accent: Target accent, ∈ {"african", "asian", "caucasian"}
        """
        if self.tts_model is None:
            raise RuntimeError("TTS model not initialized")
            
        if target_accent not in self.accent_references:
            raise ValueError(f"Unsupported accent: {target_accent}")
            
        try:
            # Use the reference voice from CREMA-D dataset for the target accent
            reference_file = self.accent_references[target_accent]
            if not os.path.exists(reference_file):
                raise FileNotFoundError(f"Reference file not found: {reference_file}")
                
            # Perform voice conversion using the reference voice
            self.tts_model.voice_conversion_to_file(
                source_wav=input_file,
                target_wav=reference_file,
                file_path=output_file
            )
        except Exception as e:
            print(f"Error in accent conversion: {e}")
            # Copy input file to output as fallback
            import shutil
            shutil.copy2(input_file, output_file)