import numpy as np
from pydub import AudioSegment
import sox
import augment
from audio_utils import WAVFile, float_to_wav, wav_to_float
import scipy.io.wavfile as wavfile
import torch
import random
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
import time

TARGET_SAMPLING_RATE = 16000
def quick_choice(list: list):
    return list[random.randint(0, len(list) - 1)]

def apply_shift_pitch(wav: WAVFile, pitch_shift: int) -> WAVFile:
    """
    Shift the pitch of the audio by the given amount (-300 to 300 is recommended).
    """
    x = wav.audio
    x_orig = x.copy()

    x = wav_to_float(x)
    x = np.expand_dims(x, axis=0)

    src_info = {
        "channels": x.shape[0],
        "length": x.shape[1],
        "precision": 32,
        "rate": TARGET_SAMPLING_RATE,
        "bits_per_sample": 32,
    }

    target_info = {
        "channels": 1,
        "length": x.shape[1],
        "precision": 32,
        "rate": TARGET_SAMPLING_RATE,
        "bits_per_sample": 32,
    }

    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_tensor = (
        augment.EffectChain()
        .pitch("-q", pitch_shift)
        .rate(TARGET_SAMPLING_RATE)
        .apply(x_tensor, src_info=src_info, target_info=target_info)
    )

    x = x_tensor.numpy()

    # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
    # and the effect chain includes eg `pitch`
    if np.isnan(x).any() or np.isinf(x).any():
        return x_orig

    x = np.squeeze(x, axis=0)
    x = float_to_wav(x)

    x = WAVFile(x)

    return x

def mix_audio_with_snr(target_audio_path, overlay_audio_paths, snrs, start_times=None, output_path=None):
    if isinstance(overlay_audio_paths, str):
        overlay_audio_paths = [overlay_audio_paths]
    if isinstance(snrs, (int, float)):
        snrs = [snrs]
    if start_times is None:
        start_times = [0] * len(overlay_audio_paths)
    
    target = AudioSegment.from_wav(target_audio_path)
    target_len = len(target)
    
    # Create silent base of target length
    mixed = target
    
    for overlay_path, snr_db, start_time in zip(overlay_audio_paths, snrs, start_times):
        overlay = AudioSegment.from_wav(overlay_path)
        
        # Convert start_time from seconds to milliseconds
        start_ms = int(start_time * 1000)
        
        # Calculate gain needed for desired SNR
        target_rms = target.rms
        overlay_rms = overlay.rms
        
        if target_rms == 0 or overlay_rms == 0:
            continue
            
        current_snr = 20 * np.log10(target_rms / overlay_rms)
        gain_needed = current_snr - snr_db
        overlay = overlay.apply_gain(gain_needed)
        
        # Handle overlay length
        required_len = target_len - start_ms
        if len(overlay) > required_len:
            overlay = overlay[:required_len]
        elif len(overlay) < required_len:
            times_to_loop = int(np.ceil(required_len / len(overlay)))
            overlay = overlay * times_to_loop
            overlay = overlay[:required_len]
        
        # Create silent segment for start_time
        if start_ms > 0:
            silence = AudioSegment.silent(duration=start_ms)
            overlay = silence + overlay
        
        # Simple overlay without crossfade
        mixed = mixed.overlay(overlay)
    
    if output_path:
        mixed.export(output_path, format='wav')
    return mixed

def apply_augmentations(input_file, output_file, seed):
    """Apply augmentations with controlled SNR for different noise types"""
    # Generate augmentation values using Gaussian distribution (σ = 0.25)
    np.random.seed(seed)
    augment_values = np.random.normal(0, 0.25, 6)
    
    # Scale values according to Equation 1
    speed_clip = 3
    log_base = np.log(speed_clip)
    mapped_value = augment_values[0] * log_base
    speed_factor = np.clip(np.exp(mapped_value), 1/speed_clip, speed_clip)
    pitch_shift = np.clip(2000 * augment_values[1], -2000, 2000)
    volume_factor = np.clip(10 ** (3 * augment_values[2]), 0.001, 1000)
    snr_speech = 20 * augment_values[3] + 10
    snr_noise = 20 * augment_values[4] + 10
    snr_music = 20 * augment_values[5] + 10
    
    # 1. Speed using tempo
    tfm = sox.Transformer()
    tfm.tempo(speed_factor) 
    tfm.build(input_file, f'cache/temp_speed_{seed}.wav')
    
    # 2. Pitch using WavAugment
    wav_file = WAVFile.from_file(f'cache/temp_speed_{seed}.wav')
    pitched_wav = apply_shift_pitch(wav_file, pitch_shift)    
    wavfile.write(f'cache/temp_pitch_{seed}.wav', TARGET_SAMPLING_RATE, pitched_wav.audio)
    
    # 3. Speech overlay
    apply_speech(f'cache/temp_pitch_{seed}.wav', f'cache/temp_speech_{seed}.wav', snr_speech)
    
    # 4. Noise overlay
    apply_noise(f'cache/temp_speech_{seed}.wav', f'cache/temp_noise_{seed}.wav', snr_noise)
    
    # 5. Volume adjustment
    audio = AudioSegment.from_file(f'cache/temp_noise_{seed}.wav')
    audio = audio + 20 * np.log10(volume_factor)
    audio.export(f'cache/temp_volume_{seed}.wav', format='wav')
    
    # 6. Music overlay
    apply_music(f'cache/temp_volume_{seed}.wav', output_file, snr_music)

    # remove temp files
    temp_files = [
        f'cache/temp_speed_{seed}.wav',
        f'cache/temp_pitch_{seed}.wav',
        f'cache/temp_speech_{seed}.wav',
        f'cache/temp_noise_{seed}.wav',
        f'cache/temp_volume_{seed}.wav'
    ]
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Error removing temporary file {temp_file}: {e}")

def apply_noise(input_path: str, output_path: str, snr: int = None):
    """Apply background noise to the audio"""
    wav = WAVFile.from_file(input_path)
    dur = round(wav.audio.shape[0] / TARGET_SAMPLING_RATE, 2)
    sounds, start_times, snrs = [], [], []
    total_noise_dur = 0
    
    noise_path = 'cache/noise-sound-bible-0083.wav'
    while total_noise_dur < dur:
        sounds.append(noise_path)
        start_times.append(total_noise_dur)
        chosen_snr = snr if snr is not None else quick_choice([15, 10])
        snrs.append(chosen_snr)
        total_noise_dur += 1

    mix_audio_with_snr(input_path, sounds, snrs, start_times, output_path)

def apply_music(input_path: str, output_path: str, snr: int = None):
    chosen_snr = snr if snr is not None else quick_choice([15, 10, 8, 5])
    mix_audio_with_snr(input_path, 'cache/music-fma-wa-0045.wav', chosen_snr, [0], output_path)

def apply_speech(input_path: str, output_path: str, snr: int = None):
    chosen_snr = snr if snr is not None else quick_choice([20, 17])
    mix_audio_with_snr(input_path, 'cache/speech-librivox-0142.wav', chosen_snr, [0], output_path)

def _split_into_chunks(lst: List, n: int) -> List[List]:
    """Split a list into n roughly equal chunks
    
    Args:
        lst (List): List to split
        n (int): Number of chunks
        
    Returns:
        List[List]: List of chunks
    """
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    start = 0
    chunks = []
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

def _process_chunk(args: Tuple[str, List[int], str]) -> None:
    """Process a chunk of augmentations in parallel
    
    Args:
        args (Tuple[str, List[int], str]): Tuple containing:
            - input_file: Path to input audio file
            - indices: List of indices to process
            - output_dir: Directory to save output files
    """
    input_file, indices, output_dir = args
    for i in indices:
        output_file = os.path.join(output_dir, f'aug_{i:04d}.wav')
        try:
            apply_augmentations(input_file, output_file, i)
        except Exception as e:
            print(f"Error processing augmentation {i} for {input_file}: {e}")

def generate_augmentations_parallel(input_file: str, output_dir: str, num_augmentations: int = 30, 
                                  num_workers: int = None) -> List[str]:
    """Generate multiple augmented versions of an audio file in parallel
    
    Args:
        input_file (str): Path to input audio file
        output_dir (str): Directory to save output files
        num_augmentations (int): Number of augmented versions to generate
        num_workers (int): Number of parallel workers. If None, uses CPU count
        
    Returns:
        List[str]: List of paths to generated audio files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 60)  # Cap at 60 workers
    
    # Generate indices and split into chunks
    indices = list(range(1, num_augmentations + 1))
    chunks = _split_into_chunks(indices, num_workers)
    
    # Prepare arguments for each worker
    work_items = [(input_file, chunk, output_dir) for chunk in chunks]
    
    # Process chunks in parallel
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(_process_chunk, work_items)
    end_time = time.time()
    
    print(f"Time taken for {input_file}: {end_time - start_time:.2f} seconds")
    
    # Collect and return paths of successfully generated files
    output_files = []
    for i in range(1, num_augmentations + 1):
        output_file = os.path.join(output_dir, f'aug_{i:04d}.wav')
        if os.path.exists(output_file):
            output_files.append(output_file)
    
    return output_files
