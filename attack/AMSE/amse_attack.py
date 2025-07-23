import os
import itertools
import random
import json
from typing import List, Dict, Optional, Tuple

from .audio_edit import AudioEditingToolbox

class AMSEAttack:
    def __init__(self):
        self.toolbox = AudioEditingToolbox()
        self.params = {
            'tone': [-8, -4, 4, 8],  # semitones for tone adjustment
            'amplification': [2, 5, 10],  # amplification factors
            'intonation': ['low', 'medium', 'high'],  # intonation types
            'speed': [0.5, 1.5],  # speed factors
            'noise': ['crowd', 'machine', 'white'],  # noise types
            'accent': ['african', 'asian', 'caucasian']  # accent types
        }
    
    def _load_original_texts(self, input_dir: str) -> Dict[str, Optional[str]]:
        """
        Load original texts from data.json if it exists
        
        Args:
            input_dir (str): Directory containing data.json
            
        Returns:
            Dict[str, Optional[str]]: Dictionary mapping file IDs to original texts
        """
        original_texts = {}
        data_json_path = os.path.join(input_dir, "data.json")
        
        if os.path.exists(data_json_path):
            try:
                with open(data_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    original_texts = {item['id']: item.get('original_text') for item in json_data}
            except Exception as e:
                print(f"Warning: Failed to load data.json: {e}")
        
        return original_texts

    def get_all_combinations(self) -> List[Dict]:
        """Get all possible combinations of audio editing parameters"""
        keys = self.params.keys()
        values = self.params.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def process_single_audio(self, audio_file: str, output_dir: str) -> List[str]:
        """
        Process a single audio file with all possible combinations of audio editing parameters
        
        Args:
            audio_file (str): Path to input audio file
            output_dir (str): Directory to save output files
            
        Returns:
            List[str]: List of paths to generated audio files
        """
        combinations = self.get_all_combinations()
        output_files = []
        
        # Create temp directory for this audio file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        temp_dir = os.path.join(output_dir, f"temp_{base_name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, params in enumerate(combinations):
            output_file = os.path.join(temp_dir, f"{base_name}_attempt_{i+1}.wav")
            temp_file = os.path.join(temp_dir, f"{base_name}_temp.wav")
            
            # Apply transformations in sequence
            current_file = audio_file
            
            # 1. Accent conversion (if different from original)
            self.toolbox.accent_conversion(current_file, temp_file, params['accent'])
            current_file = temp_file
            
            # 2. Tone adjustment
            self.toolbox.tone_adjustment(current_file, temp_file, params['tone'])
            current_file = temp_file
            
            # 3. Emphasis (apply to random segment)
            audio_length = self.toolbox.get_audio_length(current_file)
            segment_start = random.uniform(0, audio_length * 0.7)  # Ensure segment fits within audio
            segments = [(segment_start, segment_start + audio_length * 0.3)]  # Use 30% of audio length
            self.toolbox.emphasis(current_file, temp_file, segments, params['amplification'])
            current_file = temp_file
            
            # 4. Intonation
            self.toolbox.intonation_adjustment(current_file, temp_file, params['intonation'])
            current_file = temp_file
            
            # 5. Speed
            self.toolbox.speed_change(current_file, temp_file, params['speed'])
            current_file = temp_file
            
            # 6. Noise
            self.toolbox.noise_injection(current_file, output_file, params['noise'])
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            output_files.append(output_file)
        
        return output_files

    def process_audio_folder(self, input_dir: str, output_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, Optional[str]]]:
        """
        Process all audio files in a folder with AMSE attack
        
        Args:
            input_dir (str): Directory containing input audio files
            output_dir (str): Directory to save output files
            
        Returns:
            Tuple[Dict[str, List[str]], Dict[str, Optional[str]]]: 
                - Dictionary mapping original file IDs to lists of attack attempt files
                - Dictionary mapping file IDs to original texts
        """
        attack_results = {}
        
        # Load original texts from data.json if it exists
        original_texts = self._load_original_texts(input_dir)
        
        # Get all audio files
        audio_files = []
        for ext in ['mp3', 'wav']:
            audio_files.extend([f for f in os.listdir(input_dir) if f.endswith(f".{ext}")])
        
        for audio_file in audio_files:
            file_id = os.path.splitext(audio_file)[0]
            input_path = os.path.join(input_dir, audio_file)
            attack_files = self.process_single_audio(input_path, output_dir)
            attack_results[file_id] = attack_files
        
        return attack_results, original_texts 