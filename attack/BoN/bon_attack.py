import os
import json
from typing import Dict, List, Tuple, Optional
from .audio_argument import apply_augmentations, generate_augmentations_parallel

class BoNAttack:
    def __init__(self, num_augmentations: int = 30, num_workers: int = None):
        """Initialize BoN (Bag of Noise) attack
        
        Args:
            num_augmentations (int): Number of augmented versions to generate for each audio
            num_workers (int): Number of parallel workers for large-scale generation.
                             If None, uses CPU count. Only used when num_augmentations > 100.
        """
        self.num_augmentations = num_augmentations
        self.num_workers = num_workers
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists for temporary files"""
        os.makedirs("cache", exist_ok=True)
        
    def _generate_augmentations(self, audio_file: str, output_dir: str) -> List[str]:
        """Generate multiple augmented versions of an audio file
        
        Args:
            audio_file (str): Path to input audio file
            output_dir (str): Directory to save augmented files
            
        Returns:
            List[str]: List of paths to generated audio files
        """
        # Create output directory for this audio file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        audio_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Use parallel processing for large numbers of augmentations
        if self.num_augmentations > 100:
            return generate_augmentations_parallel(
                input_file=audio_file,
                output_dir=audio_output_dir,
                num_augmentations=self.num_augmentations,
                num_workers=self.num_workers
            )
        
        # Use sequential processing for smaller numbers
        augmented_files = []
        for i in range(self.num_augmentations):
            output_file = os.path.join(audio_output_dir, f"aug_{i+1:04d}.wav")
            try:
                # Apply 6 random augmentations with controlled SNR
                apply_augmentations(
                    input_file=audio_file,
                    output_file=output_file,
                    seed=i  # Use iteration as seed for reproducibility
                )
                augmented_files.append(output_file)
            except Exception as e:
                print(f"Error generating augmentation {i+1} for {audio_file}: {e}")
                continue
                
        return augmented_files
        
    def process_audio_folder(self, input_dir: str, output_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, Optional[str]]]:
        """Process all audio files in the input folder for BoN attack
        
        Args:
            input_dir (str): Input directory path containing audio files
            output_dir (str): Output directory path for augmented files
            
        Returns:
            Tuple[Dict[str, List[str]], Dict[str, Optional[str]]]: 
                - Mapping from original file ID to list of attack audio files
                - Mapping from file ID to original text
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        attack_results = {}
        original_texts = {}
        
        # Load data.json if exists
        data_json_path = os.path.join(input_dir, "data.json")
        if os.path.exists(data_json_path):
            with open(data_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                original_texts = {item['id']: item.get('original_text') for item in data}
        
        # Process each audio file
        for file_name in os.listdir(input_dir):
            if not file_name.endswith(('.wav', '.mp3')):
                continue
                
            file_path = os.path.join(input_dir, file_name)
            file_id = os.path.splitext(file_name)[0]
            
            try:
                # Generate augmented versions
                augmented_files = self._generate_augmentations(
                    audio_file=file_path,
                    output_dir=output_dir
                )
                
                if augmented_files:
                    attack_results[file_id] = augmented_files
                    if file_id not in original_texts:
                        original_texts[file_id] = None
                        
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue
                
        return attack_results, original_texts
