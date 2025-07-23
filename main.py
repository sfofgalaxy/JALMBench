# Evaluation of ALM's Safety targeting Jailbreak Attacks
from InitModel import get_model
import argparse
import os
import json
from datasets import load_dataset
import importlib.util
import soundfile as sf
from huggingface_hub import login
import dotenv
import glob
from typing import List, Dict, Any, Optional

# List of allowed subset names for JALMBench
data_subsets = [
    'ADiv', 'AHarm', 'AMSE', 'AdvWave', 'BoN', 'DAN', 'DI', 'ICA', 'PAP', 'SSJ', 'THarm'
]

model_mapping = {
    'diva': 'DiVA',
    'fo': 'Freeze-Omni',
    'glm': 'GLM-4-Voice',
    'gpt': 'GPT-4o-Audio',
    'gemini': 'Gemini-2.0-Flash',
    'lo': 'LLaMA-Omni',
    'qwen': 'Qwen2-Audio',
    'salmonn': 'SalmonN',
    'speechgpt': 'SpeechGPT',
    'spirit': 'Spirit LM Base',
    'vita': 'VITA-1.0',
    'vita_1_5': 'VITA-1.5'
}

def load_defense_prompt(defense_name):
    """Load defense prompt from defense/prompts directory"""
    prompt_path = os.path.join('defense', 'prompts', f'{defense_name}.md')
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def load_output_filter(filter_name):
    """Load output filter module from defense/output_filter directory"""
    filter_path = os.path.join('defense', 'output_filter', f'{filter_name}.py')
    if os.path.exists(filter_path):
        spec = importlib.util.spec_from_file_location(filter_name, filter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None

def load_data(data_path: str, 
             language: Optional[str] = None, 
             gender: Optional[str] = None, 
             accent: Optional[str] = None) -> tuple[List[Dict[str, Any]], bool]:
    """
    Load data from either a dataset subset or a folder containing audio files.
    
    Args:
        data_path: Path to the dataset or folder
        language: Optional language filter
        gender: Optional gender filter
        accent: Optional accent filter
        
    Returns:
        Tuple of (data list, is_folder flag)
    """
    is_folder = False
    
    # Check if input is a subset or folder
    if data_path not in data_subsets:
        if not os.path.isdir(data_path):
            raise ValueError(f"'{data_path}' is neither a valid subset nor a directory. Choose from: {data_subsets} or provide a valid directory path")
        is_folder = True
    
    if is_folder:
        # Get all audio files in the folder
        audio_files = []
        for ext in ['mp3', 'wav']:
            audio_files.extend(glob.glob(os.path.join(data_path, f"*.{ext}")))
        
        if not audio_files:
            raise ValueError(f"No mp3 or wav files found in directory '{data_path}'")
            
        # Try to load data.json if it exists
        data_json_path = os.path.join(data_path, "data.json")
        original_texts = {}
        if os.path.exists(data_json_path):
            try:
                with open(data_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    original_texts = {item['id']: item.get('original_text') for item in json_data}
            except Exception as e:
                print(f"Warning: Failed to load data.json: {e}")
        
        # Create data list
        data = []
        for audio_file in audio_files:
            file_id = os.path.splitext(os.path.basename(audio_file))[0]
            data.append({
                'id': file_id,
                'audio': audio_file,
                'original_text': original_texts.get(file_id)
            })
    else:
        # Load dataset from huggingface
        ds = load_dataset("AnonymousUser000/JALMBench", data_path)
        data = ds['train']
        
        # Apply filters for dataset
        if language is not None:
            data = [item for item in data if 'language' in item and item['language'] == language]
        if gender is not None:
            data = [item for item in data if 'gender' in item and item['gender'] == gender]
        if accent is not None:
            data = [item for item in data if 'accent' in item and item['accent'] == accent]
    
    return data, is_folder

def parse_args():
    parser = argparse.ArgumentParser(description='JALMBench entry')
    parser.add_argument('--model', type=str, required=True, help='model name, e.g. qwen, diva')
    parser.add_argument('--data', type=str, required=True, help='dataset subset name (e.g. AHarm, ADiv) or folder path containing mp3, wav files')
    parser.add_argument('--modality', type=str, required=True, choices=['audio', 'text'], help='input modality: audio or text')
    parser.add_argument('--output_dir', type=str, default='.', help='output directory')
    parser.add_argument('--prompt', type=str, default=None, help='optional prompt')
    parser.add_argument('--defense', type=str, default=None, choices=['JailbreakBench', 'FigStep', 'AdaShield', 'LLaMAGuard', 'Azure'], 
                      help='defense method to use')
    # Add filter arguments
    parser.add_argument('--language', type=str, default=None, help='filter by language, e.g. en')
    parser.add_argument('--gender', type=str, default=None, help='filter by gender, e.g. male, female, Neutral')
    parser.add_argument('--accent', type=str, default=None, help='filter by accent, e.g. US, GB, AU, IN')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Load data
        data, is_folder = load_data(
            args.data,
            language=args.language,
            gender=args.gender,
            accent=args.accent
        )
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Load model
    model = get_model(args.model.lower())
    
    # Output filename - use folder name if processing folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    folder_name = os.path.basename(args.data.rstrip('/\\')) if is_folder else args.data
    output_path = os.path.join(output_dir, f"{args.model}-{folder_name}-{args.modality}.jsonl")
    
    # Load defense if specified
    defense_prompt = None
    output_filter = None
    if args.defense:
        if args.defense in ['JailbreakBench', 'FigStep', 'AdaShield']:
            defense_prompt = load_defense_prompt(args.defense)
        else:
            output_filter = load_output_filter(args.defense)
    
    # Process each item
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, item in enumerate(data):
            # Skip invalid data
            if args.modality == 'audio':
                if 'audio' not in item or item['audio'] is None:
                    continue
                # If args.data is AdvWave, only process item with target model
                if not is_folder and args.data.lower() == 'advwave' and item['target_model'] != model_mapping[args.model.lower()]:
                    continue
                audio_data = item['audio']
                # If audio_data is not a file path, save it as a temporary wav file
                if isinstance(audio_data, str) and os.path.exists(audio_data):
                    audio_path = audio_data
                    temp_file = None
                else:
                    # Save audio to temporary wav file
                    audio_path = f"tmp_audio_{idx}.wav"
                    # audio_data can be dict with 'array' and 'sampling_rate', or datasets Audio object
                    if isinstance(audio_data, dict) and 'array' in audio_data and 'sampling_rate' in audio_data:
                        sf.write(audio_path, audio_data['array'], audio_data['sampling_rate'])
                    elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
                        sf.write(audio_path, audio_data.array, audio_data.sampling_rate)
                    else:
                        continue
                    temp_file = audio_path
                # Apply defense prompt if specified
                current_prompt = item.get('prompt', None) # if SSJ, item['prompt'] is None, SSJ must contain prompt in the text
                if args.prompt:
                    if current_prompt is not None:
                        current_prompt = f"{current_prompt}\n{args.prompt}"
                    else:
                        current_prompt = args.prompt
                response = model.process_audio(audio_path, prompt=current_prompt, addtional_system_prompt=defense_prompt)
                # Remove temporary file if created
                if temp_file is not None and os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                # Only DAN, DI, ICA, PAP, THarm support text data
                if args.data.lower() not in ['dan', 'di', 'ica', 'pap', 'tharm']:
                    continue
                if 'text' not in item or item['text'] is None:
                    continue
                # Apply defense prompt if specified
                current_prompt = item['text']
                if args.prompt:
                    current_prompt = f"{current_prompt}\n{args.prompt}"
                response = model.process_audio(audio_path=None, prompt=current_prompt, addtional_system_prompt=defense_prompt)
            
            # Apply output filter if specified
            if output_filter and hasattr(output_filter, 'is_safe'):
                if not output_filter.is_safe(response):
                    response = "Content Filtered, Unsafe Response."
            
            # Save only id, response, original_text
            out = {
                'id': item['id'],
                'response': response,
                'original_text': item.get('original_text', None)
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    dotenv.load_dotenv()
    login(token=os.getenv('HUGGING_FACE_TOKEN'))
    main()

