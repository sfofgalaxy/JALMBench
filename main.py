# Evaluation of ALM's Safety targeting Jailbreak Attacks
from InitModel import get_model
import argparse
import os
import json
from pathlib import Path
import time
from datasets import load_dataset
import importlib.util
import soundfile as sf

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

def parse_args():
    parser = argparse.ArgumentParser(description='JALMBench entry')
    parser.add_argument('--model', type=str, required=True, help='model name, e.g. qwen, diva')
    parser.add_argument('--data', type=str, required=True, help='dataset subset name, e.g. AHarm, ADiv')
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
    # Check if the specified subset is valid
    if args.data not in data_subsets:
        print(f"Error: '{args.data}' is not a valid subset. Choose from: {data_subsets}")
        exit(1)
    # Load model
    model = get_model(args.model.lower())
    # Load dataset - args.data specifies the subset name, each subset only has 'train' split
    ds = load_dataset("AnonymousUser000/JALMBench", args.data)
    data = ds['train']
    # Filter data by language, gender, accent if specified
    if args.language is not None:
        data = [item for item in data if 'language' in item and item['language'] == args.language]
    if args.gender is not None:
        data = [item for item in data if 'gender' in item and item['gender'] == args.gender]
    if args.accent is not None:
        data = [item for item in data if 'accent' in item and item['accent'] == args.accent]
    
    # Output filename
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model}-{args.data}-{args.modality}.jsonl")
    
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
                if args.data.lower() == 'advwave' and item['target_model'] != model_mapping[args.model.lower()]:
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
            
            # Save only id, response, text, original_text
            out = {
                'id': item['id'],
                'response': response,
                'text': item['text'],
                'original_text': item['original_text']
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()

