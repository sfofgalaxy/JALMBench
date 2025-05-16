# Evaluation of ALM's Safety targeting Jailbreak Attacks
from InitModel import get_model
import argparse
import os
import json
from pathlib import Path
import time
from datasets import load_dataset
import importlib.util

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
    return parser.parse_args()

def main():
    args = parse_args()
    # Load model
    model = get_model(args.model.lower())
    # Load dataset
    ds = load_dataset("AnonymousUser000/JALMBench", args.data)
    data = ds[args.data] if args.data in ds else ds['train']
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
        for item in data:
            # Skip invalid data
            if args.modality == 'audio':
                if 'audio' not in item or item['audio'] is None:
                    continue
                audio_path = item['audio']['path'] if isinstance(item['audio'], dict) and 'path' in item['audio'] else item['audio']
                if not audio_path or not os.path.exists(audio_path):
                    continue
                # Apply defense prompt if specified
                current_prompt = args.prompt
                if defense_prompt:
                    current_prompt = f"{defense_prompt}\n{args.prompt}" if args.prompt else defense_prompt
                response = model.process_audio(audio_path, prompt=current_prompt)
            else:
                if 'text' not in item or item['text'] is None:
                    continue
                # Apply defense prompt if specified
                current_prompt = args.prompt
                if defense_prompt:
                    current_prompt = f"{defense_prompt}\n{args.prompt}" if args.prompt else defense_prompt
                response = model.process_text(item['text'], prompt=current_prompt)
            
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

