import argparse
import os
import json
from typing import List
from InitModel import get_model
from attack.AMSE.amse_attack import AMSEAttack
from attack.SSJ.ssj_attack import SSJAttack
from attack.AdvWave_B.advwave_attack import AdvWaveAttack
import importlib.util

def parse_args():
    parser = argparse.ArgumentParser(description='JALMBench Attack Entry')
    parser.add_argument('--model', type=str, required=True, help='model name, e.g. qwen, diva')
    parser.add_argument('--data', type=str, required=True, help='folder path containing audio files')
    parser.add_argument('--output_dir', type=str, default='.', help='output directory')
    parser.add_argument('--prompt', type=str, default=None, help='optional prompt')
    parser.add_argument('--defense', type=str, default=None, 
                       choices=['JailbreakBench', 'FigStep', 'AdaShield', 'LLaMAGuard', 'Azure'],
                       help='defense method to use')
    parser.add_argument('--attack', type=str, required=True,
                       choices=['AMSE', 'SSJ', 'BoN', 'AdvWave'],
                       help='attack method to use')
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Verify data is a directory
    if not os.path.isdir(args.data):
        print(f"Error: '{args.data}' is not a directory")
        exit(1)
    
    # Load model
    model = get_model(args.model.lower())
    
    # Load defense if specified
    defense_prompt = None
    output_filter = None
    if args.defense:
        if args.defense in ['JailbreakBench', 'FigStep', 'AdaShield']:
            defense_prompt = load_defense_prompt(args.defense)
        else:
            output_filter = load_output_filter(args.defense)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 
                              f"{args.model}-{os.path.basename(args.data.rstrip('/\\'))}-{args.attack}.jsonl")
    
    # Process each audio file according to attack type
    if args.attack == 'AMSE':
        amse = AMSEAttack()
        
        # Process all audio files in the input directory
        attack_results, original_texts = amse.process_audio_folder(args.data, args.output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as fout:
            for file_id, attack_files in attack_results.items():
                # Process each attack attempt
                for attempt_id, attack_file in enumerate(attack_files, 1):
                    response = model.process_audio(attack_file, 
                                                 prompt=args.prompt,
                                                 addtional_system_prompt=defense_prompt)
                    
                    # Apply output filter if specified
                    if output_filter and hasattr(output_filter, 'is_safe'):
                        if not output_filter.is_safe(response):
                            response = "Content Filtered, Unsafe Response."
                    
                    # Save result with original text if available
                    out = {
                        'id': file_id,
                        'attempt_id': attempt_id,
                        'response': response,
                        'original_text': original_texts.get(file_id)
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                    
                    # Clean up attack file
                    os.remove(attack_file)
                    
    elif args.attack == 'SSJ':
        ssj = SSJAttack()
        
        # Process all texts in data.json
        attack_results, original_texts = ssj.process_audio_folder(args.data, args.output_dir)
        
        # Load SSJ prompt template
        with open("attack/SSJ/SSJ.md", "r", encoding="utf-8") as f:
            ssj_prompt_template = f.read()
        
        with open(output_path, 'w', encoding='utf-8') as fout:
            for file_id, attack_files in attack_results.items():
                # Process each attack attempt
                for attempt_id, attack_file in enumerate(attack_files, 1):
                    # Create SSJ specific prompt
                    ssj_prompt = ssj_prompt_template.replace("[REPLACE]", original_texts[file_id])
                    
                    response = model.process_audio(attack_file, 
                                                 prompt=ssj_prompt,
                                                 addtional_system_prompt=defense_prompt)
                    
                    # Apply output filter if specified
                    if output_filter and hasattr(output_filter, 'is_safe'):
                        if not output_filter.is_safe(response):
                            response = "Content Filtered, Unsafe Response."
                    
                    # Save result with original text if available
                    out = {
                        'id': file_id,
                        'attempt_id': attempt_id,
                        'response': response,
                        'original_text': original_texts.get(file_id)
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                    
                    # Clean up attack file
                    os.remove(attack_file)
                    
    elif args.attack == 'AdvWave':
        advwave = AdvWaveAttack(max_rounds=30)
        
        # Process all items in data.json
        attack_results, original_texts = advwave.process_audio_folder(
            input_dir=args.data,
            output_dir=args.output_dir,
            model=model
        )
        
        with open(output_path, 'w', encoding='utf-8') as fout:
            for file_id, attack_files in attack_results.items():
                # Process each attack attempt
                for attempt_id, attack_file in enumerate(attack_files, 1):
                    response = model.process_audio(attack_file, 
                                                 prompt=args.prompt,
                                                 addtional_system_prompt=defense_prompt)
                    
                    # Apply output filter if specified
                    if output_filter and hasattr(output_filter, 'is_safe'):
                        if not output_filter.is_safe(response):
                            response = "Content Filtered, Unsafe Response."
                    
                    # Save result with original text if available
                    out = {
                        'id': file_id,
                        'attempt_id': attempt_id,
                        'response': response,
                        'original_text': original_texts.get(file_id)
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                    
                    # Clean up attack file
                    os.remove(attack_file)
    
    else:
        print(f"Attack method '{args.attack}' not implemented yet")
        exit(1)

if __name__ == "__main__":
    main() 