import os
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf  # pip install soundfile
import json
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from InitModel import get_model

def download_audio_data():
    """Download openbookqa audio data if not already exists"""
    save_dir = "./openbookqa_audios"
    
    # Check if audio directory already exists and has files
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"Audio files already exist in {save_dir}, skipping download...")
        return save_dir
    
    print("Audio files not found, downloading...")
    
    # Load openbookqa test dataset
    dataset = load_dataset("hlt-lab/voicebench", "openbookqa", split="test", cache_dir="./openbookqa_data")
    
    # Create target save directory
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, item in enumerate(tqdm(dataset, desc="Saving audio files")):
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        target_path = os.path.join(save_dir, f"{idx}.wav")
        sf.write(target_path, audio_array, sampling_rate)
    
    print(f"All audio files saved to {save_dir} directory.")
    return save_dir

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download OpenBookQA audio data and evaluate models')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name to evaluate (e.g., qwen, vita, gemini, diva)')
    parser.add_argument('--defense', type=str, default="no_defense", 
                       choices=['no_defense', 'JailbreakBench', 'FigStep', 'AdaShield', 'LLaMAGuard', 'Azure'],
                       help='Defense method to use (default: no_defense)')
    return parser.parse_args()

def evaluate_model(model_name: str, defense_method: str = "no_defense"):
    """Evaluate a specific model on openbookqa dataset"""
    # Load dataset
    dataset = load_dataset("hlt-lab/voicebench", "openbookqa", split="test", cache_dir="./openbookqa_data")
    
    # Initialize model
    model = get_model(model_name)
    
    results = []
    audio_dir = "./openbookqa_audios"
    
    # Output file in current directory
    output_file = f"{model_name}-openbookqa-utility-{defense_method}.json"
    
    # Process each audio sample
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        audio_path = os.path.join(audio_dir, f"{idx}.wav")
        
        # Generate prediction
        pred = model.process_audio(audio_path, "Please give the correct choice of the speaker's question without any additional information.")
        
        results.append({
            "audio_path": audio_path,
            "reference": item["reference"],
            "prediction": pred,
            "question": item["prompt"]
        })
        
        # Save results incrementally
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    return output_file

def main():
    """Main function to download audio and evaluate models"""
    # Parse command line arguments
    args = parse_args()
    
    # Step 1: Download audio data
    audio_dir = download_audio_data()
    
    # Step 2: Evaluate the specified model
    try:
        result_file = evaluate_model(args.model, args.defense)
        print(f"Completed evaluation for {args.model} with {args.defense}")
        print(f"Results saved to: {result_file}")
    except Exception as e:
        print(f"Error evaluating {args.model} with {args.defense}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 