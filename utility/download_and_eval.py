import os
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf  # pip install soundfile
import json
import sys
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

def evaluate_model(model_name, defense_method="no_defense"):
    """Evaluate a specific model on openbookqa dataset"""
    # Load dataset
    dataset = load_dataset("hlt-lab/voicebench", "openbookqa", split="test", cache_dir="./openbookqa_data")
    
    # Initialize model
    model = get_model(model_name)
    
    results = []
    audio_dir = "./openbookqa_audios"
    
    # Create output directory
    output_dir = f"./results/{defense_method}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}.json")
    
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
    # List of models to evaluate
    model_names = [
        "vita", "gemini"
    ]
    
    # Defense methods to test
    defense_types = ["no_defense"]  # Can add more: ["AdaShield", "FigStep", "JailbreakBench"]
    
    # Step 1: Download audio data
    audio_dir = download_audio_data()
    
    # Step 2: Evaluate each model
    for model_name in model_names:
        for defense_type in defense_types:
            try:
                result_file = evaluate_model(model_name, defense_type)
                print(f"Completed evaluation for {model_name} with {defense_type}")
            except Exception as e:
                print(f"Error evaluating {model_name} with {defense_type}: {str(e)}")

if __name__ == "__main__":
    main() 