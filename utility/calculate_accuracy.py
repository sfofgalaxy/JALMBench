import json
import os
import argparse

def calculate_accuracy(data):
    """Calculate accuracy rate"""
    correct = 0
    total = 0
    for item in data:
        ref = item.get("reference")
        pred = item.get("choice")
        if ref and pred:  # Only count when both answers exist
            total += 1
            if ref == pred:
                correct += 1
    return correct/total if total > 0 else 0

def process_defense_method(defense_method):
    """Process all models for a specific defense method"""
    # Get all json files in answer directory
    answer_dir = f"./answer/{defense_method}"
    
    if not os.path.exists(answer_dir):
        print(f"Directory not found: {answer_dir}")
        return {}
    
    results = {}
    
    for filename in os.listdir(answer_dir):
        if filename.endswith(".json"):
            model_name = filename.replace(".json", "")
            filepath = os.path.join(answer_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                accuracy = calculate_accuracy(data)
                results[model_name] = accuracy
                
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                continue
    
    return results

def print_results(defense_method, results):
    """Print formatted results for a defense method"""
    if not results:
        print(f"No results found for {defense_method}")
        return
    
    # Sort by accuracy in descending order
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{defense_method} Accuracy Rankings:")
    print("-" * 40)
    for model, acc in sorted_results:
        print(f"{model:<20} : {acc:.2%}")

def save_results_to_file(all_results, output_file="accuracy_results.json"):
    """Save all results to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\nAll results saved to: {output_file}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate accuracy for model predictions')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name to calculate accuracy for (e.g., qwen, vita, gemini, diva)')
    parser.add_argument('--defense', type=str, default="no_defense", 
                       choices=['no_defense', 'JailbreakBench', 'FigStep', 'AdaShield', 'LLaMAGuard', 'Azure'],
                       help='Defense method to calculate accuracy for (default: no_defense)')
    return parser.parse_args()

def calculate_accuracy_for_model(model_name: str, defense_method: str):
    """Calculate accuracy for a specific model and defense method"""
    # Get the answer file from extract_answers.py
    filepath = f"{model_name}-openbookqa-utility-{defense_method}-answer.json"
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        accuracy = calculate_accuracy(data)
        return accuracy
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def main():
    """Main function to calculate accuracy for a specific model and defense method"""
    # Parse command line arguments
    args = parse_args()
    
    # Calculate accuracy for the specified model and defense method
    accuracy = calculate_accuracy_for_model(args.model, args.defense)
    
    if accuracy is not None:
        print(f"\nAccuracy Results:")
        print("-" * 40)
        print(f"Model: {args.model}")
        print(f"Defense: {args.defense}")
        print(f"Accuracy: {accuracy:.2%}")
    else:
        print(f"Failed to calculate accuracy for {args.model} with {args.defense}")

if __name__ == "__main__":
    main() 