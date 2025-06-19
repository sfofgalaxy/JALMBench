import json
import os

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

def main():
    """Main function to calculate accuracy for all defense methods"""
    # Defense methods to evaluate
    defense_methods = ["no_defense", "AdaShield", "FigStep", "JailbreakBench", "LLaMA-Guard", "Azure"]
    
    all_results = {}
    
    for defense_method in defense_methods:
        print(f"\nProcessing {defense_method}...")
        results = process_defense_method(defense_method)
        all_results[defense_method] = results
        print_results(defense_method, results)
    
    # Save all results to file
    save_results_to_file(all_results)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY - Best Model for Each Defense Method:")
    print("="*50)
    
    for defense_method, results in all_results.items():
        if results:
            best_model = max(results.items(), key=lambda x: x[1])
            print(f"{defense_method:<20}: {best_model[0]} ({best_model[1]:.2%})")
        else:
            print(f"{defense_method:<20}: No results available")

if __name__ == "__main__":
    main() 