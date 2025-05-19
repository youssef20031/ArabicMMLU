import os
import json

def extract_info_from_filename(filename_full):
    """Extracts model name, language, and strategy from the filename."""
    filename = filename_full
    if filename.endswith(".json"):
        filename = filename[:-5]

    model_name = filename # Default model name
    language = "Unknown"
    strategy = "Unknown"

    # Order of checks is important: more specific (e.g., 'arcot', 'artot') before general ('ar')
    # Arabic patterns
    if "result_prompt_ar_alpa_arcot__" in filename:
        language = "Arabic"
        strategy = "Chain of Thought"
        model_name = filename.replace("result_prompt_ar_alpa_arcot__", "")
    elif "result_prompt_ar_alpa_artot__" in filename: # Assuming 'artot' for Arabic Tree of Thought
        language = "Arabic"
        strategy = "Tree of Thought"
        model_name = filename.replace("result_prompt_ar_alpa_artot__", "")
    elif "result_prompt_ar_alpa_ar_" in filename:
        language = "Arabic"
        strategy = "Direct"
        model_name = filename.replace("result_prompt_ar_alpa_ar_", "")
    # English patterns (hypothetical, adjust prefixes as needed)
    elif "result_prompt_en_alpa_encot__" in filename: # Assuming 'encot' for English Chain of Thought
        language = "English"
        strategy = "Chain of Thought"
        model_name = filename.replace("result_prompt_en_alpa_encot__", "")
    elif "result_prompt_en_alpa_entot__" in filename: # Assuming 'entot' for English Tree of Thought
        language = "English"
        strategy = "Tree of Thought"
        model_name = filename.replace("result_prompt_en_alpa_entot__", "")
    elif "result_prompt_en_alpa_en_" in filename: # Assuming 'en' for English Direct
        language = "English"
        strategy = "Direct"
        model_name = filename.replace("result_prompt_en_alpa_en_", "")
    else:
        # Basic fallback if no known detailed prefix is matched
        # You might want to add more general prefix stripping here if needed
        # For example, if all files start with "result_prompt_"
        if filename.startswith("result_prompt_"):
            # This is a generic attempt, be cautious if model names could contain these parts
            # For now, we rely on the specific prefixes above.
            pass


    return model_name, language, strategy

def compare_model_performance(folder_path):
    """
    Compares model performance based on JSON files in the given folder,
    categorized by language and prompting strategy.

    Args:
        folder_path (str): The path to the folder containing JSON result files.
    """
    all_results = {
        "Arabic": {
            "Direct": [],
            "Chain of Thought": [],
            "Tree of Thought": []
        },
        "English": {
            "Direct": [],
            "Chain of Thought": [],
            "Tree of Thought": []
        }
    }
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name, language, strategy = extract_info_from_filename(filename)
                overall_accuracy = data.get("overall_accuracy")
                
                if overall_accuracy is not None:
                    if language != "Unknown" and strategy != "Unknown":
                        if language in all_results and strategy in all_results[language]:
                            all_results[language][strategy].append({"model": model_name, "accuracy": overall_accuracy})
                        else:
                            print(f"Warning: Uncategorized result for {filename}. Language: {language}, Strategy: {strategy} not in predefined categories.")
                    else:
                        print(f"Warning: Could not determine language/strategy for {filename}. Model: {model_name}, Accuracy: {overall_accuracy}")
                else:
                    print(f"Warning: 'overall_accuracy' not found in {filename}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Warning: Error processing {filename}: {e}")
                
    found_any_results = False
    for lang, strategies_data in all_results.items():
        has_data_for_lang = any(bool(models) for models in strategies_data.values())
        if not has_data_for_lang:
            continue
        
        found_any_results = True
        print(f"\n\n--- {lang.upper()} MODELS PERFORMANCE ---")
        
        for strat, models_list in strategies_data.items():
            if not models_list: # Skip if no models for this strategy under this language
                continue

            print(f"\nStrategy: {strat}")
            header = f"{'Rank':<5} {'Model Name':<65} {'Accuracy (%)':<15}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            
            sorted_models = sorted(models_list, key=lambda x: x["accuracy"], reverse=True)
            
            for i, res in enumerate(sorted_models):
                print(f"{i+1:<5} {res['model']:<65} {res['accuracy']:<15.2f}")
            print("-" * len(header))

    if not found_any_results:
        print(f"\nNo valid categorized JSON files with 'overall_accuracy' found in '{folder_path}'.")


if __name__ == "__main__":
    project_root = "/home/youssef/Projects/ArabicMMLU" 
    target_folder = os.path.join(project_root, "test2") 
    compare_model_performance(target_folder)