import os
import json

def extract_info_from_filename(filename_full):
    """Extracts model name, language, strategy, and reasoning type from the filename."""
    filename = filename_full
    if filename.endswith(".json"):
        filename = filename[:-5] # Remove .json

    # Initialize defaults
    model_name = filename # Fallback model name
    language = "Unknown"
    strategy = "Unknown"
    reasoning_type = "Not Specified"

    # Part 1: Determine Reasoning Type and parse based on primary prefix
    if filename.startswith("result_deductive_"):
        reasoning_type = "Deductive"
        # Assuming Arabic for deductive files based on typical output and prior logic.
        # If deductive files can be other languages, this needs adjustment or a language identifier in the filename.
        language = "Arabic" 
        
        temp_model_part = filename[len("result_deductive_"):] # e.g., "deductive_datacot__groq_model"

        if temp_model_part.startswith("deductive_datacot__"):
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("deductive_datacot__"):]
        elif temp_model_part.startswith("deductive_data_tot_"):
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("deductive_data_tot_"):]
        elif temp_model_part.startswith("deductive_data_"):
            strategy = "Direct"
            model_name = temp_model_part[len("deductive_data_"):]
        else:
            # Fallback if "result_deductive_" is not followed by a known "deductive_data..." pattern.
            model_name = temp_model_part
            # strategy remains "Unknown" (or could be defaulted, e.g., to "Direct")
            # language is already set

    elif filename.startswith("result_prompt_"):
        # For all "result_prompt_" files, assume Abductive reasoning based on user's examples.
        reasoning_type = "Abductive"
        
        # Strip "result_prompt_" and then parse language/strategy from the rest
        remaining_part = filename[len("result_prompt_"):] # e.g., "ar_alpa_arcot__modelname"

        # Arabic patterns (order is important: more specific/longer patterns first)
        if remaining_part.startswith("ar_alpa_arcot__"):
            language = "Arabic"
            strategy = "Chain of Thought"
            model_name = remaining_part[len("ar_alpa_arcot__"):]
        elif remaining_part.startswith("ar_alpa_artot__"): # Specific ToT variant
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("ar_alpa_artot__"):]
        elif remaining_part.startswith("ar_alpa_ar_tot_"): # General ToT variant for Arabic
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("ar_alpa_ar_tot_"):]
        elif remaining_part.startswith("ar_alpa_ar_"): # Direct Arabic
            language = "Arabic"
            strategy = "Direct"
            model_name = remaining_part[len("ar_alpa_ar_"):]
        # English patterns (order is important: more specific/longer patterns first)
        elif remaining_part.startswith("en_alpa_encot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = remaining_part[len("en_alpa_encot__"):]
        elif remaining_part.startswith("en_alpa_entot__"): # Specific ToT variant
            language = "English"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("en_alpa_entot__"):]
        elif remaining_part.startswith("en_alpa_en_tot_"): # General ToT variant for English
            language = "English"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("en_alpa_en_tot_"):]
        elif remaining_part.startswith("en_alpa_en_"): # Direct English
            language = "English"
            strategy = "Direct"
            model_name = remaining_part[len("en_alpa_en_"):]
        else:
            # Fallback if "result_prompt_" is not followed by known lang/strat patterns.
            # language and strategy remain "Unknown".
            # model_name is set to the part after "result_prompt_"
            model_name = remaining_part 
    
    # else:
        # Filename doesn't start with "result_deductive_" or "result_prompt_".
        # All parameters (model_name, language, strategy, reasoning_type) 
        # retain their initial default values (model_name = original filename).

    return model_name, language, strategy, reasoning_type

def compare_model_performance(folder_path):
    """
    Compares model performance based on JSON files in the given folder,
    categorized by language, reasoning type, and then prompting strategy.

    Args:
        folder_path (str): The path to the folder containing JSON result files.
    """
    all_results = {} # New structure: lang -> reasoning_type -> strategy -> [results]
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name, language, strategy, reasoning_type = extract_info_from_filename(filename)
                overall_accuracy = data.get("overall_accuracy")
                
                if overall_accuracy is not None:
                    if language != "Unknown" and strategy != "Unknown": # Reasoning type can be "Not Specified"
                        # Ensure path in dict exists according to the new structure
                        if language not in all_results:
                            all_results[language] = {}
                        if reasoning_type not in all_results[language]:
                            all_results[language][reasoning_type] = {}
                        if strategy not in all_results[language][reasoning_type]:
                            all_results[language][reasoning_type][strategy] = []
                        
                        all_results[language][reasoning_type][strategy].append({"model": model_name, "accuracy": overall_accuracy})
                    else:
                        print(f"Warning: Could not determine language/strategy for {filename}. Model: {model_name}, Accuracy: {overall_accuracy}, Reasoning: {reasoning_type}")
                else:
                    print(f"Warning: 'overall_accuracy' not found in {filename}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Warning: Error processing {filename}: {e}")
                
    found_any_results = False
    for lang, reasoning_types_map in all_results.items():
        # Check if there's any data for this language across all reasoning types and strategies
        has_data_for_lang = any(
            any(bool(models) for models in strategies_map.values()) 
            for strategies_map in reasoning_types_map.values()
        )
        if not has_data_for_lang:
            continue
        
        found_any_results = True
        print(f"\n\n--- {lang.upper()} MODELS PERFORMANCE ---")
        
        for reasoning, strategies_map in reasoning_types_map.items():
            # Check if there's any data for this reasoning type across all strategies
            has_data_for_reasoning = any(bool(models) for models in strategies_map.values())
            if not has_data_for_reasoning:
                continue

            print(f"\nReasoning Type: {reasoning}")
            
            for strat, models_list in strategies_map.items():
                if not models_list: # Skip if no models for this strategy under this reasoning type
                    continue

                print(f"  Strategy: {strat}")
                header = f"{'Rank':<5} {'Model Name':<65} {'Accuracy (%)':<15}"
                # Indent header and lines appropriately
                print(f"    {'':<2}" + "-" * len(header)) 
                print(f"    {'':<2}{header}")
                print(f"    {'':<2}" + "-" * len(header))
                
                sorted_models = sorted(models_list, key=lambda x: x["accuracy"], reverse=True)
                
                for i, res in enumerate(sorted_models):
                    print(f"    {i+1:<5} {res['model']:<65} {res['accuracy']:<15.2f}")
                print(f"    {'':<2}" + "-" * len(header))
            # print("-" * (len(header) + 4)) # Separator for strategies within a reasoning type if needed

    if not found_any_results:
        print(f"\nNo valid categorized JSON files with 'overall_accuracy' found in '{folder_path}'.")


if __name__ == "__main__":
    project_root = "/home/youssef/Projects/ArabicMMLU" 
    target_folder = os.path.join(project_root, "test2") 
    compare_model_performance(target_folder)