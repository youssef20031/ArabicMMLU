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
    initial_reasoning_is_abductive_due_to_generic_prompt_prefix = False

    # Part 1: Determine Reasoning Type and parse based on primary prefix
    if filename.startswith("result_deductive_"):
        reasoning_type = "Deductive"
        temp_model_part = filename[len("result_deductive_"):]

        # English patterns (more specific first)
        if temp_model_part.startswith("deductive_en_datacot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("deductive_en_datacot__"):]
        elif temp_model_part.startswith("deductive_en_data_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("deductive_en_data_tot_"):]
        elif temp_model_part.startswith("deductive_en_data_"): # General for Direct
            language = "English"
            strategy = "Direct"
            model_name = temp_model_part[len("deductive_en_data_"):]
        # Simpler English patterns (e.g., "en_tot_modelname")
        elif temp_model_part.startswith("en_cot__"): # Must be before "en_"
            language = "English"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("en_cot__"):]
        elif temp_model_part.startswith("en_tot_"): # Must be before "en_"
            language = "English"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("en_tot_"):]
        elif temp_model_part.startswith("en_"): # General "en_" for Direct, must be last among "en_" patterns
            language = "English"
            strategy = "Direct"
            model_name = temp_model_part[len("en_"):]
        # Arabic patterns (if not matched by English patterns above)
        # ADD THIS NEW BLOCK for the specific "deductive_deductive_data_" case
        elif temp_model_part.startswith("deductive_deductive_data_"): # More specific than "deductive_data_"
            # This case arises from evaluate.py when task is deductive, data file is also named deductive_data,
            # and language info like _prompt_ar_alpa_ar is appended.
            # The language/strategy here are placeholders; the correction step should refine them
            # if the remaining model_name starts with a prompt pattern.
            language = "Arabic" # Default/placeholder
            strategy = "Direct" # Default/placeholder
            model_name = temp_model_part[len("deductive_deductive_data_"):]
        elif temp_model_part.startswith("deductive_datacot__"):
            language = "Arabic"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("deductive_datacot__"):]
        elif temp_model_part.startswith("deductive_data_tot_"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("deductive_data_tot_"):]
        elif temp_model_part.startswith("deductive_data_"):
            language = "Arabic"
            strategy = "Direct"
            model_name = temp_model_part[len("deductive_data_"):]
        else:
            # Fallback: language and strategy remain "Unknown" (their initial default from function start)
            model_name = temp_model_part

    elif filename.startswith("result_abductive_"):
        reasoning_type = "Abductive"
        # Default language for this block, will be overridden if specific conditions met
        language = "English"
        
        temp_model_part = filename[len("result_abductive_"):] # e.g., "abductive_datacot__groq_model"

        if temp_model_part.startswith("abductive_datacot__"):
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("abductive_datacot__"):]
        elif temp_model_part.startswith("abductive_data_tot_"):
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("abductive_data_tot_"):]
        elif temp_model_part.startswith("abductive_data_"): # General case for Direct
            strategy = "Direct"
            model_name = temp_model_part[len("abductive_data_"):]
        else:
            # Fallback if "result_abductive_" is not followed by a known "abductive_data..." pattern.
            model_name = temp_model_part
            # strategy remains "Unknown"

    elif filename.startswith("result_prompt_"):
        # For all "result_prompt_" files, assume Abductive reasoning based on user's examples.
        reasoning_type = "Abductive"
        initial_reasoning_is_abductive_due_to_generic_prompt_prefix = True
        
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
    elif filename.startswith("prompt_deductive_"):
        reasoning_type = "Deductive" # This is a specific prompt type, not setting the generic flag
        temp_model_part = filename[len("prompt_deductive_"):]

        # English patterns (more specific first)
        if temp_model_part.startswith("deductive_en_datacot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("deductive_en_datacot__"):]
        elif temp_model_part.startswith("deductive_en_data_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("deductive_en_data_tot_"):]
        elif temp_model_part.startswith("deductive_en_data_"): # General for Direct
            language = "English"
            strategy = "Direct"
            model_name = temp_model_part[len("deductive_en_data_"):]
        # Simpler English patterns (e.g., "en_tot_modelname")
        elif temp_model_part.startswith("en_cot__"): # Must be before "en_"
            language = "English"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("en_cot__"):]
        elif temp_model_part.startswith("en_tot_"): # Must be before "en_"
            language = "English"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("en_tot_"):]
        elif temp_model_part.startswith("en_"): # General "en_" for Direct, must be last among "en_" patterns
            language = "English"
            strategy = "Direct"
            model_name = temp_model_part[len("en_"):]
        # Arabic patterns (if not matched by English patterns above)
        elif temp_model_part.startswith("deductive_datacot__"):
            language = "Arabic"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("deductive_datacot__"):]
        elif temp_model_part.startswith("deductive_data_tot_"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("deductive_data_tot_"):]
        elif temp_model_part.startswith("deductive_data_"):
            language = "Arabic"
            strategy = "Direct"
            model_name = temp_model_part[len("deductive_data_"):]
        else:
            # Fallback: language and strategy remain "Unknown"
            model_name = temp_model_part
    elif filename.startswith("prompt_"):
        # For "prompt_" files, assume Abductive reasoning, similar to "result_prompt_"
        reasoning_type = "Abductive"
        initial_reasoning_is_abductive_due_to_generic_prompt_prefix = True
        
        # Strip "prompt_" and then parse language/strategy from the rest
        remaining_part = filename[len("prompt_"):] # e.g., "ar_alpa_arcot__modelname"

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
            # Fallback if "prompt_" is not followed by known lang/strat patterns.
            # language and strategy remain "Unknown".
            # model_name is set to the part after "prompt_"
            model_name = remaining_part 

    # --- Correction Step ---
    # If the derived model_name itself starts with patterns that indicate a specific
    # language, strategy, and reasoning type, override the previously determined values.
    # The model_name itself is not stripped here, to match the user's output format.

    is_prompt_model_correction = False # Flag to track if a prompt-specific rule applied

    # Check for Arabic prompt patterns within the current model_name
    # These are specific and also set reasoning_type to Abductive.
    if model_name.startswith("prompt_ar_alpa_arcot__"):
        language = "Arabic"; strategy = "Chain of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_ar_alpa_artot__"): # Specific ToT variant
        language = "Arabic"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_ar_alpa_ar_tot_"): # General ToT variant for Arabic
        language = "Arabic"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_ar_alpa_ar_"): # Direct Arabic
        language = "Arabic"; strategy = "Direct"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    # Check for English prompt patterns within the current model_name
    elif model_name.startswith("prompt_en_alpa_encot__"):
        language = "English"; strategy = "Chain of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_entot__"): # Specific ToT variant
        language = "English"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_en_tot_"): # General ToT variant for English
        language = "English"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_en_"): # Direct English
        language = "English"; strategy = "Direct"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"

    # If no prompt-specific correction was applied, check for generic "en_" prefixes in model_name.
    # This corrects language/strategy if initial parsing was ambiguous or incorrect.
    if not is_prompt_model_correction:
        if model_name.startswith("en_cot__"):
            language = "English"
            strategy = "Chain of Thought"
            # reasoning_type remains as initially determined
        elif model_name.startswith("en_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            # reasoning_type remains as initially determined
        elif model_name.startswith("en_"): # General "en_" prefix, check after more specific "en_cot__", "en_tot_"
            if language != "English":
                # If language was misidentified (e.g., parsed as Arabic but model_name is "en_..."), correct it.
                language = "English"
                strategy = "Direct" # Assume Direct strategy for this correction
            elif strategy == "Unknown":
                # If language was already English but strategy was not determined, set to Direct.
                strategy = "Direct"
            # If language is English and strategy is already known (e.g. CoT, ToT from initial parse),
            # a generic "en_" prefix in model_name doesn't override that known strategy.

    # Final global check for google_translate in the model_name to override language to Arabic
    if "_google_translate" in model_name:
        language = "Arabic"
        # Strategy and reasoning_type are not changed by google_translate alone.

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
                
                # Ensure keys exist in data or use defaults
                accuracy = data.get("overall_accuracy_calculated_from_csv", 0.0)
                metrics_data = data.get("metrics_from_json", {})

                result_entry = {
                    "model_name": model_name,
                    "original_filename": filename,
                    "accuracy": accuracy,
                    "metrics_data": metrics_data # Store the loaded metrics
                }

                # Populate the all_results structure
                all_results.setdefault(language, {}).setdefault(reasoning_type, {}).setdefault(strategy, []).append(result_entry)

            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                
    found_any_results = False
    for lang, reasoning_types_map in all_results.items():
        print(f"\nLanguage: {lang}")
        for reasoning_type, strategies_map in reasoning_types_map.items():
            print(f"  Reasoning Type: {reasoning_type}")
            for strategy, results_list in strategies_map.items():
                print(f"    Strategy: {strategy}")
                if results_list:
                    found_any_results = True
                    for result in sorted(results_list, key=lambda x: x['model_name']): # Sort by model name for consistent output
                        print(f"      Model: {result['model_name']} (Source: {result['original_filename']})")
                        print(f"        Accuracy (calc. from CSV): {result['accuracy']:.2f}%")
                        
                        current_metrics = result.get("metrics_data", {})
                        accuracy_from_metrics = current_metrics.get("overall_accuracy") # This is 0-1 float
                        macro_f1 = current_metrics.get("macro_f1_score")
                        weighted_f1 = current_metrics.get("weighted_f1_score")

                        if accuracy_from_metrics is not None:
                            print(f"        Accuracy (from _metrics.json): {accuracy_from_metrics * 100:.2f}%")
                        if macro_f1 is not None:
                            print(f"        Macro F1 (from _metrics.json): {macro_f1:.4f}")
                        if weighted_f1 is not None:
                            print(f"        Weighted F1 (from _metrics.json): {weighted_f1:.4f}")
                else:
                    print("      No results for this strategy.")

    if not found_any_results:
        print(f"\nNo valid categorized JSON files with 'overall_accuracy' found in '{folder_path}'.")


if __name__ == "__main__":
    project_root = "/home/youssef/Projects/ArabicMMLU" 
    target_folder = os.path.join(project_root, "test5") 
    compare_model_performance(target_folder)