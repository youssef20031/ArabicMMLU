import os
import csv
import glob

def calculate_csv_accuracy(csv_filepath):
    """
    Calculates accuracy from a single CSV file.
    Assumes an 'is_correct' column where 'True' (case-insensitive) means correct.
    """
    correct_predictions = 0
    total_predictions = 0
    
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'is_correct' not in reader.fieldnames:
                print(f"Warning: 'is_correct' column not found in {os.path.basename(csv_filepath)}. Skipping this file.")
                return None, 0, 0 # Accuracy, correct, total

            for row_number, row in enumerate(reader, 1):
                total_predictions += 1
                is_correct_value = row.get('is_correct', '').strip().lower()
                if is_correct_value == 'true':
                    correct_predictions += 1
                elif is_correct_value not in ['false', '']: # Handle potentially empty or unexpected values
                    pass # Decide if this should count as incorrect or be handled differently
        
        if total_predictions == 0:
            if reader.fieldnames and 'is_correct' in reader.fieldnames: # File had header but no data rows
                 print(f"Warning: No data rows found in {os.path.basename(csv_filepath)}.")
            accuracy = 0.0
        else:
            accuracy = (correct_predictions / total_predictions) * 100
        
        return accuracy, correct_predictions, total_predictions
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return None, 0, 0
    except Exception as e:
        print(f"Error processing file {os.path.basename(csv_filepath)}: {e}")
        return None, 0, 0

def extract_info_from_csv_filename(filename_full, known_reasoning_type):
    """
    Extracts model name, language, and strategy from the CSV filename,
    given a known reasoning type.
    """
    filename = os.path.basename(filename_full)
    if filename.endswith(".csv"):
        filename = filename[:-4]

    # Initialize defaults
    model_name = filename # Fallback model name
    language = "Unknown"
    strategy = "Unknown"
    reasoning_type = known_reasoning_type

    # Common prefix for many result files, remove if present
    if filename.startswith("deduction_results_"):
        parse_part = filename[len("deduction_results_"):]
    elif filename.startswith("result_prompt_"): # For abductive that might follow old naming
        parse_part = filename[len("result_prompt_"):]
    elif filename.startswith("result_deductive_"): # For deductive that might follow old naming
        parse_part = filename[len("result_deductive_"):]
    else:
        parse_part = filename

    if reasoning_type == "Deductive Reasoning":
        language = "Arabic" # Default for deductive as per original logic, can be overridden
        
        if parse_part.startswith("ar_"):
            language = "Arabic"
            parse_part = parse_part[len("ar_"):]
        elif parse_part.startswith("en_"):
            language = "English"
            parse_part = parse_part[len("en_"):]
        
        # Strategy parsing for deductive
        # More specific prefixes first for patterns like "cot_deductive_..."
        if parse_part.startswith("cot_deductive_"):
            strategy = "Chain of Thought"
            model_name = parse_part[len("cot_deductive_"):]
        elif parse_part.startswith("tot_deductive_"):
            strategy = "Tree of Thought"
            model_name = parse_part[len("tot_deductive_"):]
        elif parse_part.startswith("zeroshot_deductive_"): # Existing specific prefix
            strategy = "Direct"
            model_name = parse_part[len("zeroshot_deductive_"):]
        # Older/alternative patterns
        elif parse_part.startswith("deductive_datacot__"):
            strategy = "Chain of Thought"
            model_name = parse_part[len("deductive_datacot__"):]
        elif parse_part.startswith("deductive_data_tot_"):
            strategy = "Tree of Thought"
            model_name = parse_part[len("deductive_data_tot_"):]
        elif parse_part.startswith("deductive_data_"): # General direct for older patterns
            strategy = "Direct"
            model_name = parse_part[len("deductive_data_"):]
        # Fallback generic keyword checks if specific prefixes didn't match and strategy is still Unknown
        elif "_cot_" in parse_part and strategy == "Unknown":
            strategy = "Chain of Thought"
            model_name = parse_part.replace("_cot_", "", 1) # Replace first occurrence
        elif "_tot_" in parse_part and strategy == "Unknown":
            strategy = "Tree of Thought"
            model_name = parse_part.replace("_tot_", "", 1) # Replace first occurrence
        elif ("_direct_" in parse_part or "zeroshot" in parse_part) and strategy == "Unknown":
            strategy = "Direct"
            if "_direct_" in parse_part:
                model_name = parse_part.split("_direct_", 1)[-1]
            # "zeroshot" in parse_part without a clear prefix might be part of model name, so handle carefully
            # If model_name wasn't changed by a more specific rule, it defaults to parse_part or filename
        else: 
            if model_name == filename: # model_name still holds the initial full basename (minus .csv)
                model_name = parse_part # Update to the more stripped version
            # strategy remains "Unknown" or its current value if set by language part.

    elif reasoning_type == "Abductive Reasoning":
        # Try to apply original logic for "result_prompt_" style patterns on the parse_part
        # e.g., ar_alpa_arcot__modelname or en_alpa_en_modelname
        # Arabic patterns
        if parse_part.startswith("ar_alpa_arcot__"):
            language = "Arabic"
            strategy = "Chain of Thought"
            model_name = parse_part[len("ar_alpa_arcot__"):]
        elif parse_part.startswith("ar_alpa_artot__"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = parse_part[len("ar_alpa_artot__"):]
        elif parse_part.startswith("ar_alpa_ar_tot_"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = parse_part[len("ar_alpa_ar_tot_"):]
        elif parse_part.startswith("ar_alpa_ar_"):
            language = "Arabic"
            strategy = "Direct"
            model_name = parse_part[len("ar_alpa_ar_"):]
        # English patterns
        elif parse_part.startswith("en_alpa_encot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = parse_part[len("en_alpa_encot__"):]
        elif parse_part.startswith("en_alpa_entot__"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = parse_part[len("en_alpa_entot__"):]
        elif parse_part.startswith("en_alpa_en_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = parse_part[len("en_alpa_en_tot_"):]
        elif parse_part.startswith("en_alpa_en_"):
            language = "English"
            strategy = "Direct"
            model_name = parse_part[len("en_alpa_en_"):]
        else:
            model_name = parse_part
            if language == "Unknown":
                if "_ar_" in model_name or model_name.startswith("ar_"): language = "Arabic"
                elif "_en_" in model_name or model_name.startswith("en_"): language = "English"
            if strategy == "Unknown":
                if "cot" in model_name: strategy = "Chain of Thought"
                elif "tot" in model_name: strategy = "Tree of Thought"
                elif "direct" in model_name or "zeroshot" in model_name: strategy = "Direct"
                else: # Default to Direct for Abductive if no other indicators found
                    strategy = "Direct"
    
    # Clean up model name from potential provider prefixes
    # Check for more specific "<provider>_model_" prefixes first
    if model_name.startswith("groq_model_"):
        model_name = model_name[len("groq_model_"):]
    elif model_name.startswith("hf_model_"):
        model_name = model_name[len("hf_model_"):]
    elif model_name.startswith("openai_model_"):
        model_name = model_name[len("openai_model_"):]
    # Then check for general "<provider>_" prefixes
    elif model_name.startswith("groq_"):
        model_name = model_name[len("groq_"):]
    elif model_name.startswith("hf_"):
        model_name = model_name[len("hf_"):]
    elif model_name.startswith("openai_"):
        model_name = model_name[len("openai_"):]
    
    if model_name.endswith("_"): model_name = model_name[:-1]

    return model_name, language, strategy, reasoning_type

def process_results_folder(folder_path, reasoning_type_name):
    """
    Processes all CSV files in a given folder for a specific reasoning type.
    """
    print(f"\nProcessing {reasoning_type_name} results from folder: {folder_path}")
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    categorized_results = {}
    grand_total_correct_folder = 0
    grand_total_predictions_folder = 0

    for csv_filepath in csv_files:
        filename = os.path.basename(csv_filepath)
        accuracy, correct, total = calculate_csv_accuracy(csv_filepath)
        
        if accuracy is not None and total > 0:
            grand_total_correct_folder += correct
            grand_total_predictions_folder += total
            
            model_name, language, strategy, _ = extract_info_from_csv_filename(filename, reasoning_type_name)

            if language not in categorized_results:
                categorized_results[language] = {}
            if strategy not in categorized_results[language]:
                categorized_results[language][strategy] = []
            
            categorized_results[language][strategy].append({
                "model": model_name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "filename": filename
            })
        elif accuracy is not None and total == 0 and 'is_correct' in open(csv_filepath, 'r', encoding='utf-8').readline():
             print(f"  - {filename}: No data rows to calculate accuracy.")

    found_any_categorized_results = False
    for lang, strategies_map in categorized_results.items():
        if not any(models_list for models_list in strategies_map.values()):
            continue
        
        found_any_categorized_results = True
        print(f"\n--- Language: {lang.upper()} ({reasoning_type_name}) ---")
        
        for strat, models_list in strategies_map.items():
            if not models_list:
                continue

            print(f"  Strategy: {strat}")
            sorted_models = sorted(models_list, key=lambda x: x["accuracy"], reverse=True)
            
            header = f"{'Rank':<5} {'Model Name':<60} {'Accuracy (%)':<15} {'Correct/Total':<15} {'Filename':<50}"
            print(f"    {'':<2}" + "-" * (len(header)+10))
            print(f"    {'':<2}{header}")
            print(f"    {'':<2}" + "-" * (len(header)+10))
            
            for i, res in enumerate(sorted_models):
                accuracy_str = f"{res['accuracy']:.2f}"
                correct_total_str = f"{res['correct']}/{res['total']}"
                print(f"    {i+1:<7} {res['model']:<60} {accuracy_str:<15} {correct_total_str:<15} {res['filename']:<50}")
            print(f"    {'':<2}" + "-" * (len(header)+10))
            print("") 

    if not found_any_categorized_results and csv_files:
        print(f"\n  No valid categorized data processed for {reasoning_type_name} in folder '{os.path.basename(folder_path)}'.")
    
    if grand_total_predictions_folder > 0:
        overall_folder_accuracy = (grand_total_correct_folder / grand_total_predictions_folder) * 100
        print(f"\n  Overall {reasoning_type_name} accuracy for folder '{os.path.basename(folder_path)}': {overall_folder_accuracy:.2f}% ({grand_total_correct_folder}/{grand_total_predictions_folder})")
    elif csv_files and not found_any_categorized_results:
        print(f"\n  No valid data processed to calculate overall accuracy for {reasoning_type_name} in folder '{os.path.basename(folder_path)}'.")
        
    return categorized_results

if __name__ == "__main__":
    project_root = "/home/youssef/Projects/ArabicMMLU"

    deductive_results_folder = os.path.join(project_root, "results_deductive_groq")
    abductive_results_folder = os.path.join(project_root, "results_deduction_groq")

    print("Starting results calculation...")

    process_results_folder(deductive_results_folder, "Deductive Reasoning")
    process_results_folder(abductive_results_folder, "Abductive Reasoning")
        
    print("\nResults calculation finished.")
