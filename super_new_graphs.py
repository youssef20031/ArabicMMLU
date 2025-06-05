import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Ensure the output directory for graphs exists
output_graph_dir = "/home/youssef/Projects/ArabicMMLU/super_output_graphs"
os.makedirs(output_graph_dir, exist_ok=True)

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
            model_name = temp_model_part

    elif filename.startswith("result_abductive_"):
        reasoning_type = "Abductive"
        language = "English"
        temp_model_part = filename[len("result_abductive_"):]
        if temp_model_part.startswith("abductive_datacot__"):
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("abductive_datacot__"):]
        elif temp_model_part.startswith("abductive_data_tot_"):
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("abductive_data_tot_"):]
        elif temp_model_part.startswith("abductive_data_"):
            strategy = "Direct"
            model_name = temp_model_part[len("abductive_data_"):]
        else:
            model_name = temp_model_part

    elif filename.startswith("result_prompt_"):
        reasoning_type = "Abductive"
        initial_reasoning_is_abductive_due_to_generic_prompt_prefix = True
        remaining_part = filename[len("result_prompt_"):]
        if remaining_part.startswith("ar_alpa_arcot__"):
            language = "Arabic"
            strategy = "Chain of Thought"
            model_name = remaining_part[len("ar_alpa_arcot__"):]
        elif remaining_part.startswith("ar_alpa_artot__"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("ar_alpa_artot__"):]
        elif remaining_part.startswith("ar_alpa_ar_tot_"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("ar_alpa_ar_tot_"):]
        elif remaining_part.startswith("ar_alpa_ar_"):
            language = "Arabic"
            strategy = "Direct"
            model_name = remaining_part[len("ar_alpa_ar_"):]
        elif remaining_part.startswith("en_alpa_encot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = remaining_part[len("en_alpa_encot__"):]
        elif remaining_part.startswith("en_alpa_entot__"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("en_alpa_entot__"):]
        elif remaining_part.startswith("en_alpa_en_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("en_alpa_en_tot_"):]
        elif remaining_part.startswith("en_alpa_en_"):
            language = "English"
            strategy = "Direct"
            model_name = remaining_part[len("en_alpa_en_"):]
        else:
            model_name = remaining_part
    elif filename.startswith("prompt_deductive_"):
        reasoning_type = "Deductive"
        temp_model_part = filename[len("prompt_deductive_"):]
        if temp_model_part.startswith("deductive_en_datacot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("deductive_en_datacot__"):]
        elif temp_model_part.startswith("deductive_en_data_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("deductive_en_data_tot_"):]
        elif temp_model_part.startswith("deductive_en_data_"):
            language = "English"
            strategy = "Direct"
            model_name = temp_model_part[len("deductive_en_data_"):]
        elif temp_model_part.startswith("en_cot__"): 
            language = "English"
            strategy = "Chain of Thought"
            model_name = temp_model_part[len("en_cot__"):]
        elif temp_model_part.startswith("en_tot_"): 
            language = "English"
            strategy = "Tree of Thought"
            model_name = temp_model_part[len("en_tot_"):]
        elif temp_model_part.startswith("en_"): 
            language = "English"
            strategy = "Direct"
            model_name = temp_model_part[len("en_"):]
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
            model_name = temp_model_part
    elif filename.startswith("prompt_"):
        reasoning_type = "Abductive"
        initial_reasoning_is_abductive_due_to_generic_prompt_prefix = True
        remaining_part = filename[len("prompt_"):]
        if remaining_part.startswith("ar_alpa_arcot__"):
            language = "Arabic"
            strategy = "Chain of Thought"
            model_name = remaining_part[len("ar_alpa_arcot__"):]
        elif remaining_part.startswith("ar_alpa_artot__"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("ar_alpa_artot__"):]
        elif remaining_part.startswith("ar_alpa_ar_tot_"):
            language = "Arabic"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("ar_alpa_ar_tot_"):]
        elif remaining_part.startswith("ar_alpa_ar_"):
            language = "Arabic"
            strategy = "Direct"
            model_name = remaining_part[len("ar_alpa_ar_"):]
        elif remaining_part.startswith("en_alpa_encot__"):
            language = "English"
            strategy = "Chain of Thought"
            model_name = remaining_part[len("en_alpa_encot__"):]
        elif remaining_part.startswith("en_alpa_entot__"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("en_alpa_entot__"):]
        elif remaining_part.startswith("en_alpa_en_tot_"):
            language = "English"
            strategy = "Tree of Thought"
            model_name = remaining_part[len("en_alpa_en_tot_"):]
        elif remaining_part.startswith("en_alpa_en_"):
            language = "English"
            strategy = "Direct"
            model_name = remaining_part[len("en_alpa_en_"):]
        else:
            model_name = remaining_part

    is_prompt_model_correction = False
    if model_name.startswith("prompt_ar_alpa_arcot__"):
        language = "Arabic"; strategy = "Chain of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_ar_alpa_artot__"):
        language = "Arabic"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_ar_alpa_ar_tot_"):
        language = "Arabic"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_ar_alpa_ar_"):
        language = "Arabic"; strategy = "Direct"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_encot__"):
        language = "English"; strategy = "Chain of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_entot__"):
        language = "English"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_en_tot_"):
        language = "English"; strategy = "Tree of Thought"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"
    elif model_name.startswith("prompt_en_alpa_en_"):
        language = "English"; strategy = "Direct"; is_prompt_model_correction = True
        if initial_reasoning_is_abductive_due_to_generic_prompt_prefix or reasoning_type == "Not Specified":
            reasoning_type = "Abductive"

    if not is_prompt_model_correction:
        if model_name.startswith("en_cot__"):
            language = "English"
            strategy = "Chain of Thought"
        elif model_name.startswith("en_tot_"):
            language = "English"
            strategy = "Tree of Thought"
        elif model_name.startswith("en_"):
            if language != "English":
                language = "English"
                strategy = "Direct"
            elif strategy == "Unknown":
                strategy = "Direct"

    if "_google_translate" in model_name:
        language = "Arabic"

    return model_name, language, strategy, reasoning_type

def plot_model_performance(all_results, base_output_path):
    """Generates and saves bar charts for model performance."""
    for lang, reasoning_types_map in all_results.items():
        for reasoning, strategies_map in reasoning_types_map.items():
            for strat, models_list in strategies_map.items():
                if not models_list:
                    continue

                sorted_models = sorted(models_list, key=lambda x: x["accuracy"], reverse=True)
                model_names = [res["model_name"] for res in sorted_models]
                accuracies = [res["accuracy"] for res in sorted_models]

                if not model_names: # Skip if no models after filtering
                    continue

                plt.figure(figsize=(12, 8))
                bars = plt.bar(model_names, accuracies, color='skyblue')
                plt.xlabel("Model Name")
                plt.ylabel("Overall Accuracy (%)")
                plt.title(f"Model Performance: {lang} - {reasoning} - {strat}")
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 100) # Assuming accuracy is a percentage
                plt.tight_layout() # Adjust layout to prevent labels from overlapping

                # Add accuracy values on top of bars
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')

                # Sanitize filename components
                safe_lang = lang.replace(" ", "_")
                safe_reasoning = reasoning.replace(" ", "_").replace("/", "_")
                safe_strat = strat.replace(" ", "_")
                
                plot_filename = f"{safe_lang}_{safe_reasoning}_{safe_strat}_performance.png"
                plot_path = os.path.join(base_output_path, plot_filename)
                plt.savefig(plot_path)
                plt.close() # Close the figure to free memory
                print(f"Saved plot: {plot_path}")

def plot_radar_performance(all_results, base_output_path):
    """Generates and saves radar charts showing each model's performance across strategies."""
    for lang, reasoning_types_map in all_results.items():
        for reasoning, strategies_map in reasoning_types_map.items():
            # strategies_map: strategy_name -> [{"model_name": model_name, "accuracy": acc}, ...]

            # 1. Collect all unique strategy names (radar axes)
            # Filter out strategies that might be empty or have no valid models_list
            all_strategy_names = sorted([
                strat_name for strat_name, models_list in strategies_map.items() if models_list
            ])

            if len(all_strategy_names) < 3:
                print(f"Skipping radar chart for {lang} - {reasoning}: Needs at least 3 strategies with data, found {len(all_strategy_names)} ({', '.join(all_strategy_names)}).")
                continue

            # 2. Collect all unique model names and their performance data
            # model_data: {model_name: {strategy_name: accuracy}}
            model_data = {}
            for strat_name, models_list in strategies_map.items():
                if not models_list: # Should be caught by all_strategy_names filter, but double-check
                    continue
                for res in models_list:
                    model_name = res["model_name"]
                    accuracy = res["accuracy"]
                    if model_name not in model_data:
                        model_data[model_name] = {}
                    model_data[model_name][strat_name] = accuracy
            
            if not model_data:
                # This case should ideally not be hit if all_strategy_names is populated
                # print(f"Skipping radar chart for {lang} - {reasoning}: No model data found after filtering strategies.")
                continue

            # Prepare for plotting
            num_vars = len(all_strategy_names) # These are the axes
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles_closed = angles + angles[:1] # Close the polygon

            fig, ax = plt.subplots(figsize=(13, 10), subplot_kw=dict(polar=True)) # Wider for legend

            # Get a list of distinct colors
            try:
                if len(model_data) <= 10:
                    color_cycle = plt.cm.get_cmap('tab10').colors
                elif len(model_data) <= 20:
                    color_cycle = plt.cm.get_cmap('tab20').colors
                else: # For more than 20, cycle through tab20
                    color_cycle = plt.cm.get_cmap('tab20').colors
            except AttributeError: # Fallback for older matplotlib or if specific cmap not found
                # Default to standard prop cycle colors
                prop_cycle = plt.rcParams['axes.prop_cycle']
                color_cycle = prop_cycle.by_key()['color']


            model_index = 0
            for model_name, strat_accuracies_map in model_data.items():
                # Get accuracies in the order of all_strategy_names, using 0 for missing ones
                values = [strat_accuracies_map.get(s_name, 0) for s_name in all_strategy_names]
                values_closed = values + values[:1] # Close the polygon
                
                current_color = color_cycle[model_index % len(color_cycle)]

                ax.plot(angles_closed, values_closed, linewidth=1.5, linestyle='solid', label=model_name, color=current_color)
                ax.fill(angles_closed, values_closed, color=current_color, alpha=0.15) # Light fill
                model_index += 1

            ax.set_xticks(angles)
            ax.set_xticklabels(all_strategy_names, fontsize=10)
            
            ax.set_ylim(0, 100) # Accuracy from 0 to 100
            ax.set_yticks(np.arange(0, 101, 20)) # Grid lines every 20%
            ax.set_yticklabels([f"{i}%" for i in np.arange(0, 101, 20)], fontsize=8)
            ax.set_rlabel_position(30) # Position of radial labels

            plt.title(f"Model Performance Radar: {lang} - {reasoning}", size=16, y=1.08) # Adjust title position
            
            # Add legend to the right of the plot
            if model_data: # Only add legend if there's data
                ax.legend(title="Models", loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize='small', frameon=True)

            safe_lang = lang.replace(" ", "_").replace("/", "_")
            safe_reasoning = reasoning.replace(" ", "_").replace("/", "_")
            
            plot_filename = f"radar_models_{safe_lang}_{safe_reasoning}_performance.png" # Updated filename
            plot_path = os.path.join(base_output_path, plot_filename)
            
            try:
                plt.savefig(plot_path, bbox_inches='tight') # bbox_inches='tight' helps fit the legend
                print(f"Saved radar chart: {plot_path}")
            except Exception as e:
                print(f"Error saving radar chart {plot_path}: {e}")
            finally:
                plt.close(fig) # Close the figure to free memory

def analyze_and_plot_model_performance(folder_path, output_graph_dir):
    """
    Analyzes model performance from JSON files and generates plots.
    Args:
        folder_path (str): Path to the folder containing JSON result files.
        output_graph_dir (str): Directory to save the generated graphs.
    """
    all_results = {} # lang -> reasoning_type -> strategy -> [results]
    
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
                accuracy_by_subject = data.get("accuracy_by_subject", {})
                accuracy_by_ability = data.get("accuracy_by_ability", {})

                result_entry = {
                    "model_name": model_name,
                    "original_filename": filename,
                    "accuracy": accuracy, # This is the one calculated from CSV by new_results.py
                    "metrics_data": metrics_data, # Contains overall_accuracy, F1s, report from _metrics.json
                    "accuracy_by_subject": accuracy_by_subject,
                    "accuracy_by_ability": accuracy_by_ability
                }

                # Populate the all_results structure
                all_results.setdefault(language, {}).setdefault(reasoning_type, {}).setdefault(strategy, []).append(result_entry)

            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                
    if not all_results:
        print(f"\nNo valid categorized JSON files with 'overall_accuracy' found in '{folder_path}'.")
        return

    # Call the existing bar chart plotting function
    plot_model_performance(all_results, output_graph_dir)

    # Call the new radar chart plotting function
    plot_radar_performance(all_results, output_graph_dir)

    # Print summary (optional, can be removed if only graphs are needed)
    found_any_results_for_print = False
    for lang, reasoning_types_map in all_results.items():
        has_data_for_lang = any(
            any(bool(models) for models in strategies_map.values()) 
            for strategies_map in reasoning_types_map.values()
        )
        if not has_data_for_lang:
            continue
        
        found_any_results_for_print = True
        print(f"\n\n--- {lang.upper()} MODELS PERFORMANCE SUMMARY ---")
        
        for reasoning, strategies_map in reasoning_types_map.items():
            has_data_for_reasoning = any(bool(models) for models in strategies_map.values())
            if not has_data_for_reasoning:
                continue

            print(f"\nReasoning Type: {reasoning}")
            
            for strat, models_list in strategies_map.items():
                if not models_list:
                    continue

                print(f"  Strategy: {strat}")
                header = f"{'Rank':<5} {'Model Name':<65} {'Accuracy (%)':<15}"
                print(f"    {'':<2}" + "-" * len(header)) 
                print(f"    {'':<2}{header}")
                print(f"    {'':<2}" + "-" * len(header))
                
                sorted_models = sorted(models_list, key=lambda x: x["accuracy"], reverse=True)
                
                for i, res in enumerate(sorted_models):
                    print(f"    {i+1:<5} {res['model_name']:<65} {res['accuracy']:<15.2f}")
                print(f"    {'':<2}" + "-" * len(header))

    if not found_any_results_for_print and all_results: # Check if all_results was populated but print didn't run
        print("\nData processed and plots generated (if any), but no summary printed due to filtering.")

if __name__ == "__main__":
    project_root = "/home/youssef/Projects/ArabicMMLU" 
    target_folder = os.path.join(project_root, "test5") 
    output_graph_folder = os.path.join(project_root, "output", "graphs")
    os.makedirs(output_graph_folder, exist_ok=True) # Ensure it exists
    
    analyze_and_plot_model_performance(target_folder, output_graph_folder)
