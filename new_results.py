import csv
import argparse
import json
import os
import glob
from collections import Counter

def extract_answer(text):
    """
    Extracts the answer choice (A, B, C, D) from the model's raw output,
    handling both English (A,B,C,D) and Arabic (أ,ب,ج,د) formats.
    Maps Arabic choices to their English equivalents.
    """
    if not isinstance(text, str):
        return "A" # Default or handle error appropriately

    # Prioritize specific markers (English and Arabic)
    if "[[A]]" in text or "[[أ]]" in text: return "A"
    if "[[B]]" in text or "[[ب]]" in text: return "B"
    if "[[C]]" in text or "[[ج]]" in text: return "C"
    if "[[D]]" in text or "[[د]]" in text: return "D"
    if "[A]" in text or "[أ]" in text: return "A"
    if "[B]" in text or "[ب]" in text: return "B"
    if "[C]" in text or "[ج]" in text: return "C"
    if "[D]" in text or "[د]" in text: return "D"

    # Look for the last occurrence of A, B, C, D or أ, ب, ج, د if no markers found
    for char in reversed(text):
        if char == 'A' or char == 'أ': return "A"
        if char == 'B' or char == 'ب': return "B"
        if char == 'C' or char == 'ج': return "C"
        if char == 'D' or char == 'د': return "D"

    return "A" # Default if no answer found

# Mapping from Arabic choices to English
ARABIC_TO_ENGLISH_MAP = {
    'أ': 'A',
    'ب': 'B',
    'ج': 'C',
    'د': 'D'
}

# Mapping from numerical index (as string) to English
INDEX_TO_ENGLISH_MAP = {
    '0': 'A',
    '1': 'B',
    '2': 'C',
    '3': 'D'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from CSV results in a folder. Generates a separate JSON result file for each input CSV.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input CSV files.")
    parser.add_argument("--output_json", type=str, default="results_output/results.json", help="Path for output. The directory of this path (e.g., 'results_output/' from 'results_output/results.json') will be used to store individual JSON files named after input CSVs.")
    args = parser.parse_args()

    # Determine and create output directory
    output_dir = os.path.dirname(args.output_json)
    if output_dir == "":  # If args.output_json is just a filename like "results.json"
        output_dir = "."  # Default to current directory
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    elif not os.path.isdir(output_dir):
        print(f"Error: Output path {output_dir} exists but is not a directory.")
        exit(1)

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder not found at {args.input_folder}")
        exit(1)

    csv_files = glob.glob(os.path.join(args.input_folder, '*.csv'))

    if not csv_files:
        print(f"Error: No CSV files found in the folder {args.input_folder}")
        exit(1)

    print(f"Found {len(csv_files)} CSV files to process in {args.input_folder}")
    processed_files_count = 0
    successfully_generated_files = 0

    for csv_filepath in csv_files:
        print(f"Processing file: {csv_filepath}...")
        
        # Initialize accumulators for the current file
        acc_per_subject_file = {}
        cnt_per_subject_file = Counter()
        acc_per_ability_file = {}
        cnt_per_ability_file = Counter()
        file_processed_rows = 0

        try:
            with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                expected_columns = ['golds', 'raw_preds', 'subject', 'ABILITY']
                if not reader.fieldnames: # Handle empty CSV
                    print(f"Warning: Skipping empty or header-only file: {csv_filepath}")
                    continue
                if not all(col in reader.fieldnames for col in expected_columns):
                    missing = [col for col in expected_columns if col not in reader.fieldnames]
                    print(f"Warning: Missing required columns in {csv_filepath}: {', '.join(missing)}. Skipping file.")
                    continue 

                for row_num, row in enumerate(reader, 1):
                    try:
                        gold_answer_raw = row.get('golds', '').strip()
                        gold_answer = None

                        if gold_answer_raw in INDEX_TO_ENGLISH_MAP:
                            gold_answer = INDEX_TO_ENGLISH_MAP[gold_answer_raw]
                        elif gold_answer_raw in ARABIC_TO_ENGLISH_MAP:
                            gold_answer = ARABIC_TO_ENGLISH_MAP[gold_answer_raw]
                        else:
                            gold_answer = gold_answer_raw.upper() # Assume it might be a,b,c,d

                        raw_prediction_text = row.get('raw_preds', '')
                        subject = row.get('subject', 'unknown_subject').strip()
                        ability = row.get('ABILITY', 'unknown_ability').strip()

                        if not ability: ability = 'unknown_ability'
                        if not subject: subject = 'unknown_subject'

                        if gold_answer not in ['A', 'B', 'C', 'D']:
                            print(f"Warning: Invalid gold answer '{gold_answer_raw}' (processed as '{gold_answer}') found in {os.path.basename(csv_filepath)} at row {row_num}. Skipping row.")
                            continue

                        predicted_answer = extract_answer(raw_prediction_text)

                        cnt_per_subject_file[subject] += 1
                        cnt_per_ability_file[ability] += 1
                        file_processed_rows +=1

                        if predicted_answer == gold_answer:
                            acc_per_subject_file[subject] = acc_per_subject_file.get(subject, 0) + 1
                            acc_per_ability_file[ability] = acc_per_ability_file.get(ability, 0) + 1
                    
                    except Exception as e_row:
                        print(f"Error processing row {row_num} in {os.path.basename(csv_filepath)}: {e_row}. Row data: {row}")
                        continue # Skip to the next row
            
            processed_files_count += 1 # Count as processed even if no valid data rows to generate JSON

            if file_processed_rows == 0: # Check if any rows were actually processed for data
                print(f"Warning: No valid data rows processed from {os.path.basename(csv_filepath)}. Skipping JSON generation for this file.")
                continue

            # Calculate final accuracies for the current file
            final_acc_subject_file = {}
            for subject_key, correct_count in acc_per_subject_file.items():
                total_count = cnt_per_subject_file.get(subject_key, 0)
                final_acc_subject_file[subject_key] = round((correct_count / total_count) * 100, 2) if total_count > 0 else 0.0
            for subject_key in cnt_per_subject_file:
                if subject_key not in final_acc_subject_file:
                    final_acc_subject_file[subject_key] = 0.0

            final_acc_ability_file = {}
            for ability_key, correct_count in acc_per_ability_file.items():
                total_count = cnt_per_ability_file.get(ability_key, 0)
                final_acc_ability_file[ability_key] = round((correct_count / total_count) * 100, 2) if total_count > 0 else 0.0
            for ability_key in cnt_per_ability_file:
                if ability_key not in final_acc_ability_file:
                    final_acc_ability_file[ability_key] = 0.0
            
            total_correct_file = sum(acc_per_subject_file.values())
            total_questions_file = sum(cnt_per_subject_file.values())
            overall_accuracy_file = round((total_correct_file / total_questions_file) * 100, 2) if total_questions_file > 0 else 0.0

            results_file = {
                "source_csv": os.path.basename(csv_filepath),
                "overall_accuracy": overall_accuracy_file,
                "accuracy_by_subject": dict(sorted(final_acc_subject_file.items())),
                "accuracy_by_ability": dict(sorted(final_acc_ability_file.items())),
                "counts_by_subject": dict(sorted(cnt_per_subject_file.items())),
                "counts_by_ability": dict(sorted(cnt_per_ability_file.items()))
            }

            base_csv_name = os.path.basename(csv_filepath)
            output_filename = os.path.splitext(base_csv_name)[0] + ".json"
            current_output_json_path = os.path.join(output_dir, output_filename)

            try:
                with open(current_output_json_path, "w", encoding='utf-8') as f:
                    json.dump(results_file, f, ensure_ascii=False, indent=4)
                print(f"Results for {base_csv_name} successfully saved to {current_output_json_path}")
                print(f"Overall Accuracy for {base_csv_name}: {overall_accuracy_file}%")
                successfully_generated_files += 1
            except Exception as e_json:
                print(f"An error occurred while writing the results JSON for {base_csv_name}: {e_json}")

        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {csv_filepath}. Skipping.")
            continue
        except Exception as e_file:
            print(f"An error occurred while processing the CSV file {csv_filepath}: {e_file}. Skipping.")
            continue

    print(f"\n--- Summary ---")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"CSV files attempted to process: {processed_files_count}")
    print(f"JSON result files successfully generated: {successfully_generated_files}")

   