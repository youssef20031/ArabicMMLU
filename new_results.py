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
    parser = argparse.ArgumentParser(description="Calculate accuracy from CSV results in a folder.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input CSV files.")
    parser.add_argument("--output_json", type=str, default="results_output/results.json", help="Path to save the results JSON file.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


    acc_per_subject = {}
    cnt_per_subject = Counter()
    acc_per_ability = {}
    cnt_per_ability = Counter()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder not found at {args.input_folder}")
        exit(1)

    csv_files = glob.glob(os.path.join(args.input_folder, '*.csv'))

    if not csv_files:
        print(f"Error: No CSV files found in the folder {args.input_folder}")
        exit(1)

    print(f"Found {len(csv_files)} CSV files to process in {args.input_folder}")

    for csv_filepath in csv_files:
        print(f"Processing file: {csv_filepath}")
        try:
            with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                # Verify expected columns exist
                expected_columns = ['golds', 'raw_preds', 'subject', 'ABILITY']
                if not reader.fieldnames: # Handle empty CSV
                    print(f"Warning: Skipping empty or header-only file: {csv_filepath}")
                    continue
                if not all(col in reader.fieldnames for col in expected_columns):
                    missing = [col for col in expected_columns if col not in reader.fieldnames]
                    print(f"Warning: Missing required columns in {csv_filepath}: {', '.join(missing)}. Skipping file.")
                    continue # Skip this file

                for row_num, row in enumerate(reader, 1): # Add row number for better error reporting
                    try:
                        gold_answer_raw = row.get('golds', '').strip()
                        gold_answer = None

                        # Try mapping from index first
                        if gold_answer_raw in INDEX_TO_ENGLISH_MAP:
                            gold_answer = INDEX_TO_ENGLISH_MAP[gold_answer_raw]
                        # Else, try mapping from Arabic letter
                        elif gold_answer_raw in ARABIC_TO_ENGLISH_MAP:
                            gold_answer = ARABIC_TO_ENGLISH_MAP[gold_answer_raw]
                        # Else, assume it might already be English A, B, C, D
                        else:
                            gold_answer = gold_answer_raw

                        raw_prediction_text = row.get('raw_preds', '')
                        subject = row.get('subject', 'unknown_subject').strip()
                        ability = row.get('ABILITY', 'unknown_ability').strip()

                        # Ensure ability is not empty, provide a default if necessary
                        if not ability:
                            ability = 'unknown_ability'
                        # Ensure subject is not empty
                        if not subject:
                            subject = 'unknown_subject'

                        # Ensure gold answer is one of A, B, C, D after mapping attempts
                        if gold_answer not in ['A', 'B', 'C', 'D']:
                             print(f"Warning: Invalid gold answer '{gold_answer_raw}' (processed as '{gold_answer}') found in {os.path.basename(csv_filepath)} at row {row_num}. Skipping row.")
                             continue


                        predicted_answer = extract_answer(raw_prediction_text) # This now handles Arabic choices

                        # Increment counts
                        cnt_per_subject[subject] += 1
                        cnt_per_ability[ability] += 1

                        # Check for correctness (now comparing English A,B,C,D)
                        if predicted_answer == gold_answer:
                            acc_per_subject[subject] = acc_per_subject.get(subject, 0) + 1
                            acc_per_ability[ability] = acc_per_ability.get(ability, 0) + 1
                        # else: # Optional: Debugging incorrect predictions
                        #     print(f"Debug: Row {row_num} - Gold: {gold_answer}, Pred: {predicted_answer} (Raw: {raw_prediction_text[:50]}...)")


                    except Exception as e_row:
                         print(f"Error processing row {row_num} in {os.path.basename(csv_filepath)}: {e_row}. Row data: {row}")
                         continue # Skip to the next row


        except FileNotFoundError:
            # This shouldn't happen with glob, but good practice
            print(f"Error: Input CSV file not found at {csv_filepath}")
            continue
        except Exception as e_file:
            print(f"An error occurred while processing the CSV file {csv_filepath}: {e_file}")
            continue # Continue with the next file

    # Check if any data was processed
    if not cnt_per_subject and not cnt_per_ability:
        print("Error: No valid data processed from any CSV file.")
        exit(1)


    # Calculate final accuracies
    final_acc_subject = {}
    for subject, correct_count in acc_per_subject.items():
        total_count = cnt_per_subject.get(subject, 0)
        if total_count > 0:
            final_acc_subject[subject] = round((correct_count / total_count) * 100, 2)
        else:
             final_acc_subject[subject] = 0.0

    # Add subjects with 0 accuracy if they had entries but no correct answers
    for subject in cnt_per_subject:
        if subject not in final_acc_subject:
            final_acc_subject[subject] = 0.0


    final_acc_ability = {}
    for ability, correct_count in acc_per_ability.items():
        total_count = cnt_per_ability.get(ability, 0)
        if total_count > 0:
            final_acc_ability[ability] = round((correct_count / total_count) * 100, 2)
        else:
            final_acc_ability[ability] = 0.0

    # Add abilities with 0 accuracy if they had entries but no correct answers
    for ability in cnt_per_ability:
        if ability not in final_acc_ability:
             final_acc_ability[ability] = 0.0

    # Overall Accuracy
    total_correct = sum(acc_per_subject.values())
    total_questions = sum(cnt_per_subject.values())
    overall_accuracy = round((total_correct / total_questions) * 100, 2) if total_questions > 0 else 0.0

    results = {
        "overall_accuracy": overall_accuracy,
        "accuracy_by_subject": dict(sorted(final_acc_subject.items())),
        "accuracy_by_ability": dict(sorted(final_acc_ability.items())),
        "counts_by_subject": dict(sorted(cnt_per_subject.items())),
        "counts_by_ability": dict(sorted(cnt_per_ability.items()))
    }

    # Write results to JSON
    try:
        with open(args.output_json, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results successfully calculated from {len(csv_files)} file(s) and saved to {args.output_json}")
        print(f"Overall Accuracy: {overall_accuracy}%") # Print overall accuracy
    except Exception as e:
        print(f"An error occurred while writing the results JSON file: {e}")
        exit(1)