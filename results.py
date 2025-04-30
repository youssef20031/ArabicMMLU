import csv
import os
import glob # Import glob to find files

# Mapping from index to answer letter (Arabic)
arabic_answer_map = {0: "أ", 1: "ب", 2: "ج", 3: "د"}
# Mapping from index to answer letter (English)
english_answer_map = {0: "a", 1: "b", 2: "c", 3: "d"}

# Directory containing the result CSV files
input_dir = 'english_output'
# Path for the summary file
summary_file = os.path.join(input_dir, 'results_summary.csv')

# Find all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
# Exclude the summary file itself if it exists and matches the pattern
csv_files = [f for f in csv_files if os.path.basename(f) != os.path.basename(summary_file)]

# Overwrite the summary file each time the script runs
with open(summary_file, 'w', encoding='utf-8', newline='') as outf:
    writer = csv.writer(outf)
    # Write header once at the beginning
    writer.writerow(["filename", "subject", "total_questions", "correct_predictions", "percentage_correct"])

    # Process each result CSV file
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        correct_count = 0
        total = 0
        subject_stats = {}  # Reset stats for each file

        try:
            with open(csv_file, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Get subject for each row, defaulting to unknown if missing
                    s = row.get("subject", "unknown").strip()
                    golds = row.get('golds', '').strip()  # Use .get for safety
                    preds = row.get('preds', '').strip().lower() # Use .get for safety and convert preds to lowercase

                    # Skip row if golds or preds are missing/empty
                    if not golds or not preds:
                        print(f"Skipping row due to missing 'golds' or 'preds' in {csv_file}: {row}")
                        continue

                    try:
                        gold_index = int(golds)
                    except ValueError:
                        print(f"Skipping row due to invalid 'golds' value in {csv_file}: {row}")
                        continue # Skip row if gold index is not an integer

                    total += 1
                    # Check if prediction matches either Arabic or English mapping
                    expected_arabic = arabic_answer_map.get(gold_index)
                    expected_english = english_answer_map.get(gold_index)
                    is_correct = (preds == expected_arabic or preds == expected_english)

                    if is_correct:
                        correct_count += 1

                    # Update subject-specific stats
                    if s not in subject_stats:
                        subject_stats[s] = {"total": 0, "correct": 0}
                    subject_stats[s]["total"] += 1
                    if is_correct:
                        subject_stats[s]["correct"] += 1

            # Calculate overall percentage for the current file; protect against division by zero.
            overall_percentage = (correct_count / total * 100) if total > 0 else 0
            filename = os.path.basename(csv_file) # Get just the filename

            # Write overall summary for the current file as a row.
            writer.writerow([filename, "overall", total, correct_count, f"{overall_percentage:.2f}%"])

            # Write per subject summary rows for the current file.
            # Sort subjects alphabetically for consistent output
            for subj in sorted(subject_stats.keys()):
                stats = subject_stats[subj]
                subj_total = stats["total"]
                subj_correct = stats["correct"]
                subj_percentage = (subj_correct / subj_total * 100) if subj_total > 0 else 0
                writer.writerow([filename, subj, subj_total, subj_correct, f"{subj_percentage:.2f}%"])

            print(f"Finished processing {filename}. Overall Accuracy: {overall_percentage:.2f}%")

        except FileNotFoundError:
            print(f"Error: File not found {csv_file}")
        except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")


print(f"\nSummary for all processed files written to {summary_file}")