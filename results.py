import csv
import os

# Mapping from index to answer letter
answer_map = {0: "أ", 1: "ب", 2: "ج", 3: "د"}

correct_count = 0
total = 0

csv_file = 'output/result_prompt_en_alpa_ar_mt0-base.csv'
with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        golds = row['golds'].strip()  # e.g. "2"
        preds = row['preds'].strip()  # e.g. "ج"
        try:
            gold_index = int(golds)
        except ValueError:
            continue
        total += 1
        if answer_map.get(gold_index) == preds:
            correct_count += 1

# Calculate percentage; protect against division by zero.
percentage = (correct_count / total * 100) if total > 0 else 0
filename = csv_file.split('/')[-1]

# Write summary results to a CSV file; if file exists, append under existing rows.
summary_file = 'output/results_summary.csv'
file_exists = os.path.exists(summary_file)
with open(summary_file, 'a' if file_exists else 'w', encoding='utf-8', newline='') as outf:
    writer = csv.writer(outf)
    if not file_exists:
        writer.writerow(["filename", "total_questions", "correct_predictions", "percentage_correct"])
    writer.writerow([filename, total, correct_count, f"{percentage:.2f}%"])

print(f"Correct predictions: {correct_count} out of {total}")
print(f"Summary written to {summary_file}")