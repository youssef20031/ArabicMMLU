import csv
import os

# Mapping from index to answer letter
answer_map = {0: "أ", 1: "ب", 2: "ج", 3: "د"}

correct_count = 0
total = 0
subject_stats = {}  # Dictionary to hold per subject stats

csv_file = 'output/result_prompt_en_alpa_ar_cot_AraT5v2-base-1024.csv'
with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Get subject for each row, defaulting to unknown if missing
        s = row.get("subject", "unknown").strip()
        golds = row['golds'].strip()  # e.g. "2"
        preds = row['preds'].strip()  # e.g. "ج"
        try:
            gold_index = int(golds)
        except ValueError:
            continue
        total += 1
        is_correct = (answer_map.get(gold_index) == preds)
        if is_correct:
            correct_count += 1

        # Update subject-specific stats
        if s not in subject_stats:
            subject_stats[s] = {"total": 0, "correct": 0}
        subject_stats[s]["total"] += 1
        if is_correct:
            subject_stats[s]["correct"] += 1

# Calculate overall percentage; protect against division by zero.
overall_percentage = (correct_count / total * 100) if total > 0 else 0
filename = csv_file.split('/')[-1]

# Write summary results to a CSV file; if file exists, append under existing rows.
summary_file = 'output/results_summary.csv'
file_exists = os.path.exists(summary_file)
with open(summary_file, 'a' if file_exists else 'w', encoding='utf-8', newline='') as outf:
    writer = csv.writer(outf)
    # Write header if file is new.
    if not file_exists:
        writer.writerow(["filename", "subject", "total_questions", "correct_predictions", "percentage_correct"])
    # Write overall summary as a row.
    writer.writerow([filename, "overall", total, correct_count, f"{overall_percentage:.2f}%"])
    # Write per subject summary rows.
    for subj, stats in subject_stats.items():
        subj_total = stats["total"]
        subj_correct = stats["correct"]
        subj_percentage = (subj_correct / subj_total * 100) if subj_total > 0 else 0
        writer.writerow([filename, subj, subj_total, subj_correct, f"{subj_percentage:.2f}%"])

print(f"Correct predictions: {correct_count} out of {total}")
print(f"Summary written to {summary_file}")