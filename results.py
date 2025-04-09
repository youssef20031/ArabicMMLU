import csv

# Mapping from index to answer letter
answer_map = {0: "أ", 1: "ب", 2: "ج", 3: "د"}

correct_count = 0
total = 0

csv_file = 'output/result_prompt_en_alpa_ar_bloomz-560m.csv'
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

print(f"Correct predictions: {correct_count} out of {total}")