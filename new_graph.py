import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define abbreviations and their corresponding full names or prefixes
abbreviations = {
    "UOT": "Unexpected Outcome Test",
    "SIT": "Scalar Implicature Test",
    "PST": "Persuasion Story Task",
    "FBT": "False Belief Task",
    "AST": "Ambiguous Story Task",
    "HTT": "Hinting Task Test",
    "SST": "Strange Story Task",
    "FRT": "Faux-pas Recognition Test",
    "EMO": "Emotion:", # Prefix for abilities
    "DES": "Desire:",  # Prefix for abilities
    "INT": "Intention:",# Prefix for abilities
    "KNO": "Knowledge:",# Prefix for abilities
    "BEL": "Belief:",  # Prefix for abilities
    "NLC": "Non-Literal Communication:", # Prefix for abilities - Adjusted based on JSON keys
    "NLC_alt": "Non-literal communication:", # Alternative prefix found in JSON
}

# Define which abbreviations refer to subjects and which to ability prefixes
subject_abbr = ["UOT", "SIT", "PST", "FBT", "AST", "HTT", "SST", "FRT"]
ability_abbr = ["EMO", "DES", "INT", "KNO", "BEL", "NLC", "NLC_alt"] # Include both NLC variants

# Path to the results file
#results_file_path = Path("results_output/results.json")
results_file_path = Path("test/results.json")
#results_file_path = Path("test/resultsnew.json")
#results_file_path = Path("test/results2.json")

# Load the JSON data
try:
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Results file not found at {results_file_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {results_file_path}")
    exit()

accuracy_by_subject = data.get("accuracy_by_subject", {})
accuracy_by_ability = data.get("accuracy_by_ability", {})

# --- Extract metrics from the new structure ---
metrics_from_json = data.get("metrics_from_json", {})
overall_accuracy_from_metrics_json = metrics_from_json.get("overall_accuracy") # This is likely a float 0-1
macro_f1_from_metrics_json = metrics_from_json.get("macro_f1_score")
weighted_f1_from_metrics_json = metrics_from_json.get("weighted_f1_score")
classification_report_str = metrics_from_json.get("classification_report") # String version

# Prepare data for the chart
chart_data = {}

# Get subject accuracies
for abbr in subject_abbr:
    full_name = abbreviations.get(abbr)
    if full_name and full_name in accuracy_by_subject:
        # Ensure accuracy is float, round if necessary
        acc = accuracy_by_subject[full_name]
        chart_data[abbr] = round(float(acc), 2) if isinstance(acc, (int, float)) else np.nan # Use NaN for missing data
    else:
         chart_data[abbr] = np.nan # Use NaN for missing data

# Calculate and get average ability accuracies
combined_nlc_accuracies = []
processed_prefixes = set() # Keep track of prefixes processed to avoid double counting NLC

for abbr in ability_abbr:
    prefix = abbreviations.get(abbr)
    if not prefix or prefix in processed_prefixes:
        continue

    relevant_accuracies = [
        acc for ability, acc in accuracy_by_ability.items()
        if ability.startswith(prefix) and isinstance(acc, (int, float))
    ]

    # Special handling to combine both NLC prefixes
    if abbr in ["NLC", "NLC_alt"]:
        # Find the alternative prefix as well
        alt_abbr = "NLC_alt" if abbr == "NLC" else "NLC"
        alt_prefix = abbreviations.get(alt_abbr)
        if alt_prefix and alt_prefix not in processed_prefixes:
             relevant_accuracies.extend([
                acc for ability, acc in accuracy_by_ability.items()
                if ability.startswith(alt_prefix) and isinstance(acc, (int, float))
             ])
             processed_prefixes.add(alt_prefix) # Mark alternative as processed

        # Use "NLC" as the final key
        target_abbr = "NLC"
        if not relevant_accuracies:
            chart_data[target_abbr] = np.nan
        else:
            chart_data[target_abbr] = round(sum(relevant_accuracies) / len(relevant_accuracies), 2)
        processed_prefixes.add(prefix) # Mark current prefix as processed

    else: # Handle other abilities
        if not relevant_accuracies:
            chart_data[abbr] = np.nan
        else:
            chart_data[abbr] = round(sum(relevant_accuracies) / len(relevant_accuracies), 2)
        processed_prefixes.add(prefix) # Mark current prefix as processed


# --- Plotting Section ---

# Prepare data for plotting (filter out NaN, keep track of missing)
plot_labels = []
plot_values = []
missing_labels = []

# Ensure consistent order based on original lists
all_abbrs = subject_abbr + [a for a in ability_abbr if a != "NLC_alt"] # Use combined NLC key
ordered_chart_data = {abbr: chart_data.get(abbr, np.nan) for abbr in all_abbrs}


for abbr, acc in ordered_chart_data.items():
     if isinstance(acc, (int, float)) and not np.isnan(acc):
         plot_labels.append(abbr)
         plot_values.append(acc)
     else:
         missing_labels.append(abbr)


if not plot_labels:
    print("No valid data found to plot.")
    exit()

# Create the bar chart
plt.style.use('ggplot') # Use a style for better appearance
plt.figure(figsize=(14, 8)) # Adjust figure size for better label visibility
bars = plt.bar(plot_labels, plot_values, color='steelblue')

# Calculate and plot the overall average
if plot_values:
    overall_average = np.mean(plot_values) # This average is based on the plotted subject/ability categories
    plt.axhline(overall_average, color='red', linestyle='--', linewidth=2, label=f'Avg. of Plotted Categories ({overall_average:.1f}%)')
    plt.legend() # Show the legend to label the average line

plt.xlabel("Category Abbreviation", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)

# --- Updated Title to include F1 scores if available ---
main_title = "Accuracy by Subject and Ability Category"
sub_title_parts = []
if overall_accuracy_from_metrics_json is not None:
    sub_title_parts.append(f"Overall Acc: {overall_accuracy_from_metrics_json*100:.2f}%")
if macro_f1_from_metrics_json is not None:
    sub_title_parts.append(f"Macro F1: {macro_f1_from_metrics_json:.3f}")
if weighted_f1_from_metrics_json is not None:
    sub_title_parts.append(f"Weighted F1: {weighted_f1_from_metrics_json:.3f}")

if sub_title_parts:
    plt.title(f"{main_title}\n({' | '.join(sub_title_parts)})", fontsize=14, fontweight='bold')
else:
    plt.title(main_title, fontsize=14, fontweight='bold')
# --- End of Updated Title ---

plt.ylim(0, 100) # Set y-axis limit to 0-100 for percentage
plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate labels if they overlap
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}', va='bottom', ha='center', fontsize=9) # Adjust position slightly

# Add a note about missing data if any
if missing_labels:
    missing_text = f"Note: Data not available or invalid for {', '.join(missing_labels)}"
    plt.figtext(0.5, 0.01, missing_text, wrap=True, horizontalalignment='center', fontsize=9, color='grey')

# --- Add Classification Report to the bottom if available ---
if classification_report_str:
    plt.figtext(0.02, 0.01, "Classification Report (from _metrics.json):", fontsize=8, fontweight='bold')
    plt.figtext(0.02, -0.15, classification_report_str, wrap=True, horizontalalignment='left', fontsize=6, family='monospace', va='top')
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust rect bottom to make space for report
else:
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title/label overlap
# --- End of Adding Classification Report ---

# Save the plot to a file
output_image_path = Path("accuracy_chartnew.png")
try:
    plt.savefig(output_image_path, dpi=300) # Save with higher resolution
    print(f"Chart saved to {output_image_path}")
except Exception as e:
    print(f"Error saving chart: {e}")

# Optional: Display the plot interactively (comment out if not needed)
# plt.show()