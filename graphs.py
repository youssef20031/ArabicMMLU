import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np  # Add this line


csv_path = os.path.join("output", "results_summary.csv")
# Read the CSV without header and then assign our column names.
df = pd.read_csv(csv_path, header=None, skiprows=1)
df.columns = ["filename", "subject", "total_questions", "correct_predictions", "percentage_correct"]

# Remove '%' and convert percentage_correct to float.
df["percentage_correct"] = df["percentage_correct"].str.rstrip('%').astype(float)

# Extract model info from filename.
def extract_model(filename):
    base = filename.replace("result_prompt_en_alpa_ar_", "").replace(".csv", "")
    return base

df["model"] = df["filename"].apply(extract_model)

# ---------------------------
# Graph 1: Overall performance per model
overall_df = df[df["subject"].str.lower() == "overall"].reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.bar(overall_df["model"], overall_df["percentage_correct"], color='skyblue')
plt.xlabel("Model")
plt.ylabel("Percentage Correct")
plt.title("Overall Performance Comparison between Models")

# Increase ylim based on max percentage:
max_val = overall_df["percentage_correct"].max()
plt.ylim(0, max(max_val * 1.15, 100))

plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels to avoid overlap

for idx, row in overall_df.iterrows():
    plt.text(idx, row["percentage_correct"] + 1, f'{row["percentage_correct"]:.1f}%', ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
plt.savefig("output/graph_overall.png")
plt.close()

# ---------------------------
# Graph 2: Per-subject performance comparison between models
subject_df = df[df["subject"].str.lower() != "overall"]
pivot_df = subject_df.pivot(index="subject", columns="model", values="percentage_correct")
ax = pivot_df.plot(kind="bar", figsize=(10, 7), colormap="tab20b")
plt.xlabel("Subject")
plt.ylabel("Percentage Correct")
plt.title("Per-Subject Performance Comparison Between Models")
plt.ylim(0, 100)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("output/graph_subjects.png")
plt.close()
# ---------------------------
# Graph 3 alternative: Per-subject performance as a heatmap with average
subject_df = df[df["subject"].str.lower() != "overall"]
pivot_df = subject_df.pivot(index="subject", columns="model", values="percentage_correct")

# Calculate the average for each subject
avg = pivot_df.mean(axis=1)

# Save original model columns order
model_columns = pivot_df.columns.tolist()

# Add a blank column to separate the model columns from the average column
pivot_df[" "] = np.nan

# Add the average column using the calculated average
pivot_df["Average"] = avg

# Reorder columns: original model columns, then blank column, then average
pivot_df = pivot_df[model_columns + [" ", "Average"]]

plt.figure(figsize=(14, 8))
sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Percentage Correct"})
plt.xlabel("Model / Average")
plt.ylabel("Subject")
plt.title("Per-Subject Performance Comparison Across Models (with Average)")
plt.tight_layout()
plt.savefig("output/graph_subjects_heatmap_with_avg.png")
plt.close()

print("Heatmap with average saved to output/graph_subjects_heatmap_with_avg.png")

print("Graphs saved to the output folder:")
print(" - output/graph_overall.png")
print(" - output/graph_subjects.png")
print(" - output/graph_subjects_heatmap.png")