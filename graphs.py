import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np  # Add this line
import pandas as pd

# --- Configuration ---
output_dir = "english_output" # Define the output directory here
# --- End Configuration ---

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "results_summary.csv")
# Read the CSV without header and then assign our column names.
df = pd.read_csv(csv_path, header=None, skiprows=1)
df.columns = ["filename", "subject", "total_questions", "correct_predictions", "percentage_correct"]

# Remove '%' and convert percentage_correct to float.
df["percentage_correct"] = df["percentage_correct"].str.rstrip('%').astype(float)

def extract_model(filename):
    # Remove the common prefix and the suffix
    prefix = "result_prompt_en_alpa_"
    suffix = ".csv"
    if filename.startswith(prefix):
        base = filename[len(prefix):]
    else:
        base = filename # Keep original if prefix not found

    if base.endswith(suffix):
        base = base[:-len(suffix)]

    # Remove the language code part (e.g., 'en_' or 'ar_')
    parts = base.split('_', 1) # Split only on the first underscore
    # Check if the first part looks like a 2-letter language code
    if len(parts) == 2 and len(parts[0]) == 2:
         base = parts[1] # Keep the part after the language code

    return base

df["model"] = df["filename"].apply(extract_model)
print("Unique extracted model names:\n", df['model'].unique())


# ---------------------------
# Graph 1: Overall performance per model
overall_df = df[df["subject"].str.lower() == "overall"].reset_index(drop=True)

def sort_key(m):
    m_lower = m.lower()
    if m_lower.startswith("cot_"):
        base = m_lower[4:]
        return (base, 1)
    elif "-cot" in m_lower:
        base = m_lower.replace("-cot", "")
        return (base, 1)
    else:
        return (m_lower, 0)

ordered_models = sorted(overall_df["model"].unique(), key=sort_key)
overall_df = overall_df.set_index("model").loc[ordered_models].reset_index()

plt.figure(figsize=(12, 6))
plt.bar(overall_df["model"], overall_df["percentage_correct"], color='skyblue')
plt.xlabel("Model")
plt.ylabel("Percentage Correct")
plt.title("Overall Performance Comparison between Models")

# Increase ylim based on max percentage:
max_val = overall_df["percentage_correct"].max()
plt.ylim(0, max(max_val * 1.5, 100))

plt.xticks(rotation=30, ha="right")  # Rotate x-axis labels to avoid overlap

for idx, row in overall_df.iterrows():
    plt.text(idx, row["percentage_correct"] + 1, f'{row["percentage_correct"]:.1f}%', ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
graph1_path = os.path.join(output_dir, "graph_overall.png")
plt.savefig(graph1_path)
plt.close()

# ---------------------------
# Graph 2: Per-subject performance comparison between models
subject_df = df[df["subject"].str.lower() != "overall"]
pivot_df_graph2 = subject_df.pivot(index="subject", columns="model", values="percentage_correct") # Renamed to avoid conflict
ax = pivot_df_graph2.plot(kind="bar", figsize=(10, 7), colormap="tab20b")
plt.xlabel("Subject")
plt.ylabel("Percentage Correct")
plt.title("Per-Subject Performance Comparison Between Models")
plt.ylim(0, 100)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
graph2_path = os.path.join(output_dir, "graph_subjects.png")
plt.savefig(graph2_path)
plt.close()
# ---------------------------
# Graph 3: Per-subject performance as a heatmap with average
subject_df_graph3 = df[df["subject"].str.lower() != "overall"] # Use a distinct df variable
pivot_df_graph3 = subject_df_graph3.pivot(index="subject", columns="model", values="percentage_correct")

# Calculate the average for each subject
avg = pivot_df_graph3.mean(axis=1)

# Use the same sort_key function defined earlier
model_columns = sorted(pivot_df_graph3.columns.tolist(), key=sort_key)


# Add a blank column to separate the model columns from the average column
pivot_df_graph3[" "] = np.nan

# Add the average column using the calculated average
pivot_df_graph3["Average"] = avg

# Reorder columns: original model columns, then blank column, then average
pivot_df_graph3 = pivot_df_graph3[model_columns + [" ", "Average"]]

plt.figure(figsize=(14, 10))
hm_ax = sns.heatmap(pivot_df_graph3, annot=True, fmt=".1f", cmap="YlGnBu",  cbar_kws={"label": "Percentage Correct"})
plt.xlabel("Model / Average")
plt.ylabel("Subject")
plt.title("Per-Subject Performance Comparison Across Models (Heatmap with Average)") # Clarified title
hm_ax.set_yticklabels(hm_ax.get_yticklabels(), rotation=0)  # Ensure y-axis labels are horizontal
plt.tight_layout()
graph3_path = os.path.join(output_dir, "graph_subjects_heatmap_with_avg.png")
plt.savefig(graph3_path)
plt.close()

# ---------------------------
# Graph 4: Comparison between Base and CoT models per subject (Detailed Table)

print("\n--- Base vs CoT Model Comparison per Subject (Excluding Ties) ---")

subject_df_graph4 = df[df["subject"].str.lower() != "overall"] # Use a distinct df variable
all_subjects = subject_df_graph4['subject'].unique()
comparison_data = []
subject_summary_data = [] # List to store summary counts per subject

for subj in all_subjects:
    subject_data = subject_df_graph4[subject_df_graph4['subject'] == subj]
    models_in_subject = subject_data['model'].tolist()
    scores_in_subject = subject_data.set_index('model')['percentage_correct']

    base_models = {}
    cot_models = {}
    model_map_lower_to_orig = {m.lower(): m for m in models_in_subject} # Map lowercase to original case

    # Initialize counters for this subject
    cot_better_count = 0
    base_better_count = 0

    # Identify base and CoT models
    for model_lower, model_orig in model_map_lower_to_orig.items():
        if model_lower.startswith("cot_"):
            base_key = model_lower[4:]
            if base_key in model_map_lower_to_orig: # Check if base exists
                 cot_models[base_key] = model_orig
        elif "-cot" in model_lower:
            base_key = model_lower.replace("-cot", "")
            if base_key in model_map_lower_to_orig: # Check if base exists
                cot_models[base_key] = model_orig
        else:
            potential_cot_key1 = f"cot_{model_lower}"
            potential_cot_key2 = f"{model_lower}-cot"
            if potential_cot_key1 in model_map_lower_to_orig or potential_cot_key2 in model_map_lower_to_orig:
                 base_models[model_lower] = model_orig

    # Perform comparison for found pairs
    for base_key, base_model_orig in base_models.items():
        if base_key in cot_models:
            cot_model_orig = cot_models[base_key]
            base_score = scores_in_subject.get(base_model_orig, 0)
            cot_score = scores_in_subject.get(cot_model_orig, 0)

            # Only add to comparison if scores are different
            if cot_score > base_score:
                better_model = cot_model_orig
                difference = cot_score - base_score
                comparison_data.append({
                    'Subject': subj,
                    'Model 1 (Base)': base_model_orig, 'Score 1': base_score,
                    'Model 2 (CoT)': cot_model_orig, 'Score 2': cot_score,
                    'Better Model': better_model, 'Difference (%)': difference
                })
                cot_better_count += 1 # Increment CoT counter
            elif base_score > cot_score:
                better_model = base_model_orig
                difference = base_score - cot_score
                comparison_data.append({
                    'Subject': subj,
                    'Model 1 (Base)': base_model_orig, 'Score 1': base_score,
                    'Model 2 (CoT)': cot_model_orig, 'Score 2': cot_score,
                    'Better Model': better_model, 'Difference (%)': difference
                })
                base_better_count += 1 # Increment Base counter
            # Cases where base_score == cot_score are excluded

    # Store the summary counts for this subject if there were any comparisons
    if cot_better_count > 0 or base_better_count > 0:
         subject_summary_data.append({
             'Subject': subj,
             'CoT Better Count': cot_better_count,
             'ToM Better Count': base_better_count, # Renamed for clarity
         })


# --- Create and Save the Detailed Comparison Table Plot ---
graph4_path = None # Initialize path variable
if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by=['Subject', 'Model 1 (Base)'])

    # Format data for display *before* plotting
    plot_df_comp = comparison_df.copy() # Use distinct variable name
    plot_df_comp['Score 1'] = plot_df_comp['Score 1'].map('{:.1f}%'.format)
    plot_df_comp['Score 2'] = plot_df_comp['Score 2'].map('{:.1f}%'.format)
    plot_df_comp['Difference (%)'] = plot_df_comp['Difference (%)'].map('{:.1f}'.format)

    # Create figure and axes
    fig_comp, ax_comp = plt.subplots(figsize=(12, len(plot_df_comp) * 0.4 + 1)) # Adjust size
    ax_comp.axis('tight')
    ax_comp.axis('off')

    # Create the table
    table_comp = ax_comp.table(cellText=plot_df_comp.values, colLabels=plot_df_comp.columns, cellLoc='center', loc='center')
    table_comp.auto_set_font_size(False)
    table_comp.set_fontsize(9)
    table_comp.scale(1.1, 1.1)

    plt.title("Base vs CoT Model Comparison per Subject (Excluding Ties)", y=1.02)
    plt.tight_layout()
    graph4_path = os.path.join(output_dir, "graph_base_vs_cot_comparison.png")
    plt.savefig(graph4_path, bbox_inches='tight', dpi=150)
    plt.close(fig_comp)
    print(f"Base vs CoT comparison table saved to {graph4_path}")

else:
    print("No Base/CoT model pairs found with differing scores for comparison.")


# ---------------------------
# Graph 5: Summary Counts Table Plot
graph5_path = None # Initialize path variable
if subject_summary_data:
    summary_df = pd.DataFrame(subject_summary_data)
    summary_df = summary_df.sort_values(by='Subject')

    # --- Calculate and Add Overall Average Row ---
    if not summary_df.empty:
        avg_cot_better = summary_df['CoT Better Count'].mean()
        avg_base_better = summary_df['ToM Better Count'].mean()

        average_row = pd.DataFrame({
            'Subject': ['Average'],
            'CoT Better Count': [avg_cot_better],
            'ToM Better Count': [avg_base_better],
        })

        # Format the display DataFrame (including averages)
        display_summary_df = summary_df.copy()

        # Format the average row separately
        average_row['CoT Better Count'] = average_row['CoT Better Count'].map('{:.1f}'.format)
        average_row['ToM Better Count'] = average_row['ToM Better Count'].map('{:.1f}'.format)

        display_summary_df = pd.concat([display_summary_df, average_row], ignore_index=True)

    else: # Handle case where summary_df might be empty but average row logic needs a df
        display_summary_df = summary_df.copy()


    # --- Create Table Plot for Summary ---
    # Adjust figsize based on number of columns and rows
    fig_summary, ax_summary = plt.subplots(figsize=(8, len(display_summary_df) * 0.4 + 1)) # Adjusted width
    ax_summary.axis('tight')
    ax_summary.axis('off')

    # Create the summary table using the formatted display_summary_df
    table_summary = ax_summary.table(cellText=display_summary_df.values,
                                     colLabels=display_summary_df.columns,
                                     cellLoc='center',
                                     loc='center')

    table_summary.auto_set_font_size(False)
    table_summary.set_fontsize(9) # Adjust font size if needed
    table_summary.scale(1.1, 1.1) # Adjust scale if needed

    # Style the average row if it exists
    if not display_summary_df.empty and display_summary_df.iloc[-1]['Subject'] == 'Average':
        num_rows, num_cols = display_summary_df.shape
        for j in range(num_cols):
            cell = table_summary[num_rows, j] # Get cell for the last row (average row)
            cell.set_text_props(weight='bold') # Make text bold
            cell.set_facecolor("#DDDDDD") # Add a light grey background


    plt.title("Summary: Better Model Counts per Subject", y=1.05) # Adjust title position and text
    plt.tight_layout()
    graph5_path = os.path.join(output_dir, "graph_base_vs_cot_summary.png")
    plt.savefig(graph5_path, bbox_inches='tight', dpi=150) # Save the summary table
    plt.close(fig_summary) # Close the specific figure
    print(f"Base vs CoT summary table saved to {graph5_path}")

    # --- Print Summary Table to Console (Optional) ---
    # print("\n--- Summary: Better Model Counts per Subject ---") # Adjusted print title
    # print(display_summary_df.to_string(index=False))

else:
    print("\nNo differing Base/CoT pairs found to generate a summary count.")


# ---------------------------
# Graph 6: Per-subject performance table with average

# Use the pivot_df_graph3 calculated earlier for the heatmap
pivot_table_df = pivot_df_graph3.copy()

# Format all numeric columns (model scores and Average) as percentages
numeric_cols = pivot_table_df.columns.difference([' '])
for col in numeric_cols:
    pivot_table_df[col] = pivot_table_df[col].map('{:.1f}%'.format)
pivot_table_df[' '] = pivot_table_df[' '].fillna('')

# Transpose so models (and 'Average') become rows
table_df = pivot_table_df.T

# --- Create Table Plot with models on the left ---
fig_table_perf, ax_table_perf = plt.subplots(
    figsize=(max(12, table_df.shape[1] * 1.2),
             table_df.shape[0] * 0.4 + 1)
)
ax_table_perf.axis('tight')
ax_table_perf.axis('off')

table_perf = ax_table_perf.table(
    cellText=table_df.values,
    rowLabels=table_df.index,
    colLabels=table_df.columns,
    cellLoc='center',
    loc='center'
)
table_perf.auto_set_font_size(False)
table_perf.set_fontsize(9)
table_perf.scale(1.1, 1.1)

# Bold the 'Average' row
if 'Average' in table_df.index:
    avg_row = list(table_df.index).index('Average') + 1  # +1 to skip header
    for j in range(len(table_df.columns)):
        cell = table_perf[avg_row, j]
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#DDDDDD")

plt.title("Per-Model Performance Comparison Across Subjects (Table with Average)", y=1.05)
plt.tight_layout()
graph6_path = os.path.join(output_dir, "graph_subjects_table_with_avg.png")
plt.savefig(graph6_path, bbox_inches='tight', dpi=150)
plt.close(fig_table_perf)

print(f"Per-subject performance table saved to {graph6_path}")

# ... existing code before Graph 7 ...

# -------------------------------
# Graph 7: Average Per-Subject Normalized Performance per Model

# Use subject_df which contains per-subject scores
subj_norm_df = subject_df.copy()

# Define normalization function to apply per subject
def normalize_within_group(group):
    min_val = group['percentage_correct'].min()
    max_val = group['percentage_correct'].max()
    if max_val - min_val > 0:
        group['subj_normalized_percentage'] = (group['percentage_correct'] - min_val) / (max_val - min_val) * 100
    else:
        # If all models score the same in a subject, assign 0 or 100 based on preference
        # Assigning 0 might be misleading if the score was high. Let's assign 50 as neutral? Or maybe the actual score?
        # Let's assign 0 for now, assuming we care about relative difference.
        group['subj_normalized_percentage'] = 0.0
        # Alternative: assign 100 if max_val > 0, else 0.
        # group['subj_normalized_percentage'] = 100.0 if max_val > 0 else 0.0
    return group

# Apply normalization within each subject group
subj_norm_df = subj_norm_df.groupby('subject').apply(normalize_within_group)

# Calculate the average of these normalized scores for each model
avg_subj_norm_df = subj_norm_df.groupby('model')['subj_normalized_percentage'].mean().reset_index()

# Sort the results using the same model order as Graph 1
avg_subj_norm_df = avg_subj_norm_df.set_index("model").loc[ordered_models].reset_index()


# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.bar(avg_subj_norm_df["model"], avg_subj_norm_df["subj_normalized_percentage"], color='mediumseagreen') # Different color
plt.xlabel("Model")
plt.ylabel("Average Per-Subject Normalized Score (%)")
plt.title("Average Per-Subject Normalized Performance Comparison (Subjects Weighted Equally)")

# Adjust ylim
max_avg_norm_val = avg_subj_norm_df["subj_normalized_percentage"].max()
# Since this is an average of values between 0-100, the max is likely <= 100
plt.ylim(0, max(max_avg_norm_val * 1.1, 100)) # Give some headroom, ensure it goes to at least 100

plt.xticks(rotation=30, ha="right")
for idx, row in avg_subj_norm_df.iterrows():
    plt.text(idx, row["subj_normalized_percentage"] + 1, f'{row["subj_normalized_percentage"]:.1f}%', ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
graph7_path = os.path.join(output_dir, "graph_avg_subj_normalized.png")
plt.savefig(graph7_path) # New filename
plt.close()
print(f"Average per-subject normalized performance saved to {graph7_path}")




# -------------------------------
# Graph 8: Violin Plot of Per-Subject Normalized Scores (Base vs CoT)

# Add a column to identify model type (Base or CoT)
def get_model_type(m):
    m_lower = m.lower()
    if m_lower.startswith("cot_") or "-cot" in m_lower:
        return "CoT"
    else:
        # Assuming models not explicitly marked as CoT are Base models for this comparison
        return "Base"

# Make sure subj_norm_df is the one with per-subject normalized scores
# This should already be calculated before Graph 7 plotting
subj_norm_df['model_type'] = subj_norm_df['model'].apply(get_model_type)
print("Model type counts in subj_norm_df for Graph 8:\n", subj_norm_df['model_type'].value_counts())

plt.figure(figsize=(8, 6)) # Adjusted figsize for two categories
# Use the subj_norm_df which contains the normalized score for each model on each subject
# Plot based on the new 'model_type' column
# inner='quartile' shows the median (white dot) and quartiles (black bar)
# Removed showmeans=True and meanprops due to AttributeError with PolyCollection
sns.violinplot(data=subj_norm_df, x='model_type', y='subj_normalized_percentage',
               palette='muted', inner='quartile', order=['Base', 'CoT']) # Specify order
plt.xlabel("Model Type")
plt.ylabel("Per-Subject Normalized Score (%)")
# Updated title as mean marker is not shown due to error
plt.title("Distribution of Per-Subject Normalized Scores: Base vs. CoT Models (Median and Quartiles)")
plt.ylim(-5, 105) # Normalized scores are between 0 and 100
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
# No rotation needed for 2 categories
# plt.xticks(rotation=30, ha="right")
plt.tight_layout()
# plt.subplots_adjust(bottom=0.25) # Likely not needed with tight_layout and fewer labels
graph8_path = os.path.join(output_dir, "graph_base_vs_cot_normalized_violin.png")
plt.savefig(graph8_path) # New filename
plt.close()
print(f"Violin plot comparing Base vs CoT normalized scores saved to {graph8_path}")


# ... existing print statements ...
# Update the final print list
print(f"\nGraphs saved to the {output_dir} folder:")
print(f" - {graph1_path}")
print(f" - {graph2_path}")
print(f" - {graph3_path}")
if graph4_path: # Only print if the file was created
    print(f" - {graph4_path}")
if graph5_path: # Only print if the file was created
    print(f" - {graph5_path}")
print(f" - {graph6_path}")
print(f" - {graph7_path}")
print(f" - {graph8_path}")