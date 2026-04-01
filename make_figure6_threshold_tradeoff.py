from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# FIGURE 6: THRESHOLD TRADE-OFF FOR THE FINAL SELECTED MODEL
# =============================================================================
# Purpose of this script
# ----------------------
# This script creates Figure 6 for the report. The figure shows how the main
# threshold-dependent performance metrics of the final selected model change as
# the classification threshold is adjusted.
#
# Why this figure matters
# -----------------------
# In the report, the threshold analysis is used to explain the operational
# trade-off between:
# - catching more unstable cases (higher recall),
# - avoiding false alarms (higher precision),
# - and maintaining overall balance (F1).
#
# This is especially important in an imbalanced classification setting, where the
# choice of threshold can materially change the practical usefulness of the model.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define project folder and possible input files
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

possible_input_files = [
    project_folder / "step14_threshold_results.csv",
    project_folder / "step14_threshold_summary.csv",
    project_folder / "step14_threshold_analysis.csv",
    project_folder / "step14_validation_threshold_results.csv"
]

summary_output = project_folder / "figure6_threshold_tradeoff_summary.csv"
figure_output = project_folder / "figure6_threshold_tradeoff.png"


# -----------------------------------------------------------------------------
# Step 2: Try to locate an existing threshold-results file
# -----------------------------------------------------------------------------
input_file = None
for file_path in possible_input_files:
    if file_path.exists():
        input_file = file_path
        break


# -----------------------------------------------------------------------------
# Step 3: Load threshold results
# -----------------------------------------------------------------------------
# If a saved file is found, use it directly.
# If not, fall back to the confirmed threshold values already used in the report.
# -----------------------------------------------------------------------------
if input_file is not None:
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Using existing threshold-results file: {input_file.name}")

    # Standardise likely column names if needed
    rename_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()

        if col_lower in ["threshold"]:
            rename_map[col] = "threshold"
        elif col_lower in ["precision", "val_precision", "validation_precision"]:
            rename_map[col] = "precision"
        elif col_lower in ["recall", "val_recall", "validation_recall"]:
            rename_map[col] = "recall"
        elif col_lower in ["f1", "val_f1", "validation_f1", "f1_score"]:
            rename_map[col] = "f1"

    df = df.rename(columns=rename_map)

    required_cols = {"threshold", "precision", "recall", "f1"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"The threshold file was found, but it does not contain the required columns: {missing_cols}"
        )

    threshold_df = df[["threshold", "precision", "recall", "f1"]].copy()
    threshold_df = threshold_df.sort_values("threshold").reset_index(drop=True)

else:
    print("No saved threshold-results CSV was found.")
    print("Using the confirmed threshold values already reported in the project.")

    # These are the values already discussed in the report.
    # Where intermediate precision/F1 values were not explicitly documented in
    # the report text, they are left as missing and will not be plotted.
    threshold_df = pd.DataFrame([
        {"threshold": 0.20, "precision": 0.3698, "recall": 0.8711, "f1": None},
        {"threshold": 0.30, "precision": None,   "recall": 0.7956, "f1": None},
        {"threshold": 0.40, "precision": None,   "recall": 0.6889, "f1": None},
        {"threshold": 0.50, "precision": 0.6136, "recall": 0.6000, "f1": 0.6067},
        {"threshold": 0.60, "precision": None,   "recall": None,   "f1": None},
        {"threshold": 0.70, "precision": 0.7627, "recall": None,   "f1": None},
        {"threshold": 0.80, "precision": 0.8375, "recall": 0.2978, "f1": None},
    ])


# Save the summary table for record-keeping
threshold_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Step 4: Create the threshold trade-off plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Plot only non-missing values for each metric
for metric in ["precision", "recall", "f1"]:
    metric_df = threshold_df.dropna(subset=[metric])
    if not metric_df.empty:
        ax.plot(
            metric_df["threshold"],
            metric_df[metric],
            marker="o",
            linewidth=2,
            label=metric.capitalize()
        )

# Add a vertical reference line for the default threshold of 0.50
ax.axvline(0.50, linestyle="--", linewidth=1, label="Default threshold = 0.50")

ax.set_title("Threshold trade-off for the final selected model")
ax.set_xlabel("Threshold")
ax.set_ylabel("Metric value")
ax.set_ylim(0, 1.0)
ax.set_xticks(sorted(threshold_df["threshold"].dropna().unique()))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()


# -----------------------------------------------------------------------------
# Step 5: Save the figure
# -----------------------------------------------------------------------------
plt.savefig(figure_output, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# Step 6: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 90)
print("FIGURE 6 CREATED SUCCESSFULLY")
print("=" * 90)
print(f"Summary file created: {summary_output.name}")
print(f"Figure file created:  {figure_output.name}")
print("=" * 90)
print("This PNG file is ready to be inserted into the report as Figure 6.")
print("=" * 90)