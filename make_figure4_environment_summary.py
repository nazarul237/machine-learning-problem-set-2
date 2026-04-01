from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# FIGURE 4: ENVIRONMENTAL COMPARISON AND SENSITIVITY SUMMARY
# =============================================================================
# Purpose of this script
# ----------------------
# This script creates Figure 4 for the report. The figure summarises the main
# environmental and contextual comparison results from the project.
#
# The goal is to show, in one visual, how the strongest text-only model compares
# with:
# 1. text + raw context,
# 2. text + engineered environmental features,
# 3. environment-only modelling,
# 4. complete-case text-only modelling, and
# 5. complete-case text + engineered environment modelling.
#
# Why this figure is useful
# -------------------------
# In the report, RQ3 asks whether environmental and operational context improves
# classification beyond narrative text alone. This figure helps answer that
# question by bringing together the main comparison settings in one place.
#
# Metrics shown
# -------------
# The figure focuses on:
# - PR-AUC
# - F1
# - ROC-AUC
#
# These are the most useful summary metrics here because the project deals with
# an imbalanced classification problem and these metrics were emphasised
# throughout model comparison and final interpretation.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define project folder and output paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

summary_output = project_folder / "figure4_environment_summary.csv"
figure_output = project_folder / "figure4_environment_summary.png"


# -----------------------------------------------------------------------------
# Step 2: Create a compact summary table of the environmental comparison results
# -----------------------------------------------------------------------------
# The values below are the confirmed results already used in the report-writing
# stage. They are entered directly so that the report figure is based on the
# final interpreted results rather than scattered raw files.
# -----------------------------------------------------------------------------
comparison_df = pd.DataFrame([
    {
        "setting": "Text-only\nbest model",
        "pr_auc": 0.6527,
        "f1": 0.6067,
        "roc_auc": 0.8933
    },
    {
        "setting": "Text + raw\ncontext",
        "pr_auc": 0.6053,
        "f1": 0.5499,
        "roc_auc": 0.8592
    },
    {
        "setting": "Text + engineered\nenvironment",
        "pr_auc": 0.6529,
        "f1": 0.5951,
        "roc_auc": 0.8910
    },
    {
        "setting": "Environment-\nonly",
        "pr_auc": 0.2613,
        "f1": 0.3347,
        "roc_auc": 0.6843
    },
    {
        "setting": "Complete-case\ntext-only",
        "pr_auc": 0.4439,
        "f1": 0.2941,
        "roc_auc": 0.8757
    },
    {
        "setting": "Complete-case\ntext + engineered env",
        "pr_auc": 0.3274,
        "f1": 0.3429,
        "roc_auc": 0.8101
    }
])

# Save the summary file so the figure values are documented clearly
comparison_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Step 3: Build the grouped bar chart
# -----------------------------------------------------------------------------
# A grouped bar chart is used because it allows the reader to compare the same
# three metrics across all environmental settings in one compact figure.
# -----------------------------------------------------------------------------
x_positions = range(len(comparison_df))
bar_width = 0.22

fig, ax = plt.subplots(figsize=(12, 7))

ax.bar([x - bar_width for x in x_positions],
       comparison_df["pr_auc"],
       width=bar_width,
       label="PR-AUC")

ax.bar(x_positions,
       comparison_df["f1"],
       width=bar_width,
       label="F1")

ax.bar([x + bar_width for x in x_positions],
       comparison_df["roc_auc"],
       width=bar_width,
       label="ROC-AUC")


# -----------------------------------------------------------------------------
# Step 4: Improve readability
# -----------------------------------------------------------------------------
ax.set_title("Environmental comparison and sensitivity summary")
ax.set_ylabel("Metric value")
ax.set_xticks(list(x_positions))
ax.set_xticklabels(comparison_df["setting"], rotation=0, ha="center")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Add value labels above each bar so the figure is easier to read in the report
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

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
print("FIGURE 4 CREATED SUCCESSFULLY")
print("=" * 90)
print(f"Summary file created: {summary_output.name}")
print(f"Figure file created:  {figure_output.name}")
print("=" * 90)
print("This PNG file is ready to be inserted into the report as Figure 4.")
print("=" * 90)