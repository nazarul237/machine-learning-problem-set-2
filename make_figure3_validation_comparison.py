from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# FIGURE 3: VALIDATION-BASED COMPARISON OF THE BEST MODELS
# =============================================================================
# Purpose of this script
# ----------------------
# This script creates Figure 3 for the report. The figure compares the best
# validation performance achieved by the three main text-analysis method
# families used in the project:
#
# Method 1: TF-IDF-based text classification
# Method 2: Doc2Vec-based document embedding classification
# Method 3: DistilBERT transfer learning
#
# The figure is intended to support the modelling-approach section by showing,
# in one place, how the strongest validation model from each method family
# compares on the key imbalance-aware metrics used throughout the study:
# - PR-AUC
# - F1
# - ROC-AUC
#
# Why these metrics are used
# --------------------------
# The project involves an imbalanced binary classification task, so evaluation
# should not rely on accuracy alone. PR-AUC, F1, and ROC-AUC were therefore
# treated as the most informative summary metrics for comparing competing
# models before final locked test evaluation.
#
# Output files
# ------------
# This script creates:
# 1. a compact summary CSV of the best validation metrics by method family
# 2. a PNG figure that can be inserted into the report as Figure 3
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define project folder and file paths
# -----------------------------------------------------------------------------
# The script uses paths relative to its own location so that it can be run
# reliably from within the project folder in VS Code.
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

# Method 1 (TF-IDF): these files are expected from the earlier tuning stage
method1_best_file = project_folder / "step9_best_text_only_settings.csv"
method1_all_file = project_folder / "step9_text_only_tuning_results.csv"

# Method 2 (Doc2Vec): these files were created during the Doc2Vec tuning stage
method2_best_file = project_folder / "step21_doc2vec_best_settings.csv"
method2_all_file = project_folder / "step21_doc2vec_tuning_results.csv"

# Method 3 (DistilBERT): this file stores the final validation metrics
method3_file = project_folder / "step24_distilbert_validation_metrics.csv"

# Output files
summary_output = project_folder / "figure3_validation_comparison_summary.csv"
figure_output = project_folder / "figure3_validation_comparison.png"


# -----------------------------------------------------------------------------
# Step 2: Define a helper function to load the best validation row
# -----------------------------------------------------------------------------
# For Methods 1 and 2, the workflow may contain:
# - a "best settings" file, or
# - a full tuning-results file
#
# This function first tries to use the best-settings file. If that file is not
# available, it falls back to the full tuning-results file and selects the best
# row using the same ranking logic applied in the project:
# PR-AUC first, then F1, then ROC-AUC.
# -----------------------------------------------------------------------------
def load_best_validation_row(best_file, all_file, method_label):
    """
    Load the best validation row for one method family.

    Parameters
    ----------
    best_file : pathlib.Path
        Path to the file storing the best settings or best row.
    all_file : pathlib.Path
        Path to the full tuning-results file.
    method_label : str
        Name of the method family for logging.

    Returns
    -------
    pandas.Series
        The best validation row.
    """
    if best_file.exists():
        df = pd.read_csv(best_file, low_memory=False)
        print(f"{method_label}: using best-settings file -> {best_file.name}")

        # If the file contains multiple rows (for example, one per model family),
        # sort them and take the strongest overall row by the same project logic.
        df = df.sort_values(by=["pr_auc", "f1", "roc_auc"], ascending=False)
        return df.iloc[0]

    elif all_file.exists():
        df = pd.read_csv(all_file, low_memory=False)
        print(f"{method_label}: using full tuning-results file -> {all_file.name}")

        df = df.sort_values(by=["pr_auc", "f1", "roc_auc"], ascending=False)
        return df.iloc[0]

    else:
        raise FileNotFoundError(
            f"No suitable validation file was found for {method_label}.\n"
            f"Expected one of:\n- {best_file.name}\n- {all_file.name}"
        )


# -----------------------------------------------------------------------------
# Step 3: Load the strongest validation result for each method family
# -----------------------------------------------------------------------------
# Method 1: TF-IDF
# Method 2: Doc2Vec
# Method 3: DistilBERT
# -----------------------------------------------------------------------------
method1_row = load_best_validation_row(
    best_file=method1_best_file,
    all_file=method1_all_file,
    method_label="Method 1 (TF-IDF)"
)

method2_row = load_best_validation_row(
    best_file=method2_best_file,
    all_file=method2_all_file,
    method_label="Method 2 (Doc2Vec)"
)

if not method3_file.exists():
    raise FileNotFoundError(
        f"DistilBERT validation file not found: {method3_file.name}"
    )

method3_df = pd.read_csv(method3_file, low_memory=False)
print(f"Method 3 (DistilBERT): using validation metrics file -> {method3_file.name}")
method3_row = method3_df.iloc[0]


# -----------------------------------------------------------------------------
# Step 4: Build a compact summary table for plotting
# -----------------------------------------------------------------------------
# This summary keeps only the method label and the three metrics that will be
# visualised in the report figure.
# -----------------------------------------------------------------------------
comparison_df = pd.DataFrame([
    {
        "method": "Method 1\nTF-IDF",
        "pr_auc": float(method1_row["pr_auc"]),
        "f1": float(method1_row["f1"]),
        "roc_auc": float(method1_row["roc_auc"])
    },
    {
        "method": "Method 2\nDoc2Vec",
        "pr_auc": float(method2_row["pr_auc"]),
        "f1": float(method2_row["f1"]),
        "roc_auc": float(method2_row["roc_auc"])
    },
    {
        "method": "Method 3\nDistilBERT",
        "pr_auc": float(method3_row["pr_auc"]),
        "f1": float(method3_row["f1"]),
        "roc_auc": float(method3_row["roc_auc"])
    }
])

# Save the summary table so the exact plotted values are documented clearly
comparison_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Step 5: Create the grouped bar chart
# -----------------------------------------------------------------------------
# A grouped bar chart is used because it makes it easy to compare multiple
# metrics across the three method families in one compact visual.
# -----------------------------------------------------------------------------
methods = comparison_df["method"]
x_positions = range(len(methods))
bar_width = 0.22

fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions are offset slightly so that the three metrics appear side by side
ax.bar([x - bar_width for x in x_positions], comparison_df["pr_auc"], width=bar_width, label="PR-AUC")
ax.bar(x_positions, comparison_df["f1"], width=bar_width, label="F1")
ax.bar([x + bar_width for x in x_positions], comparison_df["roc_auc"], width=bar_width, label="ROC-AUC")

# -----------------------------------------------------------------------------
# Step 6: Improve figure readability
# -----------------------------------------------------------------------------
ax.set_title("Validation-based comparison of the best models from the three text-analysis methods")
ax.set_ylabel("Metric value")
ax.set_xticks(list(x_positions))
ax.set_xticklabels(methods)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Add value labels above each bar so the figure remains easy to read in print
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

plt.tight_layout()


# -----------------------------------------------------------------------------
# Step 7: Save the figure
# -----------------------------------------------------------------------------
plt.savefig(figure_output, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# Step 8: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 90)
print("FIGURE 3 CREATED SUCCESSFULLY")
print("=" * 90)
print("Summary file created:")
print(f"- {summary_output.name}")
print()
print("Figure file created:")
print(f"- {figure_output.name}")
print()
print("This PNG file is ready to be inserted into the report as Figure 3.")
print("=" * 90)