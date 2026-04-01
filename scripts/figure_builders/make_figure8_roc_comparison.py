from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# =============================================================================
# FIGURE 8: ROC CURVES FOR THE THREE SELECTED FINAL MODELS
# =============================================================================
# Purpose of this script
# ----------------------
# This script creates a ROC-curve comparison figure for the three selected final
# models used in the project:
#
# Method 1: TF-IDF + Logistic Regression
# Method 2: Doc2Vec + Logistic Regression
# Method 3: DistilBERT
#
# The goal is to produce one clean visual showing how the three selected method
# families compare on the untouched 2025 test set in terms of ROC performance.
#
# Why this figure is useful
# -------------------------
# In the report, this figure supports the RQ2 comparison by showing the ranking
# ability of the final selected model from each method family. It is similar in
# style to the ROC figure used in the previous assessment report, but here it is
# adapted for the current binary text-classification study.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define the project folder and the expected prediction files
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

method_files = {
    "Method 1: TF-IDF": project_folder / "step12_final_test_predictions.csv",
    "Method 2: Doc2Vec": project_folder / "step22_doc2vec_final_test_predictions.csv",
    "Method 3: DistilBERT": project_folder / "step25_distilbert_final_test_predictions.csv",
}

figure_output = project_folder / "figure8_roc_curve_comparison.png"
summary_output = project_folder / "figure8_roc_curve_summary.csv"


# -----------------------------------------------------------------------------
# Step 2: Helper function to find the true-label and score columns
# -----------------------------------------------------------------------------
# Different prediction files may use slightly different column names. This
# function searches for common alternatives and returns the most likely columns.
# -----------------------------------------------------------------------------
def identify_columns(df):
    columns_lower = {col.lower(): col for col in df.columns}

    # Possible true-label column names
    true_candidates = [
        "true_label", "y_true", "label", "actual", "actual_label",
        "label_unstable", "target"
    ]

    # Possible score/probability column names
    score_candidates = [
        "predicted_probability", "pred_proba", "probability", "score",
        "positive_class_probability", "unstable_probability",
        "y_score", "prediction_score", "predicted_score",
        "prob_1", "proba_1"
    ]

    true_col = None
    score_col = None

    for candidate in true_candidates:
        if candidate in columns_lower:
            true_col = columns_lower[candidate]
            break

    for candidate in score_candidates:
        if candidate in columns_lower:
            score_col = columns_lower[candidate]
            break

    # If score was not found, try a more flexible search
    if score_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if "prob" in col_lower or "score" in col_lower:
                score_col = col
                break

    if true_col is None:
        raise ValueError(
            "Could not identify the true-label column. "
            f"Available columns: {list(df.columns)}"
        )

    if score_col is None:
        raise ValueError(
            "Could not identify the score/probability column. "
            f"Available columns: {list(df.columns)}"
        )

    return true_col, score_col


# -----------------------------------------------------------------------------
# Step 3: Load the three prediction files and compute ROC curves
# -----------------------------------------------------------------------------
roc_rows = []

fig, ax = plt.subplots(figsize=(9, 6))

for method_name, file_path in method_files.items():
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    true_col, score_col = identify_columns(df)

    y_true = df[true_col].astype(int)
    y_score = df[score_col].astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    ax.plot(fpr, tpr, linewidth=2, label=f"{method_name} (AUC={auc_value:.4f})")

    roc_rows.append({
        "method": method_name,
        "true_label_column": true_col,
        "score_column": score_col,
        "roc_auc": auc_value
    })


# -----------------------------------------------------------------------------
# Step 4: Add the random-baseline diagonal
# -----------------------------------------------------------------------------
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random baseline")


# -----------------------------------------------------------------------------
# Step 5: Improve figure readability
# -----------------------------------------------------------------------------
ax.set_title("ROC curves on the 2025 test set for the selected best models")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()


# -----------------------------------------------------------------------------
# Step 6: Save outputs
# -----------------------------------------------------------------------------
roc_summary_df = pd.DataFrame(roc_rows)
roc_summary_df.to_csv(summary_output, index=False)

plt.savefig(figure_output, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# Step 7: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 90)
print("FIGURE 8 CREATED SUCCESSFULLY")
print("=" * 90)
print(f"Summary file created: {summary_output.name}")
print(f"Figure file created:  {figure_output.name}")
print("=" * 90)
print("This PNG file is ready to be inserted into the report.")
print("=" * 90)