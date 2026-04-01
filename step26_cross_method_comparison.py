from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 26: CROSS-METHOD COMPARISON
# =============================================================================
# Purpose of this step:
# The project has now evaluated three text analysis methods on the same task:
#
# Method 1:
# TF-IDF based supervised text classification
# Best final model = tuned Logistic Regression
#
# Method 2:
# Doc2Vec document embedding classification
# Best final model = Logistic Regression on Doc2Vec vectors
#
# Method 3:
# DistilBERT transformer-based transfer learning classification
#
# The present step combines the final test results from all three methods into
# one clean comparison table so that the strongest overall method can be stated
# clearly and reported consistently.
#
# This step does not train or tune anything.
# It only summarises the already completed final test evaluations.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

method1_file = project_folder / "results" / "tables" / "step12_final_test_metrics.csv"
method2_file = project_folder / "results" / "tables" / "step22_doc2vec_final_test_metrics.csv"
method3_file = project_folder / "results" / "tables" / "step25_distilbert_final_test_metrics.csv"

comparison_output = project_folder / "results" / "tables" / "step26_cross_method_comparison.csv"
ranking_output = project_folder / "results" / "tables" / "step26_cross_method_ranking.csv"


# -----------------------------------------------------------------------------
# Load final metric files
# -----------------------------------------------------------------------------
method1_df = pd.read_csv(method1_file, low_memory=False)
method2_df = pd.read_csv(method2_file, low_memory=False)
method3_df = pd.read_csv(method3_file, low_memory=False)

print("=" * 100)
print("STEP 26: CROSS-METHOD COMPARISON")
print("=" * 100)
print("Final metric files loaded successfully.")
print()


# -----------------------------------------------------------------------------
# Standardise column names and add method labels
# -----------------------------------------------------------------------------
# Each file already contains the main final metrics, but the model names and
# exact metadata fields differ slightly. Here we standardise them so the final
# table is easier to read in the report.
# -----------------------------------------------------------------------------
method1_row = {
    "method": "Method 1",
    "text_analysis_method": "TF-IDF text classification",
    "final_model": "Logistic Regression",
    "precision": float(method1_df.loc[0, "precision"]),
    "recall": float(method1_df.loc[0, "recall"]),
    "f1": float(method1_df.loc[0, "f1"]),
    "roc_auc": float(method1_df.loc[0, "roc_auc"]),
    "pr_auc": float(method1_df.loc[0, "pr_auc"]),
    "test_rows": int(method1_df.loc[0, "test_rows"]),
    "test_unstable_cases": int(method1_df.loc[0, "test_unstable_cases"]),
    "short_interpretation": "Best overall balance across the final test metrics."
}

method2_row = {
    "method": "Method 2",
    "text_analysis_method": "Doc2Vec embedding classification",
    "final_model": "Logistic Regression",
    "precision": float(method2_df.loc[0, "precision"]),
    "recall": float(method2_df.loc[0, "recall"]),
    "f1": float(method2_df.loc[0, "f1"]),
    "roc_auc": float(method2_df.loc[0, "roc_auc"]),
    "pr_auc": float(method2_df.loc[0, "pr_auc"]),
    "test_rows": int(method2_df.loc[0, "test_rows"]),
    "test_unstable_cases": int(method2_df.loc[0, "test_unstable_cases"]),
    "short_interpretation": "Higher recall than DistilBERT, but weakest overall balance."
}

method3_row = {
    "method": "Method 3",
    "text_analysis_method": "DistilBERT transfer learning",
    "final_model": "DistilBERT classifier",
    "precision": float(method3_df.loc[0, "precision"]),
    "recall": float(method3_df.loc[0, "recall"]),
    "f1": float(method3_df.loc[0, "f1"]),
    "roc_auc": float(method3_df.loc[0, "roc_auc"]),
    "pr_auc": float(method3_df.loc[0, "pr_auc"]),
    "test_rows": int(method3_df.loc[0, "test_rows"]),
    "test_unstable_cases": int(method3_df.loc[0, "test_unstable_cases"]),
    "short_interpretation": "Stronger than Doc2Vec overall, but still below Method 1."
}

comparison_df = pd.DataFrame([method1_row, method2_row, method3_row])


# -----------------------------------------------------------------------------
# Save the main comparison table
# -----------------------------------------------------------------------------
comparison_df.to_csv(comparison_output, index=False)


# -----------------------------------------------------------------------------
# Create a ranked version of the table
# -----------------------------------------------------------------------------
# We rank the methods primarily by PR-AUC, then F1, then ROC-AUC.
# This matches the logic used elsewhere in the project for imbalanced
# classification.
# -----------------------------------------------------------------------------
ranking_df = comparison_df.sort_values(
    by=["pr_auc", "f1", "roc_auc"],
    ascending=False
).reset_index(drop=True)

ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))
ranking_df.to_csv(ranking_output, index=False)


# -----------------------------------------------------------------------------
# Print outputs
# -----------------------------------------------------------------------------
print("FINAL CROSS-METHOD COMPARISON")
print("-" * 100)
print(comparison_df.to_string(index=False))
print()

print("FINAL METHOD RANKING")
print("-" * 100)
print(ranking_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(comparison_output.name)
print(ranking_output.name)
print()

print("=" * 100)
print("STEP 26 FINISHED")
print("The final comparison across Method 1, Method 2, and Method 3 has been")
print("prepared for report writing.")
print("=" * 100)