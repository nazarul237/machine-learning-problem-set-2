from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# =============================================================================
# STEP 14: THRESHOLD ANALYSIS FOR THE FINAL SELECTED MODEL
# =============================================================================
# Purpose of this step:
# The previous steps identified the best overall model as the tuned text-only
# Logistic Regression. However, in classification problems like this one, model
# quality is not determined only by the fitted algorithm. It is also influenced
# by the decision threshold used to convert predicted probabilities into final
# class labels.
#
# By default, a Logistic Regression model usually uses a threshold of 0.50:
# - score >= 0.50 -> predict unstable approach
# - score <  0.50 -> predict non-unstable case
#
# That default threshold is convenient, but it is not always the best choice for
# an imbalanced operational problem. In safety-style classification, different
# thresholds lead to different trade-offs:
#
# - Lower threshold:
#   catches more true unstable cases (higher recall),
#   but usually creates more false alarms (lower precision).
#
# - Higher threshold:
#   creates fewer false alarms (higher precision),
#   but misses more true unstable cases (lower recall).
#
# This step is designed to make that trade-off visible and interpretable.
#
# Important methodological note:
# We use the VALIDATION predictions here, not the test set, because threshold
# discussion should be based on development data rather than tuned on the final
# test set after seeing its outcome.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

input_file = project_folder / "results" / "tables" / "step9_best_validation_predictions_logistic_regression.csv"

results_file = project_folder / "results" / "tables" / "step14_threshold_analysis_results.csv"
plot_file = project_folder / "results" / "figures" / "step14_threshold_precision_recall_f1.png"
confusion_prefix = "step14_confusion_matrix_threshold_"


# -----------------------------------------------------------------------------
# Load the saved validation predictions from the best tuned Logistic Regression
# -----------------------------------------------------------------------------
# This file should already contain:
# - true_label
# - predicted score
#
# The score represents the model's estimated probability (or equivalent ranking
# score) that the report belongs to the unstable-approach class.
# -----------------------------------------------------------------------------
df = pd.read_csv(input_file)

print("=" * 100)
print("STEP 14: THRESHOLD ANALYSIS")
print("=" * 100)
print(f"Rows loaded from validation prediction file: {len(df)}")
print()

# Safety check
required_cols = ["true_label", "score"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column missing from prediction file: {col}")

y_true = df["true_label"].astype(int)
y_score = df["score"].astype(float)


# -----------------------------------------------------------------------------
# Define the thresholds to test
# -----------------------------------------------------------------------------
# These threshold values are intentionally spread across a practical range.
# The aim is not to search every possible threshold, but to understand the broad
# operational trade-off between recall and precision.
# -----------------------------------------------------------------------------
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

results = []


# -----------------------------------------------------------------------------
# Evaluate model behaviour at each threshold
# -----------------------------------------------------------------------------
# For each threshold, we:
# 1. Convert scores into predicted class labels
# 2. Calculate precision, recall, and F1
# 3. Count how many reports are predicted as unstable
# 4. Save a confusion matrix for later interpretation
# -----------------------------------------------------------------------------
for threshold in thresholds:
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    predicted_unstable_cases = int(y_pred.sum())
    predicted_non_unstable_cases = int((y_pred == 0).sum())

    results.append({
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_unstable_cases": predicted_unstable_cases,
        "predicted_non_unstable_cases": predicted_non_unstable_cases
    })

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0_non_unstable", "actual_1_unstable"],
        columns=["predicted_0_non_unstable", "predicted_1_unstable"]
    )

    threshold_label = str(threshold).replace(".", "_")
    cm_file = project_folder / "results" / "tables" / f"{confusion_prefix}{threshold_label}.csv"
    cm_df.to_csv(cm_file)


# -----------------------------------------------------------------------------
# Save the threshold summary table
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(results_file, index=False)

print("THRESHOLD ANALYSIS SUMMARY")
print("-" * 100)
print(results_df.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Create a simple line plot for threshold trade-offs
# -----------------------------------------------------------------------------
# This plot helps visualise how precision, recall, and F1 change as the
# threshold becomes stricter.
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(results_df["threshold"], results_df["precision"], marker="o", label="Precision")
plt.plot(results_df["threshold"], results_df["recall"], marker="o", label="Recall")
plt.plot(results_df["threshold"], results_df["f1"], marker="o", label="F1")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("Threshold Trade-off for Best Tuned Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()


# -----------------------------------------------------------------------------
# Final output summary
# -----------------------------------------------------------------------------
print("FILES CREATED")
print("-" * 100)
print(results_file.name)
print(plot_file.name)

for threshold in thresholds:
    threshold_label = str(threshold).replace(".", "_")
    print(f"{confusion_prefix}{threshold_label}.csv")

print()
print("=" * 100)
print("STEP 14 FINISHED")
print("This step analysed how the validation performance of the final selected")
print("model changes under different decision thresholds.")
print("No re-tuning of the model has been done.")
print("=" * 100)