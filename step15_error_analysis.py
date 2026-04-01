from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 15: ERROR ANALYSIS ON THE FINAL TEST-SET PREDICTIONS
# =============================================================================
# Purpose of this step:
# The final model has already been selected and evaluated on the untouched 2025
# test set. That gave us the headline performance metrics.
#
# However, a strong machine learning assessment should not stop at overall
# metrics alone. It should also examine the pattern of model errors.
#
# In practical terms, this step helps answer questions such as:
# - Which cases became false positives?
# - Which cases became false negatives?
# - Are missed unstable cases shorter, vaguer, or less detailed?
# - Are false alarms associated with difficult environmental or operational
#   situations that sound risky in the narrative but are not coded as unstable?
#
# This is not a modelling step.
# We are not re-fitting, tuning, or modifying the model here.
# We are only analysing the output of the already-finalised test-set model.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

predictions_file = project_folder / "results" / "tables" / "step12_final_test_predictions.csv"
test_data_file = project_folder / "data" / "splits" / "step7_test_dataset.csv"

summary_file = project_folder / "results" / "tables" / "step15_error_summary.csv"
false_positives_file = project_folder / "results" / "tables" / "step15_top_false_positives.csv"
false_negatives_file = project_folder / "results" / "tables" / "step15_top_false_negatives.csv"
true_positives_file = project_folder / "results" / "tables" / "step15_top_true_positives.csv"
phase_summary_file = project_folder / "results" / "tables" / "step15_error_by_flight_phase.csv"
flight_conditions_summary_file = project_folder / "results" / "tables" / "step15_error_by_flight_conditions.csv"
length_summary_file = project_folder / "results" / "tables" / "step15_error_length_summary.csv"


# -----------------------------------------------------------------------------
# Load the final prediction file and the original test dataset
# -----------------------------------------------------------------------------
# The prediction file contains:
# - ACN
# - incident_year
# - true_label
# - predicted_label
# - score
#
# The test dataset contains the original fields such as:
# - text_main
# - flight phase
# - environmental fields
# -----------------------------------------------------------------------------
pred_df = pd.read_csv(predictions_file, low_memory=False)
test_df = pd.read_csv(test_data_file, low_memory=False)

print("=" * 100)
print("STEP 15: ERROR ANALYSIS")
print("=" * 100)
print(f"Prediction rows loaded: {len(pred_df)}")
print(f"Test rows loaded:       {len(test_df)}")
print()


# -----------------------------------------------------------------------------
# Merge predictions back to the original test rows
# -----------------------------------------------------------------------------
# We merge on ACN so that each prediction can be inspected alongside its
# original narrative and context.
# -----------------------------------------------------------------------------
analysis_df = test_df.merge(
    pred_df,
    on=["ACN", "incident_year"],
    how="inner"
)

print(f"Rows after merging predictions with test data: {len(analysis_df)}")
print()


# -----------------------------------------------------------------------------
# Create a simple error-type label
# -----------------------------------------------------------------------------
# TP = correctly predicted unstable case
# TN = correctly predicted non-unstable case
# FP = false alarm (predicted unstable, actually non-unstable)
# FN = missed unstable case (predicted non-unstable, actually unstable)
# -----------------------------------------------------------------------------
def classify_error_type(row):
    if row["true_label"] == 1 and row["predicted_label"] == 1:
        return "TP"
    elif row["true_label"] == 0 and row["predicted_label"] == 0:
        return "TN"
    elif row["true_label"] == 0 and row["predicted_label"] == 1:
        return "FP"
    elif row["true_label"] == 1 and row["predicted_label"] == 0:
        return "FN"
    else:
        return "Unknown"

analysis_df["error_type"] = analysis_df.apply(classify_error_type, axis=1)


# -----------------------------------------------------------------------------
# Create basic narrative-length fields
# -----------------------------------------------------------------------------
# These help us check whether some error types are associated with shorter or
# longer narratives.
# -----------------------------------------------------------------------------
analysis_df["text_main"] = analysis_df["text_main"].fillna("").astype(str)
analysis_df["text_char_length"] = analysis_df["text_main"].str.len()
analysis_df["text_word_count"] = analysis_df["text_main"].str.split().str.len()


# -----------------------------------------------------------------------------
# Overall error summary
# -----------------------------------------------------------------------------
error_counts = analysis_df["error_type"].value_counts().reset_index()
error_counts.columns = ["error_type", "count"]
error_counts["percent"] = (error_counts["count"] / len(analysis_df) * 100).round(2)
error_counts.to_csv(summary_file, index=False)

print("OVERALL ERROR SUMMARY")
print("-" * 100)
print(error_counts.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Narrative-length summary by error type
# -----------------------------------------------------------------------------
# This helps identify whether false negatives or false positives tend to have
# systematically different narrative length patterns.
# -----------------------------------------------------------------------------
length_summary = (
    analysis_df.groupby("error_type")[["text_char_length", "text_word_count"]]
    .agg(["mean", "median", "min", "max"])
)

length_summary.to_csv(length_summary_file)

print("NARRATIVE LENGTH SUMMARY BY ERROR TYPE")
print("-" * 100)
print(length_summary)
print()


# -----------------------------------------------------------------------------
# Error summary by flight phase
# -----------------------------------------------------------------------------
# This can reveal whether the model struggles more in certain flight-phase
# settings within the scoped approach-and-landing data.
# -----------------------------------------------------------------------------
phase_summary = pd.crosstab(
    analysis_df["Aircraft 1 | Flight Phase"].fillna("Missing"),
    analysis_df["error_type"]
)

phase_summary.to_csv(phase_summary_file)

print("ERROR TYPE BY FLIGHT PHASE")
print("-" * 100)
print(phase_summary)
print()


# -----------------------------------------------------------------------------
# Error summary by flight conditions
# -----------------------------------------------------------------------------
# This checks whether some flight-condition categories are associated with more
# misses or more false alarms.
# -----------------------------------------------------------------------------
flight_conditions_summary = pd.crosstab(
    analysis_df["Environment | Flight Conditions"].fillna("Missing"),
    analysis_df["error_type"]
)

flight_conditions_summary.to_csv(flight_conditions_summary_file)

print("ERROR TYPE BY FLIGHT CONDITIONS")
print("-" * 100)
print(flight_conditions_summary)
print()


# -----------------------------------------------------------------------------
# Extract top example cases
# -----------------------------------------------------------------------------
# We save a small set of the most informative cases for later manual reading.
#
# False positives:
# predicted unstable with high confidence, but actually non-unstable
#
# False negatives:
# actually unstable, but predicted non-unstable
# Here we sort by score descending so that we can see the borderline misses first
# (cases the model almost believed were unstable, but not quite enough).
#
# True positives:
# correctly predicted unstable with high confidence
# These are useful for understanding what the model is capturing well.
# -----------------------------------------------------------------------------
example_columns = [
    "ACN",
    "incident_year",
    "Aircraft 1 | Flight Phase",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
    "true_label",
    "predicted_label",
    "score",
    "error_type",
    "text_word_count",
    "text_main",
]

false_positives = analysis_df[analysis_df["error_type"] == "FP"].copy()
false_negatives = analysis_df[analysis_df["error_type"] == "FN"].copy()
true_positives = analysis_df[analysis_df["error_type"] == "TP"].copy()

# Highest-confidence false alarms
false_positives = false_positives.sort_values(by="score", ascending=False).head(25)

# Borderline misses: sort descending to inspect cases that were missed but still
# received relatively high unstable scores
false_negatives = false_negatives.sort_values(by="score", ascending=False).head(25)

# Highest-confidence correct unstable predictions
true_positives = true_positives.sort_values(by="score", ascending=False).head(25)

false_positives[example_columns].to_csv(false_positives_file, index=False)
false_negatives[example_columns].to_csv(false_negatives_file, index=False)
true_positives[example_columns].to_csv(true_positives_file, index=False)

print("TOP EXAMPLE FILES SAVED")
print("-" * 100)
print(false_positives_file.name)
print(false_negatives_file.name)
print(true_positives_file.name)
print()


# -----------------------------------------------------------------------------
# Final output summary
# -----------------------------------------------------------------------------
print("FILES CREATED")
print("-" * 100)
print(summary_file.name)
print(length_summary_file.name)
print(phase_summary_file.name)
print(flight_conditions_summary_file.name)
print(false_positives_file.name)
print(false_negatives_file.name)
print(true_positives_file.name)
print()

print("=" * 100)
print("STEP 15 FINISHED")
print("This step analysed the final model's test-set mistakes without changing")
print("the model itself.")
print("=" * 100)