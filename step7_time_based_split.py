from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 7: CREATE A TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
# Purpose of this step:
# Up to this point, we have:
# - inspected the raw ASRS files
# - rebuilt the headers
# - restricted the dataset to the project scope
# - created the unstable-approach label
# - prepared the modelling-ready base dataset
# - completed exploratory data analysis
#
# The next methodological step is to split the data into:
# - a training set
# - a validation set
# - a test set
#
# We are deliberately using a time-based split rather than a random split.
# This is important because aviation reports occur over time, and a model that
# performs well on older data should be tested on genuinely newer data.
#
# In this project, the split will be:
# - Train:      2018 to 2023
# - Validation: 2024
# - Test:       2025
#
# This mirrors the strong temporal evaluation logic used in the previous
# assessment and makes the present study more realistic and defensible.
#
# Important note:
# This script does NOT train any model. It only creates the three datasets that
# will later be used for model development and final evaluation.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
# We use the script's own folder as the working reference point. This makes the
# script more reliable because it does not depend on where the terminal happens
# to be opened from.
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "data" / "processed" / "step5_base_dataset.csv"

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"
test_file = project_folder / "data" / "splits" / "step7_test_dataset.csv"
summary_file = project_folder / "results" / "tables" / "step7_split_summary.csv"


# -----------------------------------------------------------------------------
# Load the modelling-ready base dataset
# -----------------------------------------------------------------------------
# This file should already contain:
# - the binary unstable label
# - the main text field
# - the contextual variables selected in Step 5
# - the incident year variable
# -----------------------------------------------------------------------------
df = pd.read_csv(input_file, low_memory=False)

print("=" * 90)
print("STEP 7: TIME-BASED TRAIN / VALIDATION / TEST SPLIT")
print("=" * 90)
print(f"Rows loaded from Step 5 base dataset: {len(df)}")
print()


# -----------------------------------------------------------------------------
# Safety checks before splitting
# -----------------------------------------------------------------------------
# We verify that the incident_year column exists and does not contain missing
# values. This is important because the entire logic of this step depends on
# having a valid year for each report.
# -----------------------------------------------------------------------------
if "incident_year" not in df.columns:
    raise ValueError("The column 'incident_year' was not found in step5_base_dataset.csv.")

missing_years = df["incident_year"].isna().sum()
print(f"Missing values in incident_year: {missing_years}")

if missing_years > 0:
    print("Warning: there are missing years in the dataset.")
    print("These rows would make a time-based split unreliable.")
    print("Please inspect Step 5 again before proceeding.")
    raise SystemExit()

print()


# -----------------------------------------------------------------------------
# Create the time-based splits
# -----------------------------------------------------------------------------
# We now divide the dataset using the pre-decided year boundaries.
#
# Why these ranges?
# - 2018-2023 gives us a reasonably large training period
# - 2024 gives us a separate validation period for model selection / tuning
# - 2025 gives us a clean final hold-out test set
#
# This means the test set is not touched during model development.
# -----------------------------------------------------------------------------
train_df = df[df["incident_year"].between(2018, 2023)].copy()
validation_df = df[df["incident_year"] == 2024].copy()
test_df = df[df["incident_year"] == 2025].copy()


# -----------------------------------------------------------------------------
# Create a summary table for the three splits
# -----------------------------------------------------------------------------
# This summary is useful both for checking that the split worked correctly and
# later for reporting the data partition in the assessment write-up.
# -----------------------------------------------------------------------------
def make_split_summary(split_name, split_df):
    total_rows = len(split_df)
    unstable_cases = int(split_df["label_unstable"].sum())
    non_unstable_cases = total_rows - unstable_cases

    if total_rows > 0:
        unstable_percent = round(unstable_cases / total_rows * 100, 2)
    else:
        unstable_percent = 0.0

    return {
        "split": split_name,
        "rows": total_rows,
        "unstable_cases": unstable_cases,
        "non_unstable_cases": non_unstable_cases,
        "unstable_percent": unstable_percent,
    }

summary_rows = [
    make_split_summary("train_2018_2023", train_df),
    make_split_summary("validation_2024", validation_df),
    make_split_summary("test_2025", test_df),
]

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_file, index=False)


# -----------------------------------------------------------------------------
# Save the three split datasets
# -----------------------------------------------------------------------------
# These files will be used in the later modelling steps.
# We save them separately so that the workflow remains clear and reproducible.
# -----------------------------------------------------------------------------
train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)


# -----------------------------------------------------------------------------
# Print a detailed summary to the terminal
# -----------------------------------------------------------------------------
print("SPLIT SUMMARY")
print("-" * 90)
print(summary_df.to_string(index=False))
print()

print("YEAR COVERAGE CHECK")
print("-" * 90)
print("Train years present:     ", sorted(train_df["incident_year"].unique().tolist()))
print("Validation years present:", sorted(validation_df["incident_year"].unique().tolist()))
print("Test years present:      ", sorted(test_df["incident_year"].unique().tolist()))
print()

print("FILES CREATED")
print("-" * 90)
print(train_file.name)
print(validation_file.name)
print(test_file.name)
print(summary_file.name)
print()

print("=" * 90)
print("STEP 7 FINISHED")
print("The time-based split has been created successfully.")
print("No modelling has been started yet.")
print("=" * 90)