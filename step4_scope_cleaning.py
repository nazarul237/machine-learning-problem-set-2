import pandas as pd

# =============================================================================
# STEP 4: FORMAL SCOPE CLEANING OF THE COMBINED RAW DATASET
# =============================================================================
# Purpose of this step:
# In Step 3, the eight main raw ASRS files were combined into a single dataset.
# That step showed that the key variables required for the project are present,
# but it also showed that the combined raw data still contains some records that
# fall outside the intended analytical scope.
#
# The present project is specifically focused on unstable-approach events in the
# approach-and-landing phase. Therefore, before creating labels or preparing the
# modelling dataset, we first need to apply a clear and defensible scope rule.
#
# In this step, we do NOT create the final target variable yet. We only perform
# the first formal cleaning pass so that the retained dataset genuinely belongs
# to the operational domain of interest.
#
# The two scope rules used here are:
#
# 1. STUDY PERIOD RULE
#    Keep only records from 2018 to 2025 inclusive.
#    This is consistent with the chosen study window for the assessment.
#
# 2. FLIGHT PHASE RULE
#    Keep only records whose Aircraft 1 flight-phase field contains at least one
#    of the following target phases:
#       - Initial Approach
#       - Final Approach
#       - Landing
#
# Important note:
# We intentionally keep mixed phase entries such as:
#    - Final Approach; Landing
#    - Descent; Initial Approach
#    - Landing; Taxi
# if they still contain one of the target phases.
#
# This is a deliberate methodological choice. At this stage, it is more sensible
# to retain records that clearly involve the approach-and-landing sequence than
# to exclude them too aggressively and risk discarding relevant cases.
#
# For transparency, this script also creates an EXCLUDED ROWS file so that the
# cleaning decisions remain auditable.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 4A: read the combined raw dataset from Step 3
# -----------------------------------------------------------------------------
# This file should already exist because it was created in the previous step.
# -----------------------------------------------------------------------------
input_file = project_folder / "data" / "processed" / "step3_main_phase_combined_raw.csv"
df = pd.read_csv(input_file, low_memory=False)

print("=" * 90)
print("STEP 4: FORMAL SCOPE CLEANING")
print("=" * 90)
print(f"Rows loaded from Step 3 combined dataset: {len(df)}")
print()


# -----------------------------------------------------------------------------
# Step 4B: create a year field from the ASRS date variable
# -----------------------------------------------------------------------------
# The ASRS date values are currently stored in forms such as:
#   201801
#   201901
#   202401
#
# For the current step, we only need the calendar year.
# Therefore, we take the first four characters and convert them to numeric form.
# -----------------------------------------------------------------------------
date_str = df["Time | Date"].astype(str).str.replace(".0", "", regex=False).str.strip()
df["incident_year"] = pd.to_numeric(date_str.str[:4], errors="coerce")

print("Year values created from 'Time | Date'.")
print()


# -----------------------------------------------------------------------------
# Step 4C: define the project study-period rule
# -----------------------------------------------------------------------------
# The intended study window for this assessment is 2018 to 2025 inclusive.
# Rows outside this range are considered out of scope for the final project.
# -----------------------------------------------------------------------------
df["keep_study_period"] = df["incident_year"].between(2018, 2025, inclusive="both")

print("Study-period rule applied: keep only years from 2018 to 2025.")
print()


# -----------------------------------------------------------------------------
# Step 4D: define the flight-phase scope rule
# -----------------------------------------------------------------------------
# We keep a row if its flight-phase text contains ANY of:
#   - Initial Approach
#   - Final Approach
#   - Landing
#
# This includes mixed strings such as:
#   - Final Approach; Landing
#   - Descent; Initial Approach
#   - Landing; Taxi
#
# It excludes rows such as:
#   - Taxi
#   - Cruise
#   - Climb
#   - Takeoff / Launch
# when they do not contain any of the target approach/landing phases.
# -----------------------------------------------------------------------------
target_phases = ["Initial Approach", "Final Approach", "Landing"]

phase_text = df["Aircraft 1 | Flight Phase"].fillna("").astype(str)

df["keep_phase_scope"] = phase_text.apply(
    lambda x: any(phase in x for phase in target_phases)
)

print("Flight-phase scope rule applied.")
print("Target phases retained:", ", ".join(target_phases))
print()


# -----------------------------------------------------------------------------
# Step 4E: combine both rules into one final keep/exclude flag
# -----------------------------------------------------------------------------
# A row is kept only if:
#   - it is within the study period, AND
#   - it belongs to the required phase scope
# -----------------------------------------------------------------------------
df["keep_step4"] = df["keep_study_period"] & df["keep_phase_scope"]


# -----------------------------------------------------------------------------
# Step 4F: split the dataset into kept rows and excluded rows
# -----------------------------------------------------------------------------
# We save excluded rows separately because this is good research practice:
# it creates an audit trail and makes the cleaning process transparent.
# -----------------------------------------------------------------------------
kept_df = df[df["keep_step4"]].copy()
excluded_df = df[~df["keep_step4"]].copy()

print("=" * 90)
print("CLEANING SUMMARY")
print("=" * 90)
print(f"Original rows before Step 4 cleaning: {len(df)}")
print(f"Rows kept after Step 4 cleaning:      {len(kept_df)}")
print(f"Rows excluded in Step 4:              {len(excluded_df)}")
print()


# -----------------------------------------------------------------------------
# Step 4G: explain why rows were excluded
# -----------------------------------------------------------------------------
# This is useful both for your own understanding and for later reporting.
# -----------------------------------------------------------------------------
excluded_df["exclusion_reason"] = ""

excluded_df.loc[
    (~excluded_df["keep_study_period"]) & (~excluded_df["keep_phase_scope"]),
    "exclusion_reason"
] = "Outside study period and outside flight-phase scope"

excluded_df.loc[
    (~excluded_df["keep_study_period"]) & (excluded_df["keep_phase_scope"]),
    "exclusion_reason"
] = "Outside study period"

excluded_df.loc[
    (excluded_df["keep_study_period"]) & (~excluded_df["keep_phase_scope"]),
    "exclusion_reason"
] = "Outside flight-phase scope"

print("Exclusion reasons assigned.")
print()


# -----------------------------------------------------------------------------
# Step 4H: show the year distribution after cleaning
# -----------------------------------------------------------------------------
print("=" * 90)
print("YEAR DISTRIBUTION AFTER STEP 4 CLEANING")
print("=" * 90)
print(kept_df["incident_year"].value_counts(dropna=False).sort_index())
print()


# -----------------------------------------------------------------------------
# Step 4I: show the most common flight-phase values after cleaning
# -----------------------------------------------------------------------------
print("=" * 90)
print("TOP FLIGHT-PHASE VALUES AFTER STEP 4 CLEANING")
print("=" * 90)
print(kept_df["Aircraft 1 | Flight Phase"].fillna("MISSING").value_counts().head(25))
print()


# -----------------------------------------------------------------------------
# Step 4J: show examples of excluded rows
# -----------------------------------------------------------------------------
# This helps us verify that the cleaning rules are behaving sensibly.
# -----------------------------------------------------------------------------
preview_excluded_cols = [
    "ACN",
    "Time | Date",
    "incident_year",
    "Aircraft 1 | Flight Phase",
    "Events | Anomaly",
    "exclusion_reason",
    "source_file",
]

available_excluded_cols = [col for col in preview_excluded_cols if col in excluded_df.columns]

print("=" * 90)
print("EXAMPLES OF EXCLUDED ROWS (FIRST 10)")
print("=" * 90)
if len(excluded_df) > 0:
    print(excluded_df[available_excluded_cols].head(10).to_string(index=False))
else:
    print("No rows were excluded.")
print()


# -----------------------------------------------------------------------------
# Step 4K: save the cleaned and excluded datasets
# -----------------------------------------------------------------------------
# 1. step4_scope_cleaned.csv
#    This is the cleaned dataset that remains within the selected study period
#    and operational phase scope.
#
# 2. step4_excluded_rows.csv
#    This is the audit file containing all removed rows and the reason they were
#    excluded.
# -----------------------------------------------------------------------------
kept_df.to_csv(project_folder / "data" / "processed" / "step4_scope_cleaned.csv", index=False)
excluded_df.to_csv(project_folder / "data" / "processed" / "step4_excluded_rows.csv", index=False)

print("Files created:")
print("- step4_scope_cleaned.csv")
print("- step4_excluded_rows.csv")
print()


# -----------------------------------------------------------------------------
# Step 4L: save a compact summary table
# -----------------------------------------------------------------------------
summary_table = pd.DataFrame({
    "metric": [
        "rows_before_cleaning",
        "rows_after_cleaning",
        "rows_excluded",
        "kept_unstable_preview_count",
        "kept_nonunstable_preview_count"
    ],
    "value": [
        len(df),
        len(kept_df),
        len(excluded_df),
        int(kept_df["Events | Anomaly"].fillna("").astype(str).str.contains("Unstabilized Approach", case=False, na=False).sum()),
        int((~kept_df["Events | Anomaly"].fillna("").astype(str).str.contains("Unstabilized Approach", case=False, na=False)).sum())
    ]
})

summary_table.to_csv(project_folder / "results" / "tables" / "step4_cleaning_summary.csv", index=False)

print("- step4_cleaning_summary.csv")
print()

print("=" * 90)
print("STEP 4 FINISHED")
print("At this stage, we have applied the first formal scope-cleaning rules.")
print("We have still NOT created the final target label and have NOT started modelling.")
print("=" * 90)