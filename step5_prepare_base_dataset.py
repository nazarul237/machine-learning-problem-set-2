import pandas as pd

# =============================================================================
# STEP 5: PREPARE THE MODELLING-READY BASE DATASET
# =============================================================================
# Purpose of this step:
# In Step 4, the combined raw dataset was restricted to the correct study window
# (2018-2025) and to the relevant operational phase scope
# (Initial Approach / Final Approach / Landing).
#
# The purpose of the present step is to transform that scope-cleaned dataset into
# a modelling-ready BASE dataset. At this stage, we are still not training any
# model. Instead, we are preparing the minimum set of variables that will later
# support:
# - exploratory data analysis,
# - text feature construction,
# - context feature construction,
# - and supervised classification.
#
# This step performs five important tasks:
#
# 1. Create the binary target label:
#    label_unstable = 1 if the coded anomaly field contains "Unstabilized Approach"
#    label_unstable = 0 otherwise
#
# 2. Create one main text field:
#    text_main = Narrative if available, otherwise Synopsis
#
# 3. Inspect missingness in the main analytical columns
#
# 4. Remove rows that are unusable for modelling
#    (for example, rows with missing year or completely empty text)
#
# 5. Save a clean base dataset for the next stages of the project
#
# Important note:
# We are using the coded anomaly field only to CREATE the target label.
# We are NOT using the anomaly field as a predictor, because that would create
# leakage by giving the model direct access to the answer source.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 5A: read the scope-cleaned dataset from Step 4
# -----------------------------------------------------------------------------
input_file = "step4_scope_cleaned.csv"
df = pd.read_csv(input_file, low_memory=False)

print("=" * 90)
print("STEP 5: PREPARE THE MODELLING-READY BASE DATASET")
print("=" * 90)
print(f"Rows loaded from Step 4 scope-cleaned dataset: {len(df)}")
print()


# -----------------------------------------------------------------------------
# Step 5B: create the binary unstable-approach label
# -----------------------------------------------------------------------------
# The anomaly field is a coded field from the ASRS dataset. We use it as the
# source of the supervised target variable.
#
# Rule:
# - label_unstable = 1 if 'Unstabilized Approach' appears in the anomaly text
# - label_unstable = 0 otherwise
#
# This is a much stronger approach than trying to infer the label directly from
# the narrative text, because the narrative text will later be used as an input
# feature and should not also be used to define the target.
# -----------------------------------------------------------------------------
anomaly_text = df["Events | Anomaly"].fillna("").astype(str)

df["label_unstable"] = anomaly_text.str.contains(
    "Unstabilized Approach",
    case=False,
    na=False
).astype(int)

print("Binary unstable-approach label created.")
print()


# -----------------------------------------------------------------------------
# Step 5C: create one main text field
# -----------------------------------------------------------------------------
# The main narrative text for this project should come from 'Report 1 | Narrative'
# because it is usually the richest descriptive field.
#
# However, if the narrative is missing or blank, we fall back to
# 'Report 1 | Synopsis' so that potentially useful reports are not discarded
# unnecessarily at this stage.
# -----------------------------------------------------------------------------
narrative = df["Report 1 | Narrative"].fillna("").astype(str).str.strip()
synopsis = df["Report 1 | Synopsis"].fillna("").astype(str).str.strip()

df["text_main"] = narrative
df.loc[df["text_main"] == "", "text_main"] = synopsis

print("Main text field 'text_main' created from Narrative, with Synopsis as fallback.")
print()


# -----------------------------------------------------------------------------
# Step 5D: keep the key columns needed for the modelling base file
# -----------------------------------------------------------------------------
# At this stage, we keep:
# - the report identifier
# - the date and year
# - the main text
# - the phase field
# - the safe environmental / operational context variables
# - the anomaly field (for audit/reference only)
# - the final binary label
#
# Important:
# The anomaly field is retained for traceability at this stage, but it should
# not later be used as a predictor in the actual model.
# -----------------------------------------------------------------------------
keep_cols = [
    "ACN",
    "Time | Date",
    "incident_year",
    "Aircraft 1 | Flight Phase",
    "Events | Anomaly",
    "Report 1 | Narrative",
    "Report 1 | Synopsis",
    "text_main",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
    "Aircraft 1 | ATC / Advisory",
    "Aircraft 1 | Aircraft Operator",
    "Aircraft 1 | Make Model Name",
    "Aircraft 1 | Flight Plan",
    "Aircraft 1 | Mission",
    "Aircraft 1 | Operating Under FAR Part",
    "source_file",
    "label_unstable",
]

# Keep only those columns that are actually present in the file.
available_keep_cols = [col for col in keep_cols if col in df.columns]
base_df = df[available_keep_cols].copy()

print("Key columns for the modelling-ready base dataset have been selected.")
print()


# -----------------------------------------------------------------------------
# Step 5E: inspect missing values in the most important fields
# -----------------------------------------------------------------------------
# Before removing anything, we first inspect how much missingness exists in the
# variables that matter most for later analysis.
# -----------------------------------------------------------------------------
missing_check_cols = [
    "Time | Date",
    "incident_year",
    "Aircraft 1 | Flight Phase",
    "Report 1 | Narrative",
    "Report 1 | Synopsis",
    "text_main",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
]

available_missing_cols = [col for col in missing_check_cols if col in base_df.columns]

missing_summary = pd.DataFrame({
    "column": available_missing_cols,
    "missing_count": [base_df[col].isna().sum() for col in available_missing_cols],
    "missing_percent": [round(base_df[col].isna().mean() * 100, 2) for col in available_missing_cols],
})

print("=" * 90)
print("MISSING VALUE SUMMARY")
print("=" * 90)
print(missing_summary.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Step 5F: remove rows that are unusable for modelling
# -----------------------------------------------------------------------------
# At the present stage, a row is considered unusable if:
# - incident_year is missing, or
# - text_main is empty after combining Narrative and Synopsis
#
# We do NOT drop rows just because some context columns are missing, because
# environmental variables such as ceiling or weather may naturally be absent in
# some reports and can be handled later in the modelling workflow.
# -----------------------------------------------------------------------------
base_df["text_main"] = base_df["text_main"].fillna("").astype(str).str.strip()

before_drop = len(base_df)

usable_mask = (
    base_df["incident_year"].notna()
    & (base_df["text_main"] != "")
)

excluded_unusable_df = base_df[~usable_mask].copy()
base_df = base_df[usable_mask].copy()

after_drop = len(base_df)

print("=" * 90)
print("USABILITY CLEANING SUMMARY")
print("=" * 90)
print(f"Rows before removing unusable cases: {before_drop}")
print(f"Rows after removing unusable cases:  {after_drop}")
print(f"Rows excluded at this step:          {before_drop - after_drop}")
print()


# -----------------------------------------------------------------------------
# Step 5G: add exclusion reasons for the unusable rows file
# -----------------------------------------------------------------------------
excluded_unusable_df["exclusion_reason_step5"] = ""

excluded_unusable_df.loc[
    excluded_unusable_df["incident_year"].isna() & (excluded_unusable_df["text_main"] == ""),
    "exclusion_reason_step5"
] = "Missing year and empty text"

excluded_unusable_df.loc[
    excluded_unusable_df["incident_year"].isna() & (excluded_unusable_df["text_main"] != ""),
    "exclusion_reason_step5"
] = "Missing year"

excluded_unusable_df.loc[
    excluded_unusable_df["incident_year"].notna() & (excluded_unusable_df["text_main"] == ""),
    "exclusion_reason_step5"
] = "Empty text after Narrative/Synopsis fallback"


# -----------------------------------------------------------------------------
# Step 5H: show label balance after Step 5
# -----------------------------------------------------------------------------
# This is not yet the full class-imbalance analysis, but it gives an early view
# of the positive and negative class counts in the base dataset.
# -----------------------------------------------------------------------------
print("=" * 90)
print("LABEL DISTRIBUTION AFTER STEP 5")
print("=" * 90)
print(base_df["label_unstable"].value_counts(dropna=False))
print()

positive_count = int(base_df["label_unstable"].sum())
negative_count = int((base_df["label_unstable"] == 0).sum())
positive_percent = round((positive_count / len(base_df)) * 100, 2) if len(base_df) > 0 else 0

print(f"Positive unstable-approach cases: {positive_count}")
print(f"Negative non-unstable cases:      {negative_count}")
print(f"Positive class percentage:        {positive_percent}%")
print()


# -----------------------------------------------------------------------------
# Step 5I: show a short preview of the modelling-ready base dataset
# -----------------------------------------------------------------------------
preview_cols = [
    "ACN",
    "Time | Date",
    "incident_year",
    "Aircraft 1 | Flight Phase",
    "label_unstable",
    "text_main",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
]

available_preview_cols = [col for col in preview_cols if col in base_df.columns]

print("=" * 90)
print("PREVIEW OF THE MODELLING-READY BASE DATASET (FIRST 3 ROWS)")
print("=" * 90)
print(base_df[available_preview_cols].head(3).to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Step 5J: save the outputs of this step
# -----------------------------------------------------------------------------
# 1. step5_base_dataset.csv
#    The main modelling-ready base dataset
#
# 2. step5_excluded_unusable_rows.csv
#    Rows removed because they were unusable for modelling
#
# 3. step5_missing_summary.csv
#    Missing-value summary for the key analytical columns
# -----------------------------------------------------------------------------
base_df.to_csv("step5_base_dataset.csv", index=False)
excluded_unusable_df.to_csv("step5_excluded_unusable_rows.csv", index=False)
missing_summary.to_csv("step5_missing_summary.csv", index=False)

print("Files created:")
print("- step5_base_dataset.csv")
print("- step5_excluded_unusable_rows.csv")
print("- step5_missing_summary.csv")
print()


# -----------------------------------------------------------------------------
# Step 5K: save a compact summary table
# -----------------------------------------------------------------------------
summary_table = pd.DataFrame({
    "metric": [
        "rows_loaded_from_step4",
        "rows_after_step5",
        "rows_excluded_step5",
        "positive_unstable_cases",
        "negative_nonunstable_cases",
        "positive_class_percent"
    ],
    "value": [
        len(df),
        len(base_df),
        len(excluded_unusable_df),
        positive_count,
        negative_count,
        positive_percent
    ]
})

summary_table.to_csv("step5_base_summary.csv", index=False)

print("- step5_base_summary.csv")
print()

print("=" * 90)
print("STEP 5 FINISHED")
print("At this stage, the modelling-ready base dataset has been prepared.")
print("We have still NOT started EDA or modelling.")
print("=" * 90)