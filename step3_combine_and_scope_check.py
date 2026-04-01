from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 3: COMBINE THE MAIN RAW FILES AND DO A FIRST SCOPE CHECK
# =============================================================================
# Purpose of this step:
# In Step 2, we proved that the ASRS raw files can be imported correctly when
# both header rows are used and flattened into readable column names.
#
# The purpose of the present step is not to clean the dataset yet. Instead, it
# is to bring the main flight-phase files together into one combined table and
# inspect the broad scope of the data before any filtering or transformation.
#
# In particular, this script helps answer the following questions:
# 1. How many rows do we have after combining all main raw files?
# 2. Are the key variables still present after combination?
# 3. What kinds of flight-phase values are present?
# 4. What kinds of anomaly values are present?
# 5. What years are covered by the downloaded reports?
#
# This is methodologically important because combining files blindly without
# checking the resulting structure would be poor practice. Before we move into
# cleaning, we want to understand the shape and scope of the combined dataset
# exactly as it exists in raw form.
#
# IMPORTANT:
# This script should be run only on the MAIN flight-phase files
# (Initial Approach / Final Approach / Landing search results).
# It should NOT include the separate unstable-only files.
# =============================================================================


# -----------------------------------------------------------------------------
# Step 3A: find all raw CSV files in the current folder
# -----------------------------------------------------------------------------
# We assume this script is saved in the same folder as the main raw ASRS CSV
# files. The pattern below matches files such as:
# - ASRS_DBOnline.csv
# - ASRS_DBOnline (1).csv
# - ASRS_DBOnline (2).csv
# etc.
# -----------------------------------------------------------------------------
folder = Path(".")
csv_files = sorted((project_folder / "data" / "raw").glob("ASRS_DBOnline*.csv"))

if len(csv_files) == 0:
    print("No ASRS raw CSV files were found in this folder.")
    print("Please make sure only the MAIN flight-phase raw files are saved here.")
    raise SystemExit()

print("=" * 90)
print("STEP 3: COMBINE MAIN RAW FILES AND CHECK PROJECT SCOPE")
print("=" * 90)
print(f"Number of raw files found: {len(csv_files)}")
print()


# -----------------------------------------------------------------------------
# Step 3B: function to flatten the two-row ASRS header into readable names
# -----------------------------------------------------------------------------
# Example:
# ('Time', 'Date') becomes 'Time | Date'
# ('Aircraft 1', 'Flight Phase') becomes 'Aircraft 1 | Flight Phase'
# (' ', 'ACN') becomes 'ACN'
#
# This is the same header logic from Step 2. We repeat it here so that the
# combined dataset uses the correct variable names from the start.
# -----------------------------------------------------------------------------
def flatten_asrs_columns(columns):
    flattened = []

    for top, bottom in columns:
        top = str(top).strip()
        bottom = str(bottom).strip()

        if top == "" or top.startswith("Unnamed:"):
            new_name = bottom
        elif bottom == "" or bottom.startswith("Unnamed:"):
            new_name = top
        else:
            new_name = f"{top} | {bottom}"

        flattened.append(new_name)

    return flattened


# -----------------------------------------------------------------------------
# Step 3C: read each raw file, rebuild headers, and store it in a list
# -----------------------------------------------------------------------------
# We also add a source_file column so that later we can trace each row back to
# the original CSV file if needed. This is useful for transparency and debugging.
# -----------------------------------------------------------------------------
all_dfs = []
file_summary = []

for file_path in csv_files:
    print("-" * 90)
    print(f"Reading file: {file_path.name}")

    df = pd.read_csv(file_path, header=[0, 1], low_memory=False)
    df.columns = flatten_asrs_columns(df.columns)

    # Add a source file column for traceability.
    df["source_file"] = file_path.name

    all_dfs.append(df)

    file_summary.append(
        {
            "file_name": file_path.name,
            "rows": len(df),
            "columns": len(df.columns),
        }
    )

    print(f"Rows in file: {len(df)}")
    print(f"Columns in file: {len(df.columns)}")
    print()


# -----------------------------------------------------------------------------
# Step 3D: combine all files into one raw dataset
# -----------------------------------------------------------------------------
# We use ignore_index=True so that pandas creates one clean continuous row index
# across the combined dataset.
# -----------------------------------------------------------------------------
combined_df = pd.concat(all_dfs, ignore_index=True)

print("=" * 90)
print("COMBINED DATASET SUMMARY")
print("=" * 90)
print(f"Combined rows: {len(combined_df)}")
print(f"Combined columns: {len(combined_df.columns)}")
print()


# -----------------------------------------------------------------------------
# Step 3E: save a file-level summary table
# -----------------------------------------------------------------------------
# This small table records how many rows came from each file. It is useful as
# documentation for your assessment and also helps show that the combination step
# was done transparently.
# -----------------------------------------------------------------------------
summary_df = pd.DataFrame(file_summary)
summary_df.to_csv(project_folder / "results" / "tables" / "step3_file_summary.csv", index=False)

print("A file-by-file summary has been saved as:")
print("- step3_file_summary.csv")
print()


# -----------------------------------------------------------------------------
# Step 3F: confirm that the key columns we need for the project are present
# -----------------------------------------------------------------------------
required_cols = [
    "ACN",
    "Time | Date",
    "Aircraft 1 | Flight Phase",
    "Events | Anomaly",
    "Report 1 | Narrative",
    "Report 1 | Synopsis",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
]

print("=" * 90)
print("KEY COLUMN CHECK IN THE COMBINED DATASET")
print("=" * 90)

for col in required_cols:
    print(f"{col}: {'FOUND' if col in combined_df.columns else 'NOT FOUND'}")

print()


# -----------------------------------------------------------------------------
# Step 3G: create a year field only for inspection purposes
# -----------------------------------------------------------------------------
# At this stage we are not cleaning or splitting yet. We are only extracting
# the year so that we can understand the temporal coverage of the combined raw
# dataset.
#
# The ASRS date field currently looks like values such as 201801, 202201, etc.
# We therefore take the first four characters as the year.
# -----------------------------------------------------------------------------
date_str = combined_df["Time | Date"].astype(str).str.replace(".0", "", regex=False).str.strip()
combined_df["incident_year_step3"] = date_str.str[:4]

print("=" * 90)
print("YEAR COVERAGE CHECK")
print("=" * 90)
print(combined_df["incident_year_step3"].value_counts(dropna=False).sort_index())
print()


# -----------------------------------------------------------------------------
# Step 3H: inspect flight-phase values
# -----------------------------------------------------------------------------
# This is a very important scope check. Even though the download was based on
# Initial Approach / Final Approach / Landing, it is common for aviation safety
# records to contain multiple phase labels in a single row.
#
# We are not filtering anything yet. We are only observing the raw values.
# -----------------------------------------------------------------------------
print("=" * 90)
print("TOP FLIGHT-PHASE VALUES (RAW, NO FILTERING YET)")
print("=" * 90)
print(combined_df["Aircraft 1 | Flight Phase"].fillna("MISSING").value_counts(dropna=False).head(25))
print()


# -----------------------------------------------------------------------------
# Step 3I: inspect anomaly values
# -----------------------------------------------------------------------------
# The anomaly field is expected to contain the phrase 'Unstabilized Approach'
# for positive cases, but we are still at the inspection stage. Therefore we
# only look at the raw anomaly text and count how many rows appear to mention
# that phrase.
# -----------------------------------------------------------------------------
anomaly_text = combined_df["Events | Anomaly"].fillna("").astype(str)

unstable_flag_preview = anomaly_text.str.contains("Unstabilized Approach", case=False, na=False)

print("=" * 90)
print("ANOMALY SCOPE CHECK")
print("=" * 90)
print(f"Rows whose anomaly text contains 'Unstabilized Approach': {int(unstable_flag_preview.sum())}")
print(f"Rows whose anomaly text does NOT contain it: {int((~unstable_flag_preview).sum())}")
print()


# -----------------------------------------------------------------------------
# Step 3J: show a short preview of the combined raw data
# -----------------------------------------------------------------------------
# This preview is purely for inspection. It helps confirm that the combined table
# contains the main narrative field, anomaly information, flight phase, and
# environmental context in one place.
# -----------------------------------------------------------------------------
preview_cols = [
    "ACN",
    "Time | Date",
    "Aircraft 1 | Flight Phase",
    "Events | Anomaly",
    "Report 1 | Narrative",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
    "source_file",
]

available_preview_cols = [col for col in preview_cols if col in combined_df.columns]

print("=" * 90)
print("PREVIEW OF THE COMBINED RAW DATA (FIRST 3 ROWS)")
print("=" * 90)
print(combined_df[available_preview_cols].head(3).to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Step 3K: save the combined raw dataset
# -----------------------------------------------------------------------------
# We save the combined file exactly as it exists after header reconstruction and
# concatenation. We are still not cleaning or filtering rows at this point.
# -----------------------------------------------------------------------------
combined_df.to_csv(project_folder / "data" / "processed" / "step3_main_phase_combined_raw.csv", index=False)

print("The combined raw dataset has been saved as:")
print("- step3_main_phase_combined_raw.csv")
print()

print("=" * 90)
print("STEP 3 FINISHED")
print("At this stage, the main raw files have been combined and inspected.")
print("No row filtering, no label creation, and no modelling has been done yet.")
print("=" * 90)