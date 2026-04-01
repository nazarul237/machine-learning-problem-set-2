from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 1: INSPECT THE RAW ASRS CSV FILES
# =============================================================================
# Purpose of this step:
# At this stage, we are not performing any cleaning or modelling. The aim is to
# understand the structure of the raw files exactly as they were downloaded from
# the ASRS database.
#
# This is an important part of the assessment because good data preparation does
# not begin by deleting rows or transforming variables immediately. It begins by
# examining the raw material carefully and documenting what is actually present.
#
# In practical terms, this script helps us answer the following questions:
# 1. How many raw CSV files do we currently have?
# 2. What are the names of those files?
# 3. How many rows and columns are in each file?
# 4. Do all files share the same column structure?
# 5. Do the key fields we expect for this project actually exist?
#
# This inspection stage is especially important for the present project because
# our modelling task depends on a few critical variables:
# - the narrative text
# - the anomaly/event field
# - the flight phase field
# - environmental context such as weather and flight conditions
#
# If these fields are not present consistently across the raw files, we need to
# know that before moving on to cleaning or model building.
# =============================================================================


# -----------------------------------------------------------------------------
# Locate the raw CSV files
# -----------------------------------------------------------------------------
# We assume that this Python script is saved in the same folder as the raw ASRS
# CSV files. The pattern "ASRS_DBOnline*.csv" matches files such as:
# - ASRS_DBOnline.csv
# - ASRS_DBOnline (1).csv
# - ASRS_DBOnline (2).csv
# etc.
#
# Using Path('.') means "look in the current working folder".
# -----------------------------------------------------------------------------
folder = Path(".")
csv_files = sorted((project_folder / "data" / "raw").glob("ASRS_DBOnline*.csv"))

# If no files are found, we stop the script immediately and print a helpful
# message. This prevents confusion later and makes it obvious that the script
# and the raw files are not in the same location.
if len(csv_files) == 0:
    print("No raw ASRS CSV files were found in this folder.")
    print("Please make sure your raw CSV files are saved in the same folder as this Python script.")
    raise SystemExit()

print("=" * 80)
print("STEP 1: RAW DATA INSPECTION")
print("=" * 80)
print(f"Number of raw CSV files found: {len(csv_files)}")
print()

# We will store the column names from each file in this list so that we can
# later check whether all raw files follow the same structure.
all_column_lists = []


# -----------------------------------------------------------------------------
# Loop through each raw file and inspect it
# -----------------------------------------------------------------------------
# For each CSV file, we:
# 1. Read the file into a pandas DataFrame
# 2. Print the number of rows and columns
# 3. Print all column names
# 4. Show a short preview of the most important fields, if they exist
#
# This gives us both a structural view (rows/columns) and a content view
# (sample values in key fields).
# -----------------------------------------------------------------------------
for file_path in csv_files:
    print("-" * 80)
    print(f"File name: {file_path.name}")

    # Read the raw CSV file.
    # low_memory=False is used so pandas does not guess data types in chunks,
    # which can sometimes create unnecessary warnings when working with mixed
    # data in large files.
    df = pd.read_csv(file_path, low_memory=False)

    # Print the basic dimensions of the file.
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Save the list of column names so that we can check consistency across
    # all files after the loop finishes.
    all_column_lists.append(list(df.columns))

    # Print every column name in order. This is useful because we want to know
    # the exact field names coming from the ASRS export rather than guessing.
    print("\nColumn names:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    # -------------------------------------------------------------------------
    # Define the key columns that matter most for this project
    # -------------------------------------------------------------------------
    # These are the fields we most expect to use later in the analysis.
    # At this stage, we are only checking whether they exist in the raw data.
    #
    # Why these columns matter:
    # - ACN: unique report identifier
    # - Time | Date: needed for time-based analysis and future train/test split
    # - Aircraft 1 | Flight Phase: needed to confirm scope of the dataset
    # - Events | Anomaly: likely source of the unstable-approach label
    # - Report 1 | Narrative / Synopsis: the main text for text classification
    # - Environmental fields: useful later for contextual modelling
    # -------------------------------------------------------------------------
    key_cols = [
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

    # Keep only the columns from the list above that are actually present in
    # the current file. This avoids errors if some files are missing a field.
    available_key_cols = [col for col in key_cols if col in df.columns]

    print("\nPreview of important columns (first 3 rows):")

    # If at least some expected key columns exist, print the first 3 rows for
    # those columns. This helps us understand what the raw values look like.
    # We are still not cleaning anything here — we are only inspecting.
    if available_key_cols:
        print(df[available_key_cols].head(3).to_string(index=False))
    else:
        print("None of the expected key columns were found in this file.")

    print()


# -----------------------------------------------------------------------------
# Check whether all raw files have the same column structure
# -----------------------------------------------------------------------------
# This is a very important structural check.
#
# If all files have the same columns, merging them later will be much simpler.
# If they do not, then we need to investigate those differences before starting
# the cleaning stage.
# -----------------------------------------------------------------------------
print("=" * 80)
print("COLUMN CONSISTENCY CHECK")
print("=" * 80)

first_columns = all_column_lists[0]
all_same = all(cols == first_columns for cols in all_column_lists)

if all_same:
    print("Good news: all raw CSV files have the same column structure.")
else:
    print("Warning: not all raw CSV files have the same column structure.")
    print("This does not automatically mean something is wrong, but it does")
    print("mean we need to inspect those differences carefully in the next step.")

print("=" * 80)
print("STEP 1 FINISHED")
print("This script only inspected the raw files.")
print("No cleaning, merging, labelling, or modelling has been done yet.")
print("=" * 80)