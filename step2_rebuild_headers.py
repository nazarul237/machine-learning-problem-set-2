from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 2: REBUILD THE ASRS COLUMN HEADERS PROPERLY
# =============================================================================
# Purpose of this step:
# In Step 1, we confirmed that all raw CSV files have a consistent structure,
# but we also discovered that the ASRS export uses a two-row header format.
#
# This means the first row contains broad groups such as:
# - Time
# - Place
# - Environment
# - Aircraft 1
# - Events
# - Report 1
#
# and the second row contains the specific variable names such as:
# - Date
# - Flight Phase
# - Anomaly
# - Narrative
#
# When pandas reads only the first row as the header, the true variable names
# are not reconstructed correctly. As a result, key fields such as
# "Time | Date" or "Events | Anomaly" do not appear in the imported dataset.
#
# The aim of this script is therefore NOT to clean the data yet.
# It only reconstructs the headers so that the raw ASRS files can be understood
# properly before we move on to any cleaning or modelling stage.
# =============================================================================


# -----------------------------------------------------------------------------
# Find all raw ASRS CSV files in the current folder
# -----------------------------------------------------------------------------
folder = Path(".")
csv_files = sorted(folder.glob("ASRS_DBOnline*.csv"))

if len(csv_files) == 0:
    print("No raw ASRS CSV files were found in this folder.")
    print("Please make sure the raw CSV files are saved in the same folder as this script.")
    raise SystemExit()

print("=" * 80)
print("STEP 2: REBUILDING THE ASRS HEADERS")
print("=" * 80)
print(f"Number of raw CSV files found: {len(csv_files)}")
print()


# -----------------------------------------------------------------------------
# Function to flatten the two-level ASRS header into one readable column name
# -----------------------------------------------------------------------------
# Example:
# ('Time', 'Date') -> 'Time | Date'
# ('Events', 'Anomaly') -> 'Events | Anomaly'
# (' ', 'ACN') -> 'ACN'
#
# If the top-level header is blank, we keep only the second-level header.
# -----------------------------------------------------------------------------
def flatten_asrs_columns(columns):
    flattened = []

    for top, bottom in columns:
        top = str(top).strip()
        bottom = str(bottom).strip()

        # If the top part is empty, unnamed, or blank, keep only the bottom part.
        if top == "" or top.startswith("Unnamed:"):
            new_name = bottom

        # If the bottom part is empty or unnamed, keep only the top part.
        elif bottom == "" or bottom.startswith("Unnamed:"):
            new_name = top

        # Otherwise combine both levels into one readable field name.
        else:
            new_name = f"{top} | {bottom}"

        flattened.append(new_name)

    return flattened


# We will store the flattened column lists for all files so that we can confirm
# whether the reconstructed structure is consistent across the raw exports.
all_flattened_column_lists = []


# -----------------------------------------------------------------------------
# Loop through each file and rebuild the headers
# -----------------------------------------------------------------------------
for file_path in csv_files:
    print("-" * 80)
    print(f"File name: {file_path.name}")

    # Read the file using BOTH header rows.
    # header=[0, 1] tells pandas that row 1 and row 2 together define the columns.
    df = pd.read_csv(file_path, header=[0, 1], low_memory=False)

    # Rebuild the headers into one flat list of readable column names.
    df.columns = flatten_asrs_columns(df.columns)

    # Save the flattened column structure for later consistency checking.
    all_flattened_column_lists.append(list(df.columns))

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nFirst 20 reconstructed column names:")
    for i, col in enumerate(df.columns[:20], start=1):
        print(f"{i}. {col}")

    # These are the key fields we hoped to recover.
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

    print("\nCheck whether key columns now exist:")
    for col in key_cols:
        exists = col in df.columns
        print(f"{col}: {'FOUND' if exists else 'NOT FOUND'}")

    # Show a small preview of the most important columns if they are present.
    available_key_cols = [col for col in key_cols if col in df.columns]

    print("\nPreview of important columns (first 3 rows):")
    if available_key_cols:
        print(df[available_key_cols].head(3).to_string(index=False))
    else:
        print("Key columns were still not found. We would need to inspect the raw file again.")

    print()


# -----------------------------------------------------------------------------
# Check whether all files now have the same reconstructed structure
# -----------------------------------------------------------------------------
print("=" * 80)
print("RECONSTRUCTED HEADER CONSISTENCY CHECK")
print("=" * 80)

first_columns = all_flattened_column_lists[0]
all_same = all(cols == first_columns for cols in all_flattened_column_lists)

if all_same:
    print("Good news: after rebuilding the two-row headers, all files still have the same structure.")
else:
    print("Warning: after rebuilding the headers, some files do not match exactly.")
    print("We will need to inspect those differences before cleaning.")

print("=" * 80)
print("STEP 2 FINISHED")
print("At this stage, we have only reconstructed and inspected the headers.")
print("No cleaning, merging, labelling, or modelling has been done yet.")
print("=" * 80)