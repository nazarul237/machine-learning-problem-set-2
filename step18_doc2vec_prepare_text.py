from pathlib import Path
import re
import pandas as pd

# =============================================================================
# STEP 18: PREPARE TEXT FOR DOC2VEC
# =============================================================================
# Purpose of this step:
# The project has already completed Method 1, which used TF-IDF based text
# classification. We now begin Method 2, which will use Doc2Vec as a different
# text representation approach.
#
# Before a Doc2Vec model can be trained, the text needs to be prepared in a form
# that is suitable for document embedding. This step therefore does NOT train
# any model yet. Instead, it prepares the train, validation, and test text files
# in a clean and reproducible way.
#
# In practical terms, this script does the following:
# 1. Loads the existing time-based split datasets from Step 7
# 2. Keeps the essential columns needed for Doc2Vec modelling
# 3. Performs light text cleaning
# 4. Creates tokenised text suitable for later document-vector training
# 5. Saves prepared CSV files for train, validation, and test
#
# Important methodological note:
# This step does not use the labels to alter the text in any way.
# It only prepares the narratives as input documents.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
# Using the folder of the current script makes the workflow more robust and
# avoids path errors if the terminal was opened from a different location.
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "step7_train_dataset.csv"
validation_file = project_folder / "step7_validation_dataset.csv"
test_file = project_folder / "step7_test_dataset.csv"

train_output = project_folder / "step18_doc2vec_train_prepared.csv"
validation_output = project_folder / "step18_doc2vec_validation_prepared.csv"
test_output = project_folder / "step18_doc2vec_test_prepared.csv"

summary_output = project_folder / "step18_doc2vec_prepare_summary.csv"


# -----------------------------------------------------------------------------
# Load the split datasets
# -----------------------------------------------------------------------------
# We use the already-defined time-based splits:
# - train = 2018 to 2023
# - validation = 2024
# - test = 2025
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)

print("=" * 100)
print("STEP 18: PREPARE TEXT FOR DOC2VEC")
print("=" * 100)
print(f"Train rows loaded:      {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print(f"Test rows loaded:       {len(test_df)}")
print()


# -----------------------------------------------------------------------------
# Define a light text-cleaning function
# -----------------------------------------------------------------------------
# This is intentionally simple.
#
# Why keep it light?
# For this project, the aim is not to aggressively normalise the aviation
# narratives. Over-cleaning may remove useful operational wording. Therefore,
# this function only:
# - converts text to lowercase
# - removes extra whitespace
# - removes non-alphanumeric characters except spaces
# - keeps a clean natural-language style suitable for tokenisation
#
# This is enough for a Doc2Vec preparation stage.
# -----------------------------------------------------------------------------
def clean_text_for_doc2vec(text):
    """
    Lightly clean a narrative for Doc2Vec preparation.

    Parameters
    ----------
    text : any
        Raw text from the narrative field.

    Returns
    -------
    str
        A lightly cleaned text string.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Replace line breaks and tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")

    # Keep letters, numbers, and spaces only
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------------------------------------------------
# Define a simple tokenisation function
# -----------------------------------------------------------------------------
# Doc2Vec eventually needs tokenised text.
# Here we create tokens by splitting the cleaned text on spaces.
#
# We also remove empty strings and very short one-character leftovers.
# -----------------------------------------------------------------------------
def tokenize_text(cleaned_text):
    """
    Convert cleaned text into a list of tokens suitable for Doc2Vec.

    Parameters
    ----------
    cleaned_text : str
        Cleaned text string.

    Returns
    -------
    list[str]
        Token list.
    """
    if cleaned_text == "":
        return []

    tokens = cleaned_text.split(" ")
    tokens = [tok for tok in tokens if len(tok) > 1]

    return tokens


# -----------------------------------------------------------------------------
# Prepare one split dataframe
# -----------------------------------------------------------------------------
# This function creates a clean preparation table for one dataset split.
# It keeps:
# - ACN
# - incident_year
# - label_unstable
# - original text_main
# - cleaned_text
# - token_count
# - tokenised_text (stored as a single space-joined string for CSV storage)
#
# Why store tokenised text as a string?
# CSV files do not store Python lists cleanly. Saving tokens as a single
# space-joined string makes it easy to read them back and split again in the
# next step.
# -----------------------------------------------------------------------------
def prepare_split_for_doc2vec(df, split_name):
    """
    Prepare one split (train / validation / test) for Doc2Vec.

    Parameters
    ----------
    df : pandas.DataFrame
        One of the Step 7 split dataframes.
    split_name : str
        Label for the split.

    Returns
    -------
    pandas.DataFrame
        Prepared dataframe for Doc2Vec.
    """
    work = df.copy()

    # Keep only the columns required for Method 2
    keep_cols = ["ACN", "incident_year", "label_unstable", "text_main"]
    work = work[keep_cols].copy()

    # Ensure text is present as a string
    work["text_main"] = work["text_main"].fillna("").astype(str)

    # Create a cleaned text field
    work["cleaned_text"] = work["text_main"].apply(clean_text_for_doc2vec)

    # Tokenise the cleaned text
    work["token_list"] = work["cleaned_text"].apply(tokenize_text)

    # Count tokens so we can inspect document length at this stage
    work["token_count"] = work["token_list"].apply(len)

    # Save the token list as a single string for CSV compatibility
    work["tokenised_text"] = work["token_list"].apply(lambda x: " ".join(x))

    # Add the split label for clarity
    work["data_split"] = split_name

    # Drop rows with no usable tokens
    # This is rare, but if a narrative becomes empty after cleaning, it would not
    # be useful for Doc2Vec training or inference.
    before_rows = len(work)
    work = work[work["token_count"] > 0].copy()
    after_rows = len(work)

    print(f"{split_name} rows before token filter: {before_rows}")
    print(f"{split_name} rows after token filter:  {after_rows}")
    print()

    # Final column order
    final_cols = [
        "ACN",
        "incident_year",
        "label_unstable",
        "data_split",
        "text_main",
        "cleaned_text",
        "token_count",
        "tokenised_text",
    ]

    return work[final_cols].copy()


# -----------------------------------------------------------------------------
# Run preparation for all three splits
# -----------------------------------------------------------------------------
train_prepared = prepare_split_for_doc2vec(train_df, "train")
validation_prepared = prepare_split_for_doc2vec(validation_df, "validation")
test_prepared = prepare_split_for_doc2vec(test_df, "test")


# -----------------------------------------------------------------------------
# Save the prepared files
# -----------------------------------------------------------------------------
train_prepared.to_csv(train_output, index=False)
validation_prepared.to_csv(validation_output, index=False)
test_prepared.to_csv(test_output, index=False)


# -----------------------------------------------------------------------------
# Save a compact summary table
# -----------------------------------------------------------------------------
# This summary is useful for checking whether tokenisation changed the dataset
# size meaningfully and for documenting average document length.
# -----------------------------------------------------------------------------
summary_df = pd.DataFrame([
    {
        "split": "train",
        "rows": len(train_prepared),
        "unstable_cases": int(train_prepared["label_unstable"].sum()),
        "mean_token_count": round(train_prepared["token_count"].mean(), 2),
        "median_token_count": round(train_prepared["token_count"].median(), 2),
    },
    {
        "split": "validation",
        "rows": len(validation_prepared),
        "unstable_cases": int(validation_prepared["label_unstable"].sum()),
        "mean_token_count": round(validation_prepared["token_count"].mean(), 2),
        "median_token_count": round(validation_prepared["token_count"].median(), 2),
    },
    {
        "split": "test",
        "rows": len(test_prepared),
        "unstable_cases": int(test_prepared["label_unstable"].sum()),
        "mean_token_count": round(test_prepared["token_count"].mean(), 2),
        "median_token_count": round(test_prepared["token_count"].median(), 2),
    }
])

summary_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Print final outputs
# -----------------------------------------------------------------------------
print("=" * 100)
print("DOC2VEC PREPARATION SUMMARY")
print("=" * 100)
print(summary_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(train_output.name)
print(validation_output.name)
print(test_output.name)
print(summary_output.name)
print()

print("=" * 100)
print("STEP 18 FINISHED")
print("The text has been prepared for Doc2Vec.")
print("No Doc2Vec model has been trained yet.")
print("=" * 100)