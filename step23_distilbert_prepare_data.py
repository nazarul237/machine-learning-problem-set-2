from pathlib import Path
import re
import pandas as pd

# =============================================================================
# STEP 23: PREPARE DATA FOR DISTILBERT
# =============================================================================
# Purpose of this step:
# Method 3 uses a transformer-based transfer learning model rather than a
# classical sparse representation (TF-IDF) or dense document embedding approach
# (Doc2Vec). Before that model can be trained, the train, validation, and test
# data must be prepared in a clean and consistent format.
#
# This step does NOT train DistilBERT yet. It only:
# 1. loads the time-based split datasets from Step 7
# 2. keeps the required columns
# 3. performs very light text cleaning
# 4. saves clean CSV files for later transformer training
#
# Important note:
# Transformer models are generally stronger when the text is not over-cleaned.
# Therefore, this step preserves the natural wording of the aviation narratives
# as much as possible while still removing obvious formatting noise.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"
test_file = project_folder / "data" / "splits" / "step7_test_dataset.csv"

train_output = project_folder / "data" / "processed" / "step23_distilbert_train.csv"
validation_output = project_folder / "data" / "processed" / "step23_distilbert_validation.csv"
test_output = project_folder / "data" / "processed" / "step23_distilbert_test.csv"
summary_output = project_folder / "results" / "tables" / "step23_distilbert_prepare_summary.csv"


# -----------------------------------------------------------------------------
# Load split datasets
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)

print("=" * 100)
print("STEP 23: PREPARE DATA FOR DISTILBERT")
print("=" * 100)
print(f"Train rows loaded:      {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print(f"Test rows loaded:       {len(test_df)}")
print()


# -----------------------------------------------------------------------------
# Light cleaning function for transformer text
# -----------------------------------------------------------------------------
# This cleaning is intentionally lighter than the Doc2Vec preparation.
# We preserve punctuation and normal sentence structure because transformers
# benefit from more natural text.
# -----------------------------------------------------------------------------
def clean_text_for_transformer(text):
    """
    Lightly clean a narrative for transformer training.

    Parameters
    ----------
    text : any
        Raw narrative text.

    Returns
    -------
    str
        Lightly cleaned text.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Replace tabs/newlines with spaces
    text = text.replace("\n", " ").replace("\t", " ")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------------------------------------------------
# Prepare one split
# -----------------------------------------------------------------------------
def prepare_split(df, split_name):
    """
    Prepare one split for DistilBERT training.

    Keeps only the fields needed later for training and evaluation.
    """
    work = df[["ACN", "incident_year", "label_unstable", "text_main"]].copy()

    work["text_main"] = work["text_main"].fillna("").astype(str)
    work["cleaned_text"] = work["text_main"].apply(clean_text_for_transformer)

    # Basic length checks
    work["char_length"] = work["cleaned_text"].str.len()
    work["word_count"] = work["cleaned_text"].str.split().str.len()

    # Remove rows that still have no usable text
    before_rows = len(work)
    work = work[work["cleaned_text"].str.strip() != ""].copy()
    after_rows = len(work)

    print(f"{split_name} rows before empty-text filter: {before_rows}")
    print(f"{split_name} rows after empty-text filter:  {after_rows}")
    print()

    work["data_split"] = split_name

    final_cols = [
        "ACN",
        "incident_year",
        "label_unstable",
        "data_split",
        "cleaned_text",
        "char_length",
        "word_count"
    ]

    return work[final_cols].copy()


train_prepared = prepare_split(train_df, "train")
validation_prepared = prepare_split(validation_df, "validation")
test_prepared = prepare_split(test_df, "test")


# -----------------------------------------------------------------------------
# Save prepared files
# -----------------------------------------------------------------------------
train_prepared.to_csv(train_output, index=False)
validation_prepared.to_csv(validation_output, index=False)
test_prepared.to_csv(test_output, index=False)


# -----------------------------------------------------------------------------
# Save summary table
# -----------------------------------------------------------------------------
summary_df = pd.DataFrame([
    {
        "split": "train",
        "rows": len(train_prepared),
        "unstable_cases": int(train_prepared["label_unstable"].sum()),
        "mean_word_count": round(train_prepared["word_count"].mean(), 2),
        "median_word_count": round(train_prepared["word_count"].median(), 2),
    },
    {
        "split": "validation",
        "rows": len(validation_prepared),
        "unstable_cases": int(validation_prepared["label_unstable"].sum()),
        "mean_word_count": round(validation_prepared["word_count"].mean(), 2),
        "median_word_count": round(validation_prepared["word_count"].median(), 2),
    },
    {
        "split": "test",
        "rows": len(test_prepared),
        "unstable_cases": int(test_prepared["label_unstable"].sum()),
        "mean_word_count": round(test_prepared["word_count"].mean(), 2),
        "median_word_count": round(test_prepared["word_count"].median(), 2),
    }
])

summary_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Final output
# -----------------------------------------------------------------------------
print("=" * 100)
print("DISTILBERT PREPARATION SUMMARY")
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
print("STEP 23 FINISHED")
print("The data has been prepared for DistilBERT.")
print("No transformer model has been trained yet.")
print("=" * 100)