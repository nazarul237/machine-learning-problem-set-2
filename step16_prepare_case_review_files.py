from pathlib import Path
import pandas as pd

# =============================================================================
# STEP 16: PREPARE CASE-REVIEW FILES FOR QUALITATIVE INTERPRETATION
# =============================================================================
# Purpose of this step:
# The previous step (error analysis) identified which test-set cases became:
# - false positives
# - false negatives
# - true positives
#
# That gives us the quantitative structure of model errors, but a strong report
# also needs some qualitative interpretation. In other words, we now want to
# read selected examples and ask:
# - What kind of wording appears in false alarms?
# - What kind of unstable cases are being missed?
# - What kinds of narratives does the model clearly recognise well?
#
# This step does NOT change the model or the data.
# It only prepares easier-to-read review files so that the user can manually
# inspect representative examples for the discussion section.
#
# The output files will contain:
# - ACN
# - year
# - phase
# - flight conditions
# - weather / visibility
# - score
# - word count
# - a shortened narrative excerpt
#
# The goal is to make manual case reading easier and more structured.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

fp_input = project_folder / "results" / "tables" / "step15_top_false_positives.csv"
fn_input = project_folder / "results" / "tables" / "step15_top_false_negatives.csv"
tp_input = project_folder / "results" / "tables" / "step15_top_true_positives.csv"

fp_output = project_folder / "results" / "tables" / "step16_case_review_false_positives.csv"
fn_output = project_folder / "results" / "tables" / "step16_case_review_false_negatives.csv"
tp_output = project_folder / "results" / "tables" / "step16_case_review_true_positives.csv"


# -----------------------------------------------------------------------------
# Helper function to prepare a readable case-review table
# -----------------------------------------------------------------------------
def prepare_case_review(input_file, output_file, group_name, excerpt_chars=700):
    """
    Read one of the top-example files from Step 15 and create a cleaner,
    easier-to-review version with shortened narrative excerpts.

    Parameters
    ----------
    input_file : Path
        Source CSV file created in Step 15.
    output_file : Path
        Output CSV file for manual case review.
    group_name : str
        Descriptive label for the case group (FP, FN, TP).
    excerpt_chars : int
        Maximum number of characters to keep in the narrative excerpt.
    """
    df = pd.read_csv(input_file, low_memory=False)

    # Ensure the narrative column is always a string
    df["text_main"] = df["text_main"].fillna("").astype(str).str.strip()

    # Create a shorter excerpt so the file is easier to read in Excel / CSV form
    df["narrative_excerpt"] = df["text_main"].str.slice(0, excerpt_chars)

    # Add the case group label for clarity
    df["case_group"] = group_name

    # Keep only the columns that are most useful for manual interpretation
    keep_cols = [
        "case_group",
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
        "text_word_count",
        "narrative_excerpt",
    ]

    review_df = df[keep_cols].copy()

    # Save the cleaned review file
    review_df.to_csv(output_file, index=False)

    return review_df


# -----------------------------------------------------------------------------
# Run the preparation for the three example groups
# -----------------------------------------------------------------------------
print("=" * 100)
print("STEP 16: PREPARE CASE-REVIEW FILES")
print("=" * 100)

fp_review = prepare_case_review(
    input_file=fp_input,
    output_file=fp_output,
    group_name="False Positive"
)

fn_review = prepare_case_review(
    input_file=fn_input,
    output_file=fn_output,
    group_name="False Negative"
)

tp_review = prepare_case_review(
    input_file=tp_input,
    output_file=tp_output,
    group_name="True Positive"
)

print("ROWS PREPARED FOR MANUAL REVIEW")
print("-" * 100)
print(f"False Positive review rows: {len(fp_review)}")
print(f"False Negative review rows: {len(fn_review)}")
print(f"True Positive review rows:  {len(tp_review)}")
print()

print("FILES CREATED")
print("-" * 100)
print(fp_output.name)
print(fn_output.name)
print(tp_output.name)
print()

print("=" * 100)
print("STEP 16 FINISHED")
print("Case-review files are now ready for manual reading and qualitative discussion.")
print("No modelling or re-fitting has been done.")
print("=" * 100)