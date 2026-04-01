from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =============================================================================
# FIGURE 5 (REFINED): TOP PREDICTIVE N-GRAMS AND COEFFICIENTS
# =============================================================================
# Purpose of this script
# ----------------------
# This script creates a refined interpretability figure for the final selected
# text-only Logistic Regression model.
#
# Compared with a simple coefficient plot, this version is designed to be more
# suitable for report writing because it:
# 1. refits the final selected model on the development data (train + validation),
# 2. extracts the learned TF-IDF feature coefficients,
# 3. gives preference to more interpretable n-grams (especially bigrams),
# 4. separates unstable-associated and non-unstable-associated features into
#    two clearer panels.
#
# Why this refined version is better
# ----------------------------------
# In text classification, isolated words can sometimes look misleading when read
# outside context. For example, a unigram such as "stable" may appear in phrases
# like "not stable", which changes its practical interpretation. Since the final
# model uses both unigrams and bigrams, it is useful to prioritise more
# interpretable features where possible.
#
# Interpretation
# --------------
# Positive coefficients:
#     more associated with unstable-approach reports
#
# Negative coefficients:
#     more associated with non-unstable reports
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define project folder and input/output paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "step7_train_dataset.csv"
validation_file = project_folder / "step7_validation_dataset.csv"

summary_output = project_folder / "figure5_top_predictive_ngrams_refined_summary.csv"
figure_output = project_folder / "figure5_top_predictive_ngrams_refined.png"


# -----------------------------------------------------------------------------
# Step 2: Check that the required files exist
# -----------------------------------------------------------------------------
if not train_file.exists():
    raise FileNotFoundError(f"Train file not found: {train_file.name}")

if not validation_file.exists():
    raise FileNotFoundError(f"Validation file not found: {validation_file.name}")


# -----------------------------------------------------------------------------
# Step 3: Load and combine train + validation data
# -----------------------------------------------------------------------------
# This mirrors the final model-development logic used before locked test
# evaluation. The final selected model is therefore refit on the full
# development data available before testing.
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

dev_df = pd.concat([train_df, validation_df], ignore_index=True)
dev_df = dev_df.dropna(subset=["text_main", "label_unstable"]).copy()

X_text = dev_df["text_main"].astype(str)
y = dev_df["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Step 4: Refit the final selected TF-IDF representation
# -----------------------------------------------------------------------------
# These settings match the final selected text-only model reported in the study.
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)


# -----------------------------------------------------------------------------
# Step 5: Refit the final selected Logistic Regression model
# -----------------------------------------------------------------------------
model = LogisticRegression(
    C=3.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

model.fit(X, y)


# -----------------------------------------------------------------------------
# Step 6: Build a coefficient table
# -----------------------------------------------------------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

# Mark whether each feature is a unigram or bigram
coef_df["ngram_type"] = coef_df["feature"].apply(
    lambda x: "bigram" if " " in x else "unigram"
)

# Split into positive and negative sides
positive_df = coef_df[coef_df["coefficient"] > 0].copy()
negative_df = coef_df[coef_df["coefficient"] < 0].copy()


# -----------------------------------------------------------------------------
# Step 7: Helper function to select a more interpretable feature set
# -----------------------------------------------------------------------------
# The aim is to prefer bigrams where possible because they are often easier to
# interpret in a report. If there are not enough strong bigrams, the script
# fills the remaining places with the strongest unigrams.
# -----------------------------------------------------------------------------
def select_interpretable_features(df, top_n=10, preferred_bigrams=6, side="positive"):
    """
    Select a refined set of features for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and coefficients for one side.
    top_n : int
        Total number of features to return.
    preferred_bigrams : int
        Preferred number of bigrams to include where available.
    side : str
        'positive' or 'negative', used to determine sort direction.

    Returns
    -------
    pandas.DataFrame
        Selected features for plotting.
    """
    ascending = True if side == "negative" else False

    bigrams = (
        df[df["ngram_type"] == "bigram"]
        .sort_values("coefficient", ascending=ascending)
        .head(preferred_bigrams)
    )

    remaining_needed = top_n - len(bigrams)

    used_features = set(bigrams["feature"])

    unigrams = (
        df[(df["ngram_type"] == "unigram") & (~df["feature"].isin(used_features))]
        .sort_values("coefficient", ascending=ascending)
        .head(remaining_needed)
    )

    selected = pd.concat([bigrams, unigrams], ignore_index=True)

    # Final ordering for plotting
    selected = selected.sort_values("coefficient", ascending=True)
    return selected


top_n = 10
preferred_bigrams = 6

top_positive = select_interpretable_features(
    positive_df,
    top_n=top_n,
    preferred_bigrams=preferred_bigrams,
    side="positive"
)

top_negative = select_interpretable_features(
    negative_df,
    top_n=top_n,
    preferred_bigrams=preferred_bigrams,
    side="negative"
)

# Save the selected summary table for record-keeping
summary_df = pd.concat([
    top_negative.assign(direction="non_unstable_associated"),
    top_positive.assign(direction="unstable_associated")
], ignore_index=True)

summary_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Step 8: Create the refined two-panel plot
# -----------------------------------------------------------------------------
# Two separate horizontal bar charts make interpretation easier:
# - left panel: more associated with non-unstable reports
# - right panel: more associated with unstable-approach reports
# -----------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(
    ncols=2,
    figsize=(14, 8),
    sharey=False
)

# Left panel: negative coefficients
ax_left.barh(top_negative["feature"], top_negative["coefficient"])
ax_left.set_title("More associated with\nnon-unstable reports")
ax_left.set_xlabel("Coefficient")
ax_left.set_ylabel("Feature")
ax_left.axvline(0, linestyle="--", linewidth=1)

# Right panel: positive coefficients
ax_right.barh(top_positive["feature"], top_positive["coefficient"])
ax_right.set_title("More associated with\nunstable-approach reports")
ax_right.set_xlabel("Coefficient")
ax_right.axvline(0, linestyle="--", linewidth=1)

# Main title for the full figure
fig.suptitle("Top predictive n-grams and coefficients from the final Logistic Regression model", fontsize=15)

plt.tight_layout(rect=[0, 0, 1, 0.96])


# -----------------------------------------------------------------------------
# Step 9: Save the refined figure
# -----------------------------------------------------------------------------
plt.savefig(figure_output, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# Step 10: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 90)
print("REFINED FIGURE 5 CREATED SUCCESSFULLY")
print("=" * 90)
print(f"Summary file created: {summary_output.name}")
print(f"Figure file created:  {figure_output.name}")
print("=" * 90)
print("This refined PNG file is ready to be inserted into the report as Figure 5.")
print("=" * 90)