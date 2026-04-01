from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =============================================================================
# FIGURE 5: TOP PREDICTIVE WORDS AND COEFFICIENTS
# =============================================================================
# Purpose of this script
# ----------------------
# This script recreates the final selected text-only model used in the study and
# extracts the strongest textual features from it.
#
# The final selected model in the project was:
# - TF-IDF text representation
# - Logistic Regression classifier
# - trained using the development data available before final testing
#
# To stay consistent with the final locked evaluation procedure, this script:
# 1. loads the train and validation datasets,
# 2. combines them into one development dataset,
# 3. refits the final selected TF-IDF + Logistic Regression model, and
# 4. extracts the largest positive and negative coefficients.
#
# Why this figure is useful
# -------------------------
# In the report, Figure 5 supports the interpretability discussion. It helps show
# which words or short phrases are most strongly associated with unstable-approach
# classification and which are associated with the non-unstable class.
#
# Interpretation of coefficients
# ------------------------------
# Positive coefficients:
#     features associated more strongly with the unstable class
#
# Negative coefficients:
#     features associated more strongly with the non-unstable class
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Define project folder and file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "step7_train_dataset.csv"
validation_file = project_folder / "step7_validation_dataset.csv"

summary_output = project_folder / "figure5_top_predictive_words_summary.csv"
figure_output = project_folder / "figure5_top_predictive_words.png"


# -----------------------------------------------------------------------------
# Step 2: Check that the required input files exist
# -----------------------------------------------------------------------------
if not train_file.exists():
    raise FileNotFoundError(f"Train file not found: {train_file.name}")

if not validation_file.exists():
    raise FileNotFoundError(f"Validation file not found: {validation_file.name}")


# -----------------------------------------------------------------------------
# Step 3: Load the train and validation datasets
# -----------------------------------------------------------------------------
# The final selected model was trained on development data before evaluation on
# the untouched 2025 test set. To mirror that setup, this script combines the
# train and validation sets before fitting the model.
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

dev_df = pd.concat([train_df, validation_df], ignore_index=True)

# Keep only rows where text and label are available
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
# Step 5: Refit the final selected Logistic Regression classifier
# -----------------------------------------------------------------------------
# These settings match the chosen tuned Logistic Regression model.
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
# Step 6: Extract feature names and coefficients
# -----------------------------------------------------------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

# Sort coefficients from most negative to most positive
coef_df = coef_df.sort_values("coefficient")


# -----------------------------------------------------------------------------
# Step 7: Select the strongest positive and negative features
# -----------------------------------------------------------------------------
# Positive features are most associated with the unstable class.
# Negative features are most associated with the non-unstable class.
#
# You can change the number 10 below if you want to show more or fewer features.
# -----------------------------------------------------------------------------
top_n = 10

top_negative = coef_df.head(top_n).copy()
top_positive = coef_df.tail(top_n).copy()

# Combine them into one plotting table
plot_df = pd.concat([top_negative, top_positive], ignore_index=True)

# Save summary for reference
plot_df.to_csv(summary_output, index=False)


# -----------------------------------------------------------------------------
# Step 8: Create the coefficient plot
# -----------------------------------------------------------------------------
# A horizontal bar chart is used because it is the clearest way to display word
# features and their associated coefficients in a readable report figure.
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

ax.barh(plot_df["feature"], plot_df["coefficient"])

ax.set_title("Top predictive words and coefficients from the final Logistic Regression model")
ax.set_xlabel("Coefficient")
ax.set_ylabel("Feature")
ax.axvline(0, linestyle="--", linewidth=1)

# Add short text labels to help interpretation
ax.text(
    x=plot_df["coefficient"].min(),
    y=-1.2,
    s="More associated with non-unstable reports",
    fontsize=9,
    ha="left"
)

ax.text(
    x=plot_df["coefficient"].max(),
    y=-1.2,
    s="More associated with unstable-approach reports",
    fontsize=9,
    ha="right"
)

plt.tight_layout()


# -----------------------------------------------------------------------------
# Step 9: Save the figure
# -----------------------------------------------------------------------------
plt.savefig(figure_output, dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# Step 10: Print confirmation
# -----------------------------------------------------------------------------
print("=" * 90)
print("FIGURE 5 CREATED SUCCESSFULLY")
print("=" * 90)
print(f"Summary file created: {summary_output.name}")
print(f"Figure file created:  {figure_output.name}")
print("=" * 90)
print("This PNG file is ready to be inserted into the report as Figure 5.")
print("=" * 90)