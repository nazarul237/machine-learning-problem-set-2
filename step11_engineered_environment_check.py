from pathlib import Path
import re
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =============================================================================
# STEP 11: ENGINEERED ENVIRONMENTAL FEATURE CHECK
# =============================================================================
# Purpose of this step:
# In Step 10, we tested a broad text + context model using many safe context
# variables together. That result did not outperform the best tuned text-only
# model from Step 9.
#
# However, that does not necessarily mean that environmental information has no
# value. A more likely explanation is that the raw context fields were too broad,
# too sparse, or too messy when used directly.
#
# Therefore, in this step, we run a more focused and methodologically cleaner
# comparison:
#
# Model A: best text-only Logistic Regression from Step 9
# Model B: the same Logistic Regression, but with a smaller set of engineered
#          environmental features added to the text
#
# This is an ablation / robustness style check. The purpose is to ask whether a
# more carefully designed environmental feature set adds value beyond narrative
# text alone.
#
# Important methodological note:
# - We still do NOT use the test set.
# - We are only working with the training and validation sets.
# - We keep the same tuned TF-IDF and Logistic Regression settings so that the
#   comparison remains fair.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"

comparison_file = project_folder / "results" / "tables" / "step11_engineered_environment_comparison.csv"
feature_preview_file = project_folder / "results" / "tables" / "step11_engineered_environment_feature_preview.csv"
predictions_file = project_folder / "results" / "tables" / "step11_engineered_environment_validation_predictions.csv"
cm_file = project_folder / "results" / "tables" / "step11_engineered_environment_confusion_matrix.csv"


# -----------------------------------------------------------------------------
# Load the train and validation sets
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 11: ENGINEERED ENVIRONMENTAL FEATURE CHECK")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Helper functions for environmental feature engineering
# -----------------------------------------------------------------------------
# These functions convert raw fields into simpler, more interpretable categories.
# The goal is not to create perfect aviation weather variables, but to create a
# cleaner and more usable feature set than the raw mixed-format fields.
# -----------------------------------------------------------------------------
def clean_string(x):
    """
    Convert values into a stripped string while preserving missingness as the
    explicit category 'Missing'. This is useful for categorical modelling.
    """
    if pd.isna(x):
        return "Missing"
    x = str(x).strip()
    return x if x != "" else "Missing"


def extract_month_from_date(x):
    """
    The ASRS date field is typically in YYYYMM format, for example:
    201901 -> January 2019

    This function extracts the month as an integer if possible.
    """
    if pd.isna(x):
        return None

    s = str(x).replace(".0", "").strip()

    if len(s) >= 6 and s[:6].isdigit():
        month_value = int(s[4:6])
        if 1 <= month_value <= 12:
            return month_value

    return None


def month_to_season(month_value):
    """
    Convert a numeric month into a coarse season category.
    This is a simple derived environmental feature that may capture broad
    seasonal operating differences.
    """
    if month_value is None:
        return "Missing"

    if month_value in [12, 1, 2]:
        return "Winter"
    elif month_value in [3, 4, 5]:
        return "Spring"
    elif month_value in [6, 7, 8]:
        return "Summer"
    elif month_value in [9, 10, 11]:
        return "Autumn"
    else:
        return "Missing"


def bucket_ceiling(x):
    """
    Convert the raw ceiling field into a smaller set of categories.

    Why do this?
    The raw ceiling field is very sparse and contains many unique values.
    Using buckets is more stable and more interpretable.

    Rules:
    - Missing -> Missing
    - CLR -> Clear
    - numeric values -> bucketed into low/medium/high groups
    - anything else -> Other
    """
    if pd.isna(x):
        return "Missing"

    s = str(x).strip().upper()

    if s == "" or s == "NAN":
        return "Missing"

    if s == "CLR":
        return "Clear"

    if s.isdigit():
        val = int(s)

        if val <= 1000:
            return "Very Low"
        elif val <= 3000:
            return "Low"
        elif val <= 10000:
            return "Medium"
        else:
            return "High"

    return "Other"


def extract_visibility_bucket(x):
    """
    The weather/visibility field is mixed-format. Some rows contain text such as
    'Turbulence' or 'Rain', while others contain values like 10, 20, etc.

    Here we look for the first numeric value and convert it into a coarse bucket.
    If no number is present, we return 'No Numeric Visibility'.
    """
    if pd.isna(x):
        return "Missing"

    s = str(x).strip()
    if s == "":
        return "Missing"

    numbers = re.findall(r"\b\d+\b", s)

    if len(numbers) == 0:
        return "No Numeric Visibility"

    value = int(numbers[0])

    if value <= 5:
        return "0_to_5"
    elif value <= 10:
        return "6_to_10"
    else:
        return "11_plus"


def contains_keyword(raw_text, keyword_list):
    """
    Check whether any keyword from a list appears in the weather text.
    Returns 'Yes' or 'No' so that the variable remains clearly categorical.
    """
    if pd.isna(raw_text):
        return "No"

    s = str(raw_text).lower()

    for kw in keyword_list:
        if kw in s:
            return "Yes"

    return "No"


def build_engineered_environment_features(df):
    """
    Create a new dataframe containing the original text plus engineered
    environmental features.
    """
    work = df.copy()

    # Ensure text is always a usable string
    work["text_main"] = work["text_main"].fillna("").astype(str)

    # Clean basic categorical environmental fields
    work["env_flight_conditions"] = work["Environment | Flight Conditions"].apply(clean_string)
    work["env_light"] = work["Environment | Light"].apply(clean_string)

    # Month and season derived from the incident date
    work["env_month_num"] = work["Time | Date"].apply(extract_month_from_date)
    work["env_month"] = work["env_month_num"].apply(
        lambda x: f"Month_{int(x):02d}" if x is not None else "Missing"
    )
    work["env_season"] = work["env_month_num"].apply(month_to_season)

    # Bucket the ceiling field
    work["env_ceiling_bucket"] = work["Environment | Ceiling"].apply(bucket_ceiling)

    # Derive a coarse visibility bucket from the mixed weather/visibility field
    work["env_visibility_bucket"] = work["Environment | Weather Elements / Visibility"].apply(extract_visibility_bucket)

    # Raw weather text used only for keyword extraction
    weather_raw = work["Environment | Weather Elements / Visibility"]

    # Create simple weather hazard flags
    work["env_has_turbulence"] = weather_raw.apply(lambda x: contains_keyword(x, ["turbulence"]))
    work["env_has_windshear"] = weather_raw.apply(lambda x: contains_keyword(x, ["windshear"]))
    work["env_has_rain"] = weather_raw.apply(lambda x: contains_keyword(x, ["rain"]))
    work["env_has_icing"] = weather_raw.apply(lambda x: contains_keyword(x, ["icing"]))
    work["env_has_cloudy"] = weather_raw.apply(lambda x: contains_keyword(x, ["cloudy"]))
    work["env_has_fog"] = weather_raw.apply(lambda x: contains_keyword(x, ["fog"]))
    work["env_has_haze_smoke"] = weather_raw.apply(lambda x: contains_keyword(x, ["haze", "smoke"]))
    work["env_has_thunderstorm"] = weather_raw.apply(lambda x: contains_keyword(x, ["thunderstorm"]))
    work["env_has_snow"] = weather_raw.apply(lambda x: contains_keyword(x, ["snow"]))

    # Broad flag to indicate whether any adverse weather wording appears
    adverse_keywords = [
        "turbulence", "windshear", "rain", "icing", "cloudy",
        "fog", "haze", "smoke", "thunderstorm", "snow"
    ]
    work["env_any_adverse_weather"] = weather_raw.apply(lambda x: contains_keyword(x, adverse_keywords))

    return work


# -----------------------------------------------------------------------------
# Build engineered feature versions of the train and validation sets
# -----------------------------------------------------------------------------
train_eng = build_engineered_environment_features(train_df)
valid_eng = build_engineered_environment_features(validation_df)

engineered_columns = [
    "env_flight_conditions",
    "env_light",
    "env_month",
    "env_season",
    "env_ceiling_bucket",
    "env_visibility_bucket",
    "env_has_turbulence",
    "env_has_windshear",
    "env_has_rain",
    "env_has_icing",
    "env_has_cloudy",
    "env_has_fog",
    "env_has_haze_smoke",
    "env_has_thunderstorm",
    "env_has_snow",
    "env_any_adverse_weather",
]

# Save a small preview so you can inspect the engineered features later
preview_cols = ["ACN", "incident_year", "label_unstable", "text_main"] + engineered_columns
train_eng[preview_cols].head(30).to_csv(feature_preview_file, index=False)

print("ENGINEERED ENVIRONMENTAL FEATURES CREATED")
print("-" * 100)
for col in engineered_columns:
    print(f" - {col}")
print()


# -----------------------------------------------------------------------------
# Prepare the target variable
# -----------------------------------------------------------------------------
y_train = train_eng["label_unstable"].astype(int)
y_valid = valid_eng["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Model A: best text-only Logistic Regression from Step 9
# -----------------------------------------------------------------------------
# We keep the best Step 9 settings fixed so that the comparison remains fair.
# -----------------------------------------------------------------------------
text_only_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3
)

text_only_model = LogisticRegression(
    C=3.0,
    class_weight="balanced",
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

X_train_text = train_eng["text_main"]
X_valid_text = valid_eng["text_main"]

X_train_text_tfidf = text_only_vectorizer.fit_transform(X_train_text)
X_valid_text_tfidf = text_only_vectorizer.transform(X_valid_text)

text_only_model.fit(X_train_text_tfidf, y_train)

y_pred_text = text_only_model.predict(X_valid_text_tfidf)
y_score_text = text_only_model.predict_proba(X_valid_text_tfidf)[:, 1]


# -----------------------------------------------------------------------------
# Model B: text + engineered environmental features
# -----------------------------------------------------------------------------
# The text settings and Logistic Regression settings are intentionally kept the
# same as the best text-only model. This means the only real change is the
# addition of the engineered environmental variables.
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=3
            ),
            "text_main"
        ),
        (
            "environment",
            OneHotEncoder(handle_unknown="ignore"),
            engineered_columns
        )
    ]
)

text_plus_environment_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=3.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])

X_train_env = train_eng[["text_main"] + engineered_columns]
X_valid_env = valid_eng[["text_main"] + engineered_columns]

text_plus_environment_pipeline.fit(X_train_env, y_train)

y_pred_env = text_plus_environment_pipeline.predict(X_valid_env)

# For PR-AUC and ROC-AUC we need scores rather than only class labels
X_valid_env_transformed = text_plus_environment_pipeline.named_steps["preprocessor"].transform(X_valid_env)
y_score_env = text_plus_environment_pipeline.named_steps["model"].predict_proba(X_valid_env_transformed)[:, 1]


# -----------------------------------------------------------------------------
# Helper function to calculate metrics
# -----------------------------------------------------------------------------
def make_metric_row(model_name, y_true, y_pred, y_score):
    """
    Compute a set of imbalance-aware metrics for one candidate model.
    """
    return {
        "model": model_name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "validation_rows": len(y_true),
        "validation_unstable_cases": int(y_true.sum())
    }


# -----------------------------------------------------------------------------
# Create the direct comparison table
# -----------------------------------------------------------------------------
comparison_df = pd.DataFrame([
    make_metric_row("best_text_only_logistic_regression", y_valid, y_pred_text, y_score_text),
    make_metric_row("text_plus_engineered_environment_logistic_regression", y_valid, y_pred_env, y_score_env),
])

comparison_df.to_csv(comparison_file, index=False)


# -----------------------------------------------------------------------------
# Save validation predictions and confusion matrix for the engineered model
# -----------------------------------------------------------------------------
predictions_df = pd.DataFrame({
    "ACN": valid_eng["ACN"],
    "incident_year": valid_eng["incident_year"],
    "true_label": y_valid,
    "predicted_label": y_pred_env,
    "score": y_score_env
})

predictions_df.to_csv(predictions_file, index=False)

cm = confusion_matrix(y_valid, y_pred_env)
cm_df = pd.DataFrame(
    cm,
    index=["actual_0_non_unstable", "actual_1_unstable"],
    columns=["predicted_0_non_unstable", "predicted_1_unstable"]
)
cm_df.to_csv(cm_file)


# -----------------------------------------------------------------------------
# Print the final comparison
# -----------------------------------------------------------------------------
print("=" * 100)
print("COMPARISON: BEST TEXT-ONLY MODEL VS ENGINEERED ENVIRONMENT MODEL")
print("=" * 100)
print(comparison_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(comparison_file.name)
print(feature_preview_file.name)
print(predictions_file.name)
print(cm_file.name)
print()

print("=" * 100)
print("STEP 11 FINISHED")
print("This step compared the best text-only model against a smaller engineered")
print("environmental feature set on the validation data.")
print("The test set has still NOT been used.")
print("=" * 100)