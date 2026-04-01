from pathlib import Path
import re
import pandas as pd

from sklearn.compose import ColumnTransformer
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
# STEP 13: ENVIRONMENT-ONLY LEAKAGE DIAGNOSTIC
# =============================================================================
# Purpose of this step:
# In earlier steps, the text-only model remained the strongest overall model,
# while the engineered environmental feature model came close but did not
# outperform it. That pattern does not strongly suggest obvious leakage.
#
# However, to test this more carefully, we now run a targeted diagnostic:
#
# 1. Fit an environment-only Logistic Regression model
#    - This model uses NO narrative text at all.
#    - It only uses engineered environmental features.
#
# 2. Remove potentially suspicious engineered features one by one
#    - env_any_adverse_weather
#    - env_has_windshear
#    - env_has_turbulence
#    - env_visibility_bucket
#
# Why do this?
# If one particular feature is acting like a hidden shortcut, then removing it
# may cause performance to drop sharply. If performance changes only a little,
# that is more consistent with genuine but limited environmental signal rather
# than leakage.
#
# Important note:
# This step is a DIAGNOSTIC check, not a final model-building step.
# We are still working only with the train and validation sets.
# The test set remains untouched.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"

results_file = project_folder / "results" / "tables" / "step13_environment_only_leakage_results.csv"
feature_preview_file = project_folder / "results" / "tables" / "step13_environment_only_feature_preview.csv"
predictions_file = project_folder / "results" / "tables" / "step13_environment_only_full_model_predictions.csv"
confusion_matrix_file = project_folder / "results" / "tables" / "step13_environment_only_full_model_confusion_matrix.csv"


# -----------------------------------------------------------------------------
# Load the train and validation data
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 13: ENVIRONMENT-ONLY LEAKAGE DIAGNOSTIC")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Helper functions for engineered environmental features
# -----------------------------------------------------------------------------
def clean_string(x):
    """
    Convert raw values into clean strings while preserving missingness as the
    explicit category 'Missing'. This helps categorical modelling.
    """
    if pd.isna(x):
        return "Missing"
    x = str(x).strip()
    return x if x != "" else "Missing"


def extract_month_from_date(x):
    """
    Extract the month from the ASRS date field, which typically appears in
    YYYYMM format, for example:
    201901 -> month 01
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
    Convert a numeric month into a broad season category.
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
    Convert the raw ceiling field into a smaller number of categories.
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
    Extract the first numeric visibility value from the mixed-format
    weather/visibility field and bucket it into coarse groups.
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
    Return 'Yes' if any keyword appears in the raw weather text, otherwise 'No'.
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
    Build the engineered environmental feature table that will be used for the
    environment-only diagnostic.
    """
    work = df.copy()

    work["env_flight_conditions"] = work["Environment | Flight Conditions"].apply(clean_string)
    work["env_light"] = work["Environment | Light"].apply(clean_string)

    work["env_month_num"] = work["Time | Date"].apply(extract_month_from_date)
    work["env_month"] = work["env_month_num"].apply(
        lambda x: f"Month_{int(x):02d}" if x is not None else "Missing"
    )
    work["env_season"] = work["env_month_num"].apply(month_to_season)

    work["env_ceiling_bucket"] = work["Environment | Ceiling"].apply(bucket_ceiling)
    work["env_visibility_bucket"] = work["Environment | Weather Elements / Visibility"].apply(extract_visibility_bucket)

    weather_raw = work["Environment | Weather Elements / Visibility"]

    work["env_has_turbulence"] = weather_raw.apply(lambda x: contains_keyword(x, ["turbulence"]))
    work["env_has_windshear"] = weather_raw.apply(lambda x: contains_keyword(x, ["windshear"]))
    work["env_has_rain"] = weather_raw.apply(lambda x: contains_keyword(x, ["rain"]))
    work["env_has_icing"] = weather_raw.apply(lambda x: contains_keyword(x, ["icing"]))
    work["env_has_cloudy"] = weather_raw.apply(lambda x: contains_keyword(x, ["cloudy"]))
    work["env_has_fog"] = weather_raw.apply(lambda x: contains_keyword(x, ["fog"]))
    work["env_has_haze_smoke"] = weather_raw.apply(lambda x: contains_keyword(x, ["haze", "smoke"]))
    work["env_has_thunderstorm"] = weather_raw.apply(lambda x: contains_keyword(x, ["thunderstorm"]))
    work["env_has_snow"] = weather_raw.apply(lambda x: contains_keyword(x, ["snow"]))

    adverse_keywords = [
        "turbulence", "windshear", "rain", "icing", "cloudy",
        "fog", "haze", "smoke", "thunderstorm", "snow"
    ]
    work["env_any_adverse_weather"] = weather_raw.apply(lambda x: contains_keyword(x, adverse_keywords))

    return work


# -----------------------------------------------------------------------------
# Build engineered features for train and validation data
# -----------------------------------------------------------------------------
train_eng = build_engineered_environment_features(train_df)
valid_eng = build_engineered_environment_features(validation_df)

all_environment_features = [
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

# Save a small preview for inspection
preview_cols = ["ACN", "incident_year", "label_unstable"] + all_environment_features
train_eng[preview_cols].head(30).to_csv(feature_preview_file, index=False)

print("ENGINEERED ENVIRONMENTAL FEATURES AVAILABLE")
print("-" * 100)
for col in all_environment_features:
    print(f" - {col}")
print()


# -----------------------------------------------------------------------------
# Define target variable
# -----------------------------------------------------------------------------
y_train = train_eng["label_unstable"].astype(int)
y_valid = valid_eng["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Helper function to fit and evaluate an environment-only Logistic Regression
# -----------------------------------------------------------------------------
def evaluate_environment_model(feature_list, model_name_for_output=None):
    """
    Fit an environment-only Logistic Regression model using the supplied feature
    list and return a dictionary of evaluation metrics.
    """
    X_train = train_eng[feature_list].copy()
    X_valid = valid_eng[feature_list].copy()

    # One-hot encode the engineered categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("environment", OneHotEncoder(handle_unknown="ignore"), feature_list)
        ]
    )

    model = LogisticRegression(
        C=3.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    # Get probability scores from the fitted Logistic Regression model
    X_valid_transformed = pipeline.named_steps["preprocessor"].transform(X_valid)
    y_score = pipeline.named_steps["model"].predict_proba(X_valid_transformed)[:, 1]

    result = {
        "model_name": model_name_for_output if model_name_for_output else "environment_only_model",
        "feature_count": len(feature_list),
        "features_removed": "None",
        "precision": round(precision_score(y_valid, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_valid, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_valid, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_valid, y_score), 4),
        "pr_auc": round(average_precision_score(y_valid, y_score), 4),
    }

    return result, y_pred, y_score


# -----------------------------------------------------------------------------
# Run the full environment-only model first
# -----------------------------------------------------------------------------
results = []

full_result, full_y_pred, full_y_score = evaluate_environment_model(
    feature_list=all_environment_features,
    model_name_for_output="environment_only_full_feature_set"
)

results.append(full_result)

# Save predictions and confusion matrix for the full environment-only model
predictions_df = pd.DataFrame({
    "ACN": valid_eng["ACN"],
    "incident_year": valid_eng["incident_year"],
    "true_label": y_valid,
    "predicted_label": full_y_pred,
    "score": full_y_score
})
predictions_df.to_csv(predictions_file, index=False)

cm = confusion_matrix(y_valid, full_y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["actual_0_non_unstable", "actual_1_unstable"],
    columns=["predicted_0_non_unstable", "predicted_1_unstable"]
)
cm_df.to_csv(confusion_matrix_file)


# -----------------------------------------------------------------------------
# Ablation: remove suspicious features one by one
# -----------------------------------------------------------------------------
# These are the features we want to inspect more carefully because, if any of
# them behaves like a hidden shortcut, removing it may cause a sharp drop.
# -----------------------------------------------------------------------------
suspicious_features = [
    "env_any_adverse_weather",
    "env_has_windshear",
    "env_has_turbulence",
    "env_visibility_bucket",
]

for suspicious_feature in suspicious_features:
    reduced_features = [f for f in all_environment_features if f != suspicious_feature]

    ablation_result, _, _ = evaluate_environment_model(
        feature_list=reduced_features,
        model_name_for_output=f"environment_only_minus_{suspicious_feature}"
    )

    ablation_result["features_removed"] = suspicious_feature
    results.append(ablation_result)


# -----------------------------------------------------------------------------
# Save and print the diagnostic comparison table
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(results_file, index=False)

print("=" * 100)
print("ENVIRONMENT-ONLY LEAKAGE DIAGNOSTIC RESULTS")
print("=" * 100)
print(results_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(results_file.name)
print(feature_preview_file.name)
print(predictions_file.name)
print(confusion_matrix_file.name)
print()

print("=" * 100)
print("STEP 13 FINISHED")
print("This step evaluated an environment-only Logistic Regression model and")
print("tested whether removing suspicious environmental features caused a")
print("large performance collapse on the validation set.")
print("The test set has still NOT been used.")
print("=" * 100)