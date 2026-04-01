from pathlib import Path
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# =============================================================================
# STEP 17: COMPLETE-CASE ENVIRONMENTAL SENSITIVITY CHECK
# =============================================================================
# Purpose of this step:
# Earlier comparisons showed that adding environmental/context variables did not
# outperform the best text-only model. However, one possible explanation is that
# many environmental fields contain substantial missingness.
#
# To test this more fairly, we now create a "complete-case" subset from the main
# dataset only. This means we keep only rows where the chosen environmental
# fields are actually present for both unstable and non-unstable cases.
#
# We then compare two models on the SAME subset:
# 1. Text-only Logistic Regression
# 2. Text + engineered environmental Logistic Regression
#
# This is a sensitivity analysis. It does not replace the final selected model.
# It simply helps answer whether environment becomes more useful when we remove
# the bias introduced by missing environmental data.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"

results_file = project_folder / "results" / "tables" / "step17_complete_case_environment_comparison.csv"
subset_summary_file = project_folder / "results" / "tables" / "step17_complete_case_subset_summary.csv"


# -----------------------------------------------------------------------------
# Load train and validation data
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 17: COMPLETE-CASE ENVIRONMENTAL SENSITIVITY CHECK")
print("=" * 100)
print(f"Original training rows:   {len(train_df)}")
print(f"Original validation rows: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Define the core environmental fields for complete-case filtering
# -----------------------------------------------------------------------------
# We deliberately use the main environmental fields rather than every possible
# context field. The purpose is to test whether environment helps when the
# information is actually available.
# -----------------------------------------------------------------------------
core_env_cols = [
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
]

# Complete-case filter: keep rows where all four environmental fields are present
train_cc = train_df.dropna(subset=core_env_cols).copy()
valid_cc = validation_df.dropna(subset=core_env_cols).copy()

print("COMPLETE-CASE SUBSET CREATED")
print("-" * 100)
print(f"Training rows after complete-case filter:   {len(train_cc)}")
print(f"Validation rows after complete-case filter: {len(valid_cc)}")
print()


# -----------------------------------------------------------------------------
# Save subset summary so the reduction is documented
# -----------------------------------------------------------------------------
subset_summary = pd.DataFrame([
    {
        "dataset": "train_original",
        "rows": len(train_df),
        "unstable_cases": int(train_df["label_unstable"].sum()),
        "unstable_percent": round(train_df["label_unstable"].mean() * 100, 2),
    },
    {
        "dataset": "train_complete_case",
        "rows": len(train_cc),
        "unstable_cases": int(train_cc["label_unstable"].sum()),
        "unstable_percent": round(train_cc["label_unstable"].mean() * 100, 2),
    },
    {
        "dataset": "validation_original",
        "rows": len(validation_df),
        "unstable_cases": int(validation_df["label_unstable"].sum()),
        "unstable_percent": round(validation_df["label_unstable"].mean() * 100, 2),
    },
    {
        "dataset": "validation_complete_case",
        "rows": len(valid_cc),
        "unstable_cases": int(valid_cc["label_unstable"].sum()),
        "unstable_percent": round(valid_cc["label_unstable"].mean() * 100, 2),
    },
])

subset_summary.to_csv(subset_summary_file, index=False)

print("COMPLETE-CASE SUBSET SUMMARY")
print("-" * 100)
print(subset_summary.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Helper functions for engineered environmental features
# -----------------------------------------------------------------------------
def clean_string(x):
    if pd.isna(x):
        return "Missing"
    x = str(x).strip()
    return x if x != "" else "Missing"


def extract_month_from_date(x):
    if pd.isna(x):
        return None

    s = str(x).replace(".0", "").strip()

    if len(s) >= 6 and s[:6].isdigit():
        month_value = int(s[4:6])
        if 1 <= month_value <= 12:
            return month_value

    return None


def month_to_season(month_value):
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
    if pd.isna(raw_text):
        return "No"

    s = str(raw_text).lower()

    for kw in keyword_list:
        if kw in s:
            return "Yes"

    return "No"


def build_engineered_environment_features(df):
    work = df.copy()

    work["text_main"] = work["text_main"].fillna("").astype(str)

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
# Build engineered features on the complete-case subset
# -----------------------------------------------------------------------------
train_eng = build_engineered_environment_features(train_cc)
valid_eng = build_engineered_environment_features(valid_cc)

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

y_train = train_eng["label_unstable"].astype(int)
y_valid = valid_eng["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Model 1: text-only Logistic Regression on the complete-case subset
# -----------------------------------------------------------------------------
text_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3
)

X_train_text = train_eng["text_main"]
X_valid_text = valid_eng["text_main"]

X_train_text_tfidf = text_vectorizer.fit_transform(X_train_text)
X_valid_text_tfidf = text_vectorizer.transform(X_valid_text)

text_model = LogisticRegression(
    C=3.0,
    class_weight="balanced",
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

text_model.fit(X_train_text_tfidf, y_train)

y_pred_text = text_model.predict(X_valid_text_tfidf)
y_score_text = text_model.predict_proba(X_valid_text_tfidf)[:, 1]


# -----------------------------------------------------------------------------
# Model 2: text + engineered environment Logistic Regression on the same subset
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

text_plus_env_model = Pipeline([
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

text_plus_env_model.fit(X_train_env, y_train)

y_pred_env = text_plus_env_model.predict(X_valid_env)
X_valid_env_transformed = text_plus_env_model.named_steps["preprocessor"].transform(X_valid_env)
y_score_env = text_plus_env_model.named_steps["model"].predict_proba(X_valid_env_transformed)[:, 1]


# -----------------------------------------------------------------------------
# Helper function for metrics
# -----------------------------------------------------------------------------
def make_metric_row(model_name, y_true, y_pred, y_score):
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
# Save and print comparison
# -----------------------------------------------------------------------------
results_df = pd.DataFrame([
    make_metric_row("text_only_complete_case", y_valid, y_pred_text, y_score_text),
    make_metric_row("text_plus_engineered_environment_complete_case", y_valid, y_pred_env, y_score_env),
])

results_df.to_csv(results_file, index=False)

print("=" * 100)
print("COMPLETE-CASE ENVIRONMENTAL SENSITIVITY RESULTS")
print("=" * 100)
print(results_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(results_file.name)
print(subset_summary_file.name)
print()

print("=" * 100)
print("STEP 17 FINISHED")
print("This step compared text-only and text+engineered-environment models on the")
print("same complete-case subset from the main dataset only.")
print("No unstable-only raw files were used.")
print("=" * 100)