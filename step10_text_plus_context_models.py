from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =============================================================================
# STEP 10: TEXT + SAFE CONTEXT MODELS ON THE VALIDATION SET
# =============================================================================
# Purpose of this step:
# In Step 9, we established the best text-only versions of the three candidate
# models using the narrative text alone. That gave us a strong baseline.
#
# The aim of the present step is to test Research Question 3:
# "Does incorporating environmental and operational context improve the
# classification of unstable-approach events beyond what can be achieved from
# narrative text alone?"
#
# We therefore keep the same train/validation split, but now extend the input
# space so that each model can use:
# - the main narrative text ('text_main')
# - a set of safe contextual variables
#
# Important methodological principles in this step:
# 1. We still do NOT use the test set.
# 2. We still avoid leakage-prone variables such as the anomaly field.
# 3. We handle missing context values explicitly by converting them to the
#    category label "Missing", rather than dropping rows.
# 4. We use the same core model families as before so that the comparison
#    remains fair and interpretable.
#
# Why use one-hot encoding for context?
# Most of the selected context variables are categorical rather than continuous.
# One-hot encoding converts each category into a machine-readable sparse
# representation while preserving the original category information.
#
# Why combine TF-IDF with one-hot encoded context?
# This lets the model learn jointly from:
# - what is said in the narrative
# - and the broader environmental/operational setting in which the event
#   occurred.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"

metrics_file = project_folder / "results" / "tables" / "step10_text_plus_context_validation_metrics.csv"

pred_lr_file = project_folder / "results" / "tables" / "step10_validation_predictions_logistic_regression.csv"
pred_svm_file = project_folder / "results" / "tables" / "step10_validation_predictions_linear_svm.csv"
pred_nb_file = project_folder / "results" / "tables" / "step10_validation_predictions_multinomial_nb.csv"

cm_lr_file = project_folder / "results" / "tables" / "step10_confusion_matrix_logistic_regression.csv"
cm_svm_file = project_folder / "results" / "tables" / "step10_confusion_matrix_linear_svm.csv"
cm_nb_file = project_folder / "results" / "tables" / "step10_confusion_matrix_multinomial_nb.csv"


# -----------------------------------------------------------------------------
# Load the train and validation datasets
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 10: TEXT + SAFE CONTEXT MODELS")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Define the target variable
# -----------------------------------------------------------------------------
y_train = train_df["label_unstable"].astype(int)
y_valid = validation_df["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Define the context columns to include
# -----------------------------------------------------------------------------
# These are "safe" context columns in the sense that they describe the
# operational setting but should not directly leak the label itself.
# -----------------------------------------------------------------------------
context_columns = [
    "Aircraft 1 | Flight Phase",
    "Environment | Flight Conditions",
    "Environment | Weather Elements / Visibility",
    "Environment | Light",
    "Environment | Ceiling",
    "Aircraft 1 | Make Model Name",
    "Aircraft 1 | Flight Plan",
    "Aircraft 1 | Mission",
    "Aircraft 1 | Aircraft Operator",
    "Aircraft 1 | Operating Under FAR Part",
    "Aircraft 1 | ATC / Advisory",
]

# We prepare copies of the train/validation tables so that we can safely fill
# missing values without modifying the original imported dataframes in place.
train_work = train_df.copy()
valid_work = validation_df.copy()

# -----------------------------------------------------------------------------
# Prepare the text and context fields
# -----------------------------------------------------------------------------
# Text:
# We ensure the narrative text is a non-missing string.
#
# Context:
# We convert missing values into the explicit string "Missing". This allows the
# one-hot encoder to treat missingness as an observed category rather than
# silently dropping those rows.
# -----------------------------------------------------------------------------
train_work["text_main"] = train_work["text_main"].fillna("").astype(str)
valid_work["text_main"] = valid_work["text_main"].fillna("").astype(str)

for col in context_columns:
    train_work[col] = train_work[col].fillna("Missing").astype(str).str.strip()
    valid_work[col] = valid_work[col].fillna("Missing").astype(str).str.strip()

# The modelling dataframe now consists of:
# - one text column
# - multiple context columns
feature_columns = ["text_main"] + context_columns

X_train = train_work[feature_columns]
X_valid = valid_work[feature_columns]

print("FEATURE SET USED IN STEP 10")
print("-" * 100)
print("Text column:")
print(" - text_main")
print("\nContext columns:")
for col in context_columns:
    print(f" - {col}")
print()


# -----------------------------------------------------------------------------
# Build the preprocessing transformer
# -----------------------------------------------------------------------------
# The ColumnTransformer applies different feature engineering steps to different
# types of input:
#
# 1. text_main -> TF-IDF
#    We use the best-performing text settings identified in Step 9:
#    - ngram_range = (1, 2)
#    - min_df = 3
#
# 2. context columns -> OneHotEncoder
#    This converts each category into a sparse indicator representation.
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
            "context",
            OneHotEncoder(handle_unknown="ignore"),
            context_columns
        )
    ]
)


# -----------------------------------------------------------------------------
# Define the three text + context models
# -----------------------------------------------------------------------------
# To keep the comparison fair, we use the best model-specific settings found in
# Step 9 for each model family.
# -----------------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(
        C=3.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ),
    "linear_svm": LinearSVC(
        C=0.1,
        class_weight=None,
        random_state=42
    ),
    "multinomial_nb": MultinomialNB(
        alpha=0.1
    )
}


# -----------------------------------------------------------------------------
# Helper function to fit, predict, evaluate, and save results
# -----------------------------------------------------------------------------
def evaluate_pipeline(model_name, pipeline, X_train, y_train, X_valid, y_valid, validation_source_df):
    """
    Fit a full pipeline (preprocessing + model) on the training set and evaluate
    it on the validation set using imbalance-aware metrics.
    """
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    # Obtain a continuous score for ROC-AUC and PR-AUC.
    model = pipeline.named_steps["model"]
    X_valid_transformed = pipeline.named_steps["preprocessor"].transform(X_valid)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_valid_transformed)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_valid_transformed)
    else:
        y_score = y_pred

    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_valid, y_score)
    pr_auc = average_precision_score(y_valid, y_score)

    # Save validation predictions for later comparison and error analysis
    predictions_df = pd.DataFrame({
        "ACN": validation_source_df["ACN"],
        "incident_year": validation_source_df["incident_year"],
        "true_label": y_valid,
        "predicted_label": y_pred,
        "score": y_score
    })

    # Save to the correct file
    if model_name == "logistic_regression":
        predictions_df.to_csv(pred_lr_file, index=False)
        cm_output_file = cm_lr_file
    elif model_name == "linear_svm":
        predictions_df.to_csv(pred_svm_file, index=False)
        cm_output_file = cm_svm_file
    elif model_name == "multinomial_nb":
        predictions_df.to_csv(pred_nb_file, index=False)
        cm_output_file = cm_nb_file
    else:
        cm_output_file = project_folder / "results" / "tables" / f"step10_confusion_matrix_{model_name}.csv"

    cm = confusion_matrix(y_valid, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0_non_unstable", "actual_1_unstable"],
        columns=["predicted_0_non_unstable", "predicted_1_unstable"]
    )
    cm_df.to_csv(cm_output_file)

    return {
        "model": model_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "validation_rows": len(y_valid),
        "validation_unstable_cases": int(y_valid.sum())
    }


# -----------------------------------------------------------------------------
# Train and evaluate the three text + context pipelines
# -----------------------------------------------------------------------------
results = []

for model_name, model in models.items():
    print(f"Training and evaluating text + context model: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    result = evaluate_pipeline(
        model_name=model_name,
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        validation_source_df=validation_df
    )

    results.append(result)

print()


# -----------------------------------------------------------------------------
# Save and print the validation metrics summary
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["pr_auc", "f1", "roc_auc"], ascending=False)
results_df.to_csv(metrics_file, index=False)

print("=" * 100)
print("TEXT + CONTEXT VALIDATION METRICS SUMMARY")
print("=" * 100)
print(results_df.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Final output summary
# -----------------------------------------------------------------------------
print("FILES CREATED")
print("-" * 100)
print(metrics_file.name)
print(pred_lr_file.name)
print(pred_svm_file.name)
print(pred_nb_file.name)
print(cm_lr_file.name)
print(cm_svm_file.name)
print(cm_nb_file.name)
print()

print("=" * 100)
print("STEP 10 FINISHED")
print("The text + context models have been trained and evaluated on the validation set.")
print("The test set has NOT been used yet.")
print("=" * 100)