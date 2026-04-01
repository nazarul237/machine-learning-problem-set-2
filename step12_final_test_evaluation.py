from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
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
# STEP 12: FINAL MODEL SELECTION AND UNTOUCHED TEST-SET EVALUATION
# =============================================================================
# Purpose of this step:
# Up to this point, the project has followed a disciplined sequence:
# - train set used for model fitting
# - validation set used for model comparison and tuning
# - test set kept untouched
#
# Based on the previous steps, the best overall candidate model is:
# - text-only Logistic Regression
# - TF-IDF with ngram_range = (1, 2)
# - min_df = 3
# - class_weight = balanced
# - C = 3.0
#
# In this step, we now perform the final evaluation properly.
#
# Methodological logic:
# 1. Since model selection is finished, we combine the train and validation sets
#    into one larger development dataset.
# 2. We refit the chosen final model on that combined development data.
# 3. We evaluate the model once on the untouched 2025 test set.
#
# This preserves the integrity of the test set while allowing the final model
# to learn from as much pre-test information as possible.
#
# Important note:
# This step is not tuning the model anymore.
# It is the final locked evaluation of the chosen model.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "splits" / "step7_train_dataset.csv"
validation_file = project_folder / "data" / "splits" / "step7_validation_dataset.csv"
test_file = project_folder / "data" / "splits" / "step7_test_dataset.csv"

metrics_file = project_folder / "results" / "tables" / "step12_final_test_metrics.csv"
predictions_file = project_folder / "results" / "tables" / "step12_final_test_predictions.csv"
confusion_matrix_file = project_folder / "results" / "tables" / "step12_final_test_confusion_matrix.csv"


# -----------------------------------------------------------------------------
# Load the train, validation, and test datasets
# -----------------------------------------------------------------------------
# We load all three files here, but only because model selection is already over.
# The train and validation sets will be combined into one development set.
# The test set remains the final untouched evaluation target.
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)

print("=" * 100)
print("STEP 12: FINAL TEST-SET EVALUATION")
print("=" * 100)
print(f"Train rows loaded:      {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print(f"Test rows loaded:       {len(test_df)}")
print()


# -----------------------------------------------------------------------------
# Combine train and validation into one development dataset
# -----------------------------------------------------------------------------
# This is now appropriate because the validation set has already done its job:
# it helped us choose the final model and parameter settings.
#
# By combining train + validation, the final model gets access to more labelled
# pre-test data before being evaluated on the untouched 2025 set.
# -----------------------------------------------------------------------------
development_df = pd.concat([train_df, validation_df], ignore_index=True)

print("DEVELOPMENT DATASET CREATED")
print("-" * 100)
print(f"Rows in development set (train + validation): {len(development_df)}")
print()


# -----------------------------------------------------------------------------
# Define the text input and target variable
# -----------------------------------------------------------------------------
# We use only the text_main narrative field because the final selected model is
# the best text-only Logistic Regression identified in earlier steps.
# -----------------------------------------------------------------------------
X_dev_text = development_df["text_main"].fillna("").astype(str)
y_dev = development_df["label_unstable"].astype(int)

X_test_text = test_df["text_main"].fillna("").astype(str)
y_test = test_df["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Rebuild the final selected TF-IDF representation
# -----------------------------------------------------------------------------
# We use the exact best settings found in Step 9:
# - ngram_range = (1, 2)
# - min_df = 3
# - stop_words = english
#
# We fit the vectorizer only on the development data, then transform the test
# set using that learned vocabulary.
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3
)

X_dev_tfidf = vectorizer.fit_transform(X_dev_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print("TF-IDF VECTORIZATION COMPLETED")
print("-" * 100)
print(f"Development TF-IDF shape: {X_dev_tfidf.shape}")
print(f"Test TF-IDF shape:        {X_test_tfidf.shape}")
print()


# -----------------------------------------------------------------------------
# Rebuild and fit the final selected Logistic Regression model
# -----------------------------------------------------------------------------
# We use the exact best-performing settings identified earlier.
# This is the final locked model specification.
# -----------------------------------------------------------------------------
final_model = LogisticRegression(
    C=3.0,
    class_weight="balanced",
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

final_model.fit(X_dev_tfidf, y_dev)


# -----------------------------------------------------------------------------
# Generate final test predictions and scores
# -----------------------------------------------------------------------------
# We use:
# - hard labels for precision / recall / F1 / confusion matrix
# - probability scores for ROC-AUC and PR-AUC
# -----------------------------------------------------------------------------
y_test_pred = final_model.predict(X_test_tfidf)
y_test_score = final_model.predict_proba(X_test_tfidf)[:, 1]


# -----------------------------------------------------------------------------
# Compute the final test metrics
# -----------------------------------------------------------------------------
# These are imbalance-aware metrics and are therefore more appropriate than
# plain accuracy for the present unstable-approach classification problem.
# -----------------------------------------------------------------------------
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_score)
pr_auc = average_precision_score(y_test, y_test_score)

metrics_df = pd.DataFrame([{
    "model": "final_text_only_logistic_regression",
    "development_rows": len(development_df),
    "test_rows": len(test_df),
    "test_unstable_cases": int(y_test.sum()),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4)
}])

metrics_df.to_csv(metrics_file, index=False)


# -----------------------------------------------------------------------------
# Save the final test predictions
# -----------------------------------------------------------------------------
# These predictions will be useful later for:
# - error analysis
# - threshold discussion
# - inspecting false positives and false negatives
# -----------------------------------------------------------------------------
predictions_df = pd.DataFrame({
    "ACN": test_df["ACN"],
    "incident_year": test_df["incident_year"],
    "true_label": y_test,
    "predicted_label": y_test_pred,
    "score": y_test_score
})

predictions_df.to_csv(predictions_file, index=False)


# -----------------------------------------------------------------------------
# Save the final confusion matrix
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)

cm_df = pd.DataFrame(
    cm,
    index=["actual_0_non_unstable", "actual_1_unstable"],
    columns=["predicted_0_non_unstable", "predicted_1_unstable"]
)

cm_df.to_csv(confusion_matrix_file)


# -----------------------------------------------------------------------------
# Print final results
# -----------------------------------------------------------------------------
print("=" * 100)
print("FINAL TEST METRICS")
print("=" * 100)
print(metrics_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(metrics_file.name)
print(predictions_file.name)
print(confusion_matrix_file.name)
print()

print("=" * 100)
print("STEP 12 FINISHED")
print("The final selected model has now been evaluated on the untouched test set.")
print("No further tuning should be done using the test results.")
print("=" * 100)