from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
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
# STEP 8: TEXT-ONLY BASELINE MODELS ON THE VALIDATION SET
# =============================================================================
# Purpose of this step:
# We now begin the first modelling stage of the project. However, this is still
# a deliberately simple and controlled step. We are not tuning hyperparameters
# yet, we are not using context variables yet, and we are not touching the final
# test set yet.
#
# The goal here is to establish a clean text-only baseline using the narrative
# field. This helps us answer an important early question:
#
# "If we only use the aviation narratives, which standard text-classification
# model performs best on a held-out future validation year?"
#
# The three baseline models compared here are:
# 1. Logistic Regression
# 2. Linear Support Vector Machine (Linear SVM)
# 3. Multinomial Naive Bayes
#
# Why these three models?
# - They are standard and well-established text-classification baselines.
# - They work well with sparse TF-IDF text features.
# - They provide a strong benchmark before we move to tuning and context models.
#
# Important methodological note:
# We are using ONLY the training set to fit the TF-IDF vectorizer.
# We then transform the validation set using that same fitted vectorizer.
# This is essential because fitting the vectorizer on the validation data would
# leak information from the future set into the training process.
#
# Also, because the dataset is imbalanced, we focus on metrics such as:
# - Precision
# - Recall
# - F1-score
# - ROC-AUC
# - PR-AUC (average precision)
#
# We do NOT use plain accuracy as the main headline metric because, in an
# imbalanced setting, accuracy can be misleading.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
# Using the script's own folder makes the workflow more reliable and avoids
# path errors if the terminal was opened from a different location.
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "step7_train_dataset.csv"
validation_file = project_folder / "step7_validation_dataset.csv"

metrics_file = project_folder / "step8_text_only_validation_metrics.csv"

pred_lr_file = project_folder / "step8_validation_predictions_logistic_regression.csv"
pred_svm_file = project_folder / "step8_validation_predictions_linear_svm.csv"
pred_nb_file = project_folder / "step8_validation_predictions_multinomial_nb.csv"

cm_lr_file = project_folder / "step8_confusion_matrix_logistic_regression.csv"
cm_svm_file = project_folder / "step8_confusion_matrix_linear_svm.csv"
cm_nb_file = project_folder / "step8_confusion_matrix_multinomial_nb.csv"


# -----------------------------------------------------------------------------
# Load the train and validation datasets
# -----------------------------------------------------------------------------
# At this stage, we use:
# - train set: 2018-2023
# - validation set: 2024
#
# The test set from 2025 is intentionally left untouched for now.
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 8: TEXT-ONLY BASELINE MODELS")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Define the text input and target variable
# -----------------------------------------------------------------------------
# We use only the 'text_main' column in this step because this is the text-only
# baseline stage. The target variable is the binary unstable-approach label.
# -----------------------------------------------------------------------------
X_train_text = train_df["text_main"].fillna("").astype(str)
y_train = train_df["label_unstable"].astype(int)

X_valid_text = validation_df["text_main"].fillna("").astype(str)
y_valid = validation_df["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Convert the narrative text into TF-IDF features
# -----------------------------------------------------------------------------
# TF-IDF (Term Frequency-Inverse Document Frequency) converts each narrative into
# a numerical representation based on the words it contains.
#
# Why use TF-IDF here?
# - It is a strong classical baseline for text classification.
# - It gives higher weight to informative words and lower weight to very common
#   words that appear everywhere.
# - It works well with models such as Logistic Regression, Linear SVM, and NB.
#
# Baseline settings used here:
# - lowercase=True: standardise text by ignoring case differences
# - stop_words="english": remove very common English filler words
# - ngram_range=(1, 2): use both single words and two-word phrases
# - min_df=3: ignore extremely rare terms that appear in fewer than 3 documents
#
# These are sensible baseline settings, not tuned final settings.
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3
)

# Fit ONLY on training text
X_train_tfidf = vectorizer.fit_transform(X_train_text)

# Transform validation text using the vocabulary learned from training only
X_valid_tfidf = vectorizer.transform(X_valid_text)

print("TF-IDF VECTORIZATION COMPLETED")
print("-" * 100)
print(f"Training TF-IDF shape:   {X_train_tfidf.shape}")
print(f"Validation TF-IDF shape: {X_valid_tfidf.shape}")
print()


# -----------------------------------------------------------------------------
# Define the three baseline models
# -----------------------------------------------------------------------------
# We use class_weight='balanced' for Logistic Regression and Linear SVM because
# the positive unstable-approach class is much smaller than the negative class.
# This is a simple baseline imbalance adjustment.
#
# For MultinomialNB, class_weight is not available in the same way, so we use
# the standard version here as a benchmark.
# -----------------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ),
    "linear_svm": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    ),
    "multinomial_nb": MultinomialNB(
        alpha=1.0
    )
}


# -----------------------------------------------------------------------------
# Helper function to evaluate a model on the validation set
# -----------------------------------------------------------------------------
# We calculate:
# - precision
# - recall
# - F1
# - ROC-AUC
# - PR-AUC
#
# We also save:
# - predictions for later error analysis
# - confusion matrices for later reporting
#
# For ROC-AUC and PR-AUC, we need a score rather than just a hard class label.
# Different models provide different score functions:
# - Logistic Regression -> predict_proba
# - LinearSVC -> decision_function
# - MultinomialNB -> predict_proba
# -----------------------------------------------------------------------------
def evaluate_model(model_name, model, X_train, y_train, X_valid, y_valid, validation_source_df):
    # Fit the model on the training data only
    model.fit(X_train, y_train)

    # Generate hard class predictions for the validation set
    y_pred = model.predict(X_valid)

    # Generate continuous scores for AUC-based metrics
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_valid)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_valid)
    else:
        # In case a future model has neither probability nor decision scores
        y_score = y_pred

    # Calculate evaluation metrics suited to class imbalance
    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_valid, y_score)
    pr_auc = average_precision_score(y_valid, y_score)

    # Create confusion matrix
    cm = confusion_matrix(y_valid, y_pred)

    # Save validation predictions for later threshold analysis and error analysis
    predictions_df = pd.DataFrame({
        "ACN": validation_source_df["ACN"],
        "incident_year": validation_source_df["incident_year"],
        "true_label": y_valid,
        "predicted_label": y_pred,
        "score": y_score
    })

    # Save predictions to the correct file
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
        cm_output_file = project_folder / f"step8_confusion_matrix_{model_name}.csv"

    # Save confusion matrix in a readable table
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0_non_unstable", "actual_1_unstable"],
        columns=["predicted_0_non_unstable", "predicted_1_unstable"]
    )
    cm_df.to_csv(cm_output_file)

    # Return metrics as one row
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
# Train and evaluate all three baseline models
# -----------------------------------------------------------------------------
results = []

for model_name, model in models.items():
    print(f"Training and evaluating: {model_name}")
    result = evaluate_model(
        model_name=model_name,
        model=model,
        X_train=X_train_tfidf,
        y_train=y_train,
        X_valid=X_valid_tfidf,
        y_valid=y_valid,
        validation_source_df=validation_df
    )
    results.append(result)

print()


# -----------------------------------------------------------------------------
# Save and print the validation metrics summary
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)

# Sort by PR-AUC first, because this is often very informative in imbalanced
# classification problems. We also keep ROC-AUC, precision, recall, and F1.
results_df = results_df.sort_values(by=["pr_auc", "f1", "roc_auc"], ascending=False)

results_df.to_csv(metrics_file, index=False)

print("=" * 100)
print("VALIDATION METRICS SUMMARY")
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
print("STEP 8 FINISHED")
print("The text-only baseline models have been trained and evaluated on the validation set.")
print("The test set has NOT been used yet.")
print("=" * 100)