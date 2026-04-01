from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =============================================================================
# STEP 20: BASELINE CLASSIFIERS ON DOC2VEC VECTORS
# =============================================================================
# Purpose of this step:
# In Step 19, each narrative was converted into a dense 100-dimensional Doc2Vec
# representation. The present step now asks a straightforward modelling question:
#
# "If the narratives are represented as Doc2Vec document embeddings, how well do
# standard classifiers perform on the held-out validation set?"
#
# This step deliberately uses baseline classifier settings rather than tuned
# settings. The aim is to establish an initial Method 2 benchmark before any
# later tuning stage.
#
# Two classifiers are used:
# 1. Logistic Regression
# 2. Linear SVM
#
# These were chosen because:
# - both are standard discriminative classifiers
# - both work well with dense numeric vectors
# - both support imbalance-aware class weighting
#
# Important note:
# The test set is NOT used in this step.
# We fit on the training vectors and evaluate only on the validation vectors.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_vectors_file = project_folder / "data" / "processed" / "step19_doc2vec_train_vectors.csv"
validation_vectors_file = project_folder / "data" / "processed" / "step19_doc2vec_validation_vectors.csv"

metrics_output = project_folder / "results" / "tables" / "step20_doc2vec_validation_metrics.csv"

pred_lr_output = project_folder / "results" / "tables" / "step20_doc2vec_validation_predictions_logistic_regression.csv"
pred_svm_output = project_folder / "results" / "tables" / "step20_doc2vec_validation_predictions_linear_svm.csv"

cm_lr_output = project_folder / "results" / "tables" / "step20_doc2vec_confusion_matrix_logistic_regression.csv"
cm_svm_output = project_folder / "results" / "tables" / "step20_doc2vec_confusion_matrix_linear_svm.csv"


# -----------------------------------------------------------------------------
# Load the Doc2Vec vector files
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_vectors_file, low_memory=False)
validation_df = pd.read_csv(validation_vectors_file, low_memory=False)

print("=" * 100)
print("STEP 20: BASELINE CLASSIFIERS ON DOC2VEC VECTORS")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Define feature columns and target variable
# -----------------------------------------------------------------------------
# The vector columns are named:
# vec_0, vec_1, ..., vec_99
#
# We identify them automatically so the script remains robust.
# -----------------------------------------------------------------------------
vector_cols = [col for col in train_df.columns if col.startswith("vec_")]

X_train = train_df[vector_cols].copy()
y_train = train_df["label_unstable"].astype(int)

X_valid = validation_df[vector_cols].copy()
y_valid = validation_df["label_unstable"].astype(int)

print("DOC2VEC FEATURE SET")
print("-" * 100)
print(f"Number of vector features: {len(vector_cols)}")
print(f"Training feature matrix shape:   {X_train.shape}")
print(f"Validation feature matrix shape: {X_valid.shape}")
print()


# -----------------------------------------------------------------------------
# Define the baseline classifiers
# -----------------------------------------------------------------------------
# Logistic Regression:
# A simple linear decision boundary over the dense document vectors.
#
# Linear SVM:
# Another strong linear classifier, often competitive in text settings.
#
# We use class_weight='balanced' for both because the unstable class is smaller
# than the non-unstable class.
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
    )
}


# -----------------------------------------------------------------------------
# Helper function for evaluation
# -----------------------------------------------------------------------------
def evaluate_model(model_name, model, X_train, y_train, X_valid, y_valid, validation_source_df):
    """
    Fit one baseline classifier on the Doc2Vec training vectors and evaluate it
    on the validation vectors using imbalance-aware metrics.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    # Use probability or decision scores for ranking metrics
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_valid)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_valid)
    else:
        y_score = y_pred

    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_valid, y_score)
    pr_auc = average_precision_score(y_valid, y_score)

    predictions_df = pd.DataFrame({
        "ACN": validation_source_df["ACN"],
        "incident_year": validation_source_df["incident_year"],
        "true_label": y_valid,
        "predicted_label": y_pred,
        "score": y_score
    })

    cm = confusion_matrix(y_valid, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0_non_unstable", "actual_1_unstable"],
        columns=["predicted_0_non_unstable", "predicted_1_unstable"]
    )

    if model_name == "logistic_regression":
        predictions_df.to_csv(pred_lr_output, index=False)
        cm_df.to_csv(cm_lr_output)
    elif model_name == "linear_svm":
        predictions_df.to_csv(pred_svm_output, index=False)
        cm_df.to_csv(cm_svm_output)

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
# Train and evaluate both baseline classifiers
# -----------------------------------------------------------------------------
results = []

for model_name, model in models.items():
    print(f"Training and evaluating: {model_name}")

    result = evaluate_model(
        model_name=model_name,
        model=model,
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
results_df.to_csv(metrics_output, index=False)

print("=" * 100)
print("DOC2VEC VALIDATION METRICS SUMMARY")
print("=" * 100)
print(results_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(metrics_output.name)
print(pred_lr_output.name)
print(pred_svm_output.name)
print(cm_lr_output.name)
print(cm_svm_output.name)
print()

print("=" * 100)
print("STEP 20 FINISHED")
print("Baseline Logistic Regression and Linear SVM have been evaluated on the")
print("Doc2Vec validation vectors.")
print("No tuning has been done yet, and the test set has NOT been used.")
print("=" * 100)