from pathlib import Path
import pandas as pd

from sklearn.model_selection import ParameterGrid
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
# STEP 21: TUNE THE DOC2VEC-BASED CLASSIFIERS
# =============================================================================
# Purpose of this step:
# In Step 20, baseline Logistic Regression and Linear SVM models were trained on
# the Doc2Vec document vectors. Those baseline results showed that Method 2 is
# workable, but not yet competitive enough with the best TF-IDF model.
#
# The purpose of the present step is therefore to tune the classifier layer on
# top of the fixed Doc2Vec vectors.
#
# Important methodological note:
# - The Doc2Vec vectors themselves were learned in Step 19.
# - In this step, those vectors are treated as fixed inputs.
# - We tune only the classifiers that sit on top of those vectors.
# - The validation set is used for model comparison.
# - The test set remains untouched.
#
# Two model families are tuned:
# 1. Logistic Regression
# 2. Linear SVM
#
# The ranking metric used here is PR-AUC first, followed by F1 and ROC-AUC,
# because this remains an imbalanced classification problem.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_vectors_file = project_folder / "data" / "processed" / "step19_doc2vec_train_vectors.csv"
validation_vectors_file = project_folder / "data" / "processed" / "step19_doc2vec_validation_vectors.csv"

all_results_output = project_folder / "results" / "tables" / "step21_doc2vec_tuning_results.csv"
best_settings_output = project_folder / "results" / "tables" / "step21_doc2vec_best_settings.csv"

pred_lr_output = project_folder / "results" / "tables" / "step21_doc2vec_best_validation_predictions_logistic_regression.csv"
pred_svm_output = project_folder / "results" / "tables" / "step21_doc2vec_best_validation_predictions_linear_svm.csv"

cm_lr_output = project_folder / "results" / "tables" / "step21_doc2vec_best_confusion_matrix_logistic_regression.csv"
cm_svm_output = project_folder / "results" / "tables" / "step21_doc2vec_best_confusion_matrix_linear_svm.csv"


# -----------------------------------------------------------------------------
# Load Doc2Vec train and validation vectors
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_vectors_file, low_memory=False)
validation_df = pd.read_csv(validation_vectors_file, low_memory=False)

print("=" * 100)
print("STEP 21: TUNE THE DOC2VEC-BASED CLASSIFIERS")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()

# Identify the vector columns automatically
vector_cols = [col for col in train_df.columns if col.startswith("vec_")]

X_train = train_df[vector_cols].copy()
y_train = train_df["label_unstable"].astype(int)

X_valid = validation_df[vector_cols].copy()
y_valid = validation_df["label_unstable"].astype(int)

print("DOC2VEC FEATURE MATRIX")
print("-" * 100)
print(f"Number of vector features: {len(vector_cols)}")
print(f"Training shape:   {X_train.shape}")
print(f"Validation shape: {X_valid.shape}")
print()


# -----------------------------------------------------------------------------
# Define tuning grids
# -----------------------------------------------------------------------------
# The grids are intentionally moderate in size.
# The aim is to show real systematic tuning without making the workflow
# unnecessarily large or difficult to interpret.
# -----------------------------------------------------------------------------
lr_grid = list(ParameterGrid({
    "C": [0.1, 1.0, 3.0, 10.0],
    "class_weight": [None, "balanced"]
}))

svm_grid = list(ParameterGrid({
    "C": [0.1, 1.0, 3.0, 10.0],
    "class_weight": [None, "balanced"]
}))

print("TUNING GRID SIZES")
print("-" * 100)
print(f"Logistic Regression settings: {len(lr_grid)}")
print(f"Linear SVM settings:          {len(svm_grid)}")
print(f"Total model runs:             {len(lr_grid) + len(svm_grid)}")
print()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_model(model_name, params):
    """
    Build the correct classifier from the model name and parameter dictionary.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=2000,
            solver="liblinear",
            random_state=42
        )
    elif model_name == "linear_svm":
        return LinearSVC(
            C=params["C"],
            class_weight=params["class_weight"],
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_candidate(model_name, model, X_train, y_train, X_valid, y_valid):
    """
    Fit one candidate model and evaluate it on the validation set using
    imbalance-aware metrics.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_valid)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_valid)
    else:
        y_score = y_pred

    return {
        "precision": precision_score(y_valid, y_pred, zero_division=0),
        "recall": recall_score(y_valid, y_pred, zero_division=0),
        "f1": f1_score(y_valid, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, y_score),
        "pr_auc": average_precision_score(y_valid, y_score),
        "y_pred": y_pred,
        "y_score": y_score
    }


def save_best_model_outputs(model_name, best_row, X_train, y_train, X_valid, y_valid, validation_source_df):
    """
    Refit the best-tuned model and save validation predictions and confusion
    matrices for later analysis and reporting.
    """
    params = {
        "C": float(best_row["C"]),
        "class_weight": best_row["class_weight"]
    }

    if pd.isna(params["class_weight"]):
        params["class_weight"] = None

    model = get_model(model_name, params)
    result = evaluate_candidate(model_name, model, X_train, y_train, X_valid, y_valid)

    predictions_df = pd.DataFrame({
        "ACN": validation_source_df["ACN"],
        "incident_year": validation_source_df["incident_year"],
        "true_label": y_valid,
        "predicted_label": result["y_pred"],
        "score": result["y_score"]
    })

    cm = confusion_matrix(y_valid, result["y_pred"])
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


# -----------------------------------------------------------------------------
# Run tuning search
# -----------------------------------------------------------------------------
all_results = []
run_counter = 0
total_runs = len(lr_grid) + len(svm_grid)

# Logistic Regression tuning
for params in lr_grid:
    run_counter += 1
    print(f"[{run_counter}/{total_runs}] logistic_regression | params={params}")

    model = get_model("logistic_regression", params)
    result = evaluate_candidate("logistic_regression", model, X_train, y_train, X_valid, y_valid)

    all_results.append({
        "model": "logistic_regression",
        "C": params["C"],
        "class_weight": params["class_weight"],
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "f1": round(result["f1"], 4),
        "roc_auc": round(result["roc_auc"], 4),
        "pr_auc": round(result["pr_auc"], 4)
    })

# Linear SVM tuning
for params in svm_grid:
    run_counter += 1
    print(f"[{run_counter}/{total_runs}] linear_svm | params={params}")

    model = get_model("linear_svm", params)
    result = evaluate_candidate("linear_svm", model, X_train, y_train, X_valid, y_valid)

    all_results.append({
        "model": "linear_svm",
        "C": params["C"],
        "class_weight": params["class_weight"],
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "f1": round(result["f1"], 4),
        "roc_auc": round(result["roc_auc"], 4),
        "pr_auc": round(result["pr_auc"], 4)
    })


# -----------------------------------------------------------------------------
# Save all tuning results
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=["pr_auc", "f1", "roc_auc"], ascending=False)
results_df.to_csv(all_results_output, index=False)


# -----------------------------------------------------------------------------
# Select the best row for each model family
# -----------------------------------------------------------------------------
best_rows = []

for model_name in ["logistic_regression", "linear_svm"]:
    model_results = results_df[results_df["model"] == model_name].copy()
    model_results = model_results.sort_values(
        by=["pr_auc", "f1", "roc_auc"],
        ascending=False
    )
    best_rows.append(model_results.iloc[0])

best_df = pd.DataFrame(best_rows)
best_df.to_csv(best_settings_output, index=False)


# -----------------------------------------------------------------------------
# Save predictions and confusion matrices for the best tuned versions
# -----------------------------------------------------------------------------
for _, row in best_df.iterrows():
    save_best_model_outputs(
        model_name=row["model"],
        best_row=row,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        validation_source_df=validation_df
    )


# -----------------------------------------------------------------------------
# Print final outputs
# -----------------------------------------------------------------------------
print()
print("=" * 100)
print("BEST TUNED SETTING FOR EACH DOC2VEC-BASED MODEL")
print("=" * 100)
print(best_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(all_results_output.name)
print(best_settings_output.name)
print(pred_lr_output.name)
print(pred_svm_output.name)
print(cm_lr_output.name)
print(cm_svm_output.name)
print()

print("=" * 100)
print("STEP 21 FINISHED")
print("The Doc2Vec-based Logistic Regression and Linear SVM models have been tuned")
print("on the validation set. The test set has NOT been used.")
print("=" * 100)