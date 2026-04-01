from pathlib import Path
import pandas as pd

from sklearn.model_selection import ParameterGrid
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
# STEP 9: SYSTEMATIC HYPERPARAMETER TUNING FOR TEXT-ONLY MODELS
# =============================================================================
# Purpose of this step:
# In Step 8, we created the first text-only baseline models using sensible
# default settings. That was useful because it gave us an initial benchmark.
# However, one of the major improvement points from the previous assessment was
# the lack of systematic hyperparameter tuning.
#
# This step addresses that weakness directly.
#
# We are still working with text only, so the input remains the 'text_main'
# narrative field. We are not adding environmental or operational variables yet.
# The purpose here is to improve the narrative-only models first before moving
# on to richer feature sets.
#
# What we are tuning:
# 1. TF-IDF settings:
#    - ngram_range
#    - min_df
#
# 2. Model-specific settings:
#    - Logistic Regression: C, class_weight
#    - Linear SVM: C, class_weight
#    - Multinomial Naive Bayes: alpha
#
# Why use the validation set here?
# Because the validation set is the correct place to compare candidate settings.
# The test set must remain untouched until the model-development stage is over.
#
# How we rank model settings:
# Since this is an imbalanced classification problem, we use PR-AUC as the main
# ranking metric. We then use F1, recall, and ROC-AUC as supporting metrics.
# This is more appropriate than using accuracy.
# =============================================================================


# -----------------------------------------------------------------------------
# Set up file paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "step7_train_dataset.csv"
validation_file = project_folder / "step7_validation_dataset.csv"

all_results_file = project_folder / "step9_text_only_tuning_results.csv"
best_settings_file = project_folder / "step9_best_text_only_settings.csv"

pred_lr_file = project_folder / "step9_best_validation_predictions_logistic_regression.csv"
pred_svm_file = project_folder / "step9_best_validation_predictions_linear_svm.csv"
pred_nb_file = project_folder / "step9_best_validation_predictions_multinomial_nb.csv"

cm_lr_file = project_folder / "step9_best_confusion_matrix_logistic_regression.csv"
cm_svm_file = project_folder / "step9_best_confusion_matrix_linear_svm.csv"
cm_nb_file = project_folder / "step9_best_confusion_matrix_multinomial_nb.csv"


# -----------------------------------------------------------------------------
# Load the train and validation datasets
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 9: SYSTEMATIC TUNING OF TEXT-ONLY MODELS")
print("=" * 100)
print(f"Training rows loaded:   {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()

# Use only the text column and label at this stage
X_train_text = train_df["text_main"].fillna("").astype(str)
y_train = train_df["label_unstable"].astype(int)

X_valid_text = validation_df["text_main"].fillna("").astype(str)
y_valid = validation_df["label_unstable"].astype(int)


# -----------------------------------------------------------------------------
# Define the tuning grids
# -----------------------------------------------------------------------------
# We keep the grids sensible and not excessively large.
# The goal is to show real, systematic tuning without making the process
# unnecessarily huge for a student project.
# -----------------------------------------------------------------------------

# Shared TF-IDF settings to test
tfidf_grid = list(ParameterGrid({
    "ngram_range": [(1, 1), (1, 2)],
    "min_df": [2, 3]
}))

# Logistic Regression tuning grid
lr_grid = list(ParameterGrid({
    "C": [0.1, 1.0, 3.0],
    "class_weight": [None, "balanced"]
}))

# Linear SVM tuning grid
svm_grid = list(ParameterGrid({
    "C": [0.1, 1.0, 3.0],
    "class_weight": [None, "balanced"]
}))

# Multinomial Naive Bayes tuning grid
nb_grid = list(ParameterGrid({
    "alpha": [0.1, 0.5, 1.0]
}))

print("TUNING GRID SIZES")
print("-" * 100)
print(f"TF-IDF combinations:           {len(tfidf_grid)}")
print(f"Logistic Regression settings:  {len(lr_grid)}")
print(f"Linear SVM settings:           {len(svm_grid)}")
print(f"Multinomial NB settings:       {len(nb_grid)}")
print()

total_combinations = (
    len(tfidf_grid) * len(lr_grid)
    + len(tfidf_grid) * len(svm_grid)
    + len(tfidf_grid) * len(nb_grid)
)

print(f"Total model runs to evaluate: {total_combinations}")
print()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_model(model_name, params):
    """
    Create the correct model object from a model name and parameter dictionary.
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
    elif model_name == "multinomial_nb":
        return MultinomialNB(
            alpha=params["alpha"]
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_candidate(model_name, model, X_train, y_train, X_valid, y_valid):
    """
    Fit a candidate model on the training set and evaluate it on the validation
    set using metrics appropriate for an imbalanced classification problem.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    # For AUC metrics we need a continuous score rather than just 0/1 labels.
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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_score": y_score
    }


def save_best_model_outputs(model_name, best_row, train_text, train_labels, valid_text, valid_labels, valid_source_df):
    """
    Rebuild the best TF-IDF vectorizer and the best model for a given model type,
    then save validation predictions and the confusion matrix.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=best_row["ngram_range"],
        min_df=int(best_row["min_df"])
    )

    X_train = vectorizer.fit_transform(train_text)
    X_valid = vectorizer.transform(valid_text)

    model_params = {}

    if model_name in ["logistic_regression", "linear_svm"]:
        model_params["C"] = float(best_row["C"])
        model_params["class_weight"] = best_row["class_weight"]
        if pd.isna(model_params["class_weight"]):
            model_params["class_weight"] = None
    elif model_name == "multinomial_nb":
        model_params["alpha"] = float(best_row["alpha"])

    model = get_model(model_name, model_params)
    eval_result = evaluate_candidate(
        model_name=model_name,
        model=model,
        X_train=X_train,
        y_train=train_labels,
        X_valid=X_valid,
        y_valid=valid_labels
    )

    predictions_df = pd.DataFrame({
        "ACN": valid_source_df["ACN"],
        "incident_year": valid_source_df["incident_year"],
        "true_label": valid_labels,
        "predicted_label": eval_result["y_pred"],
        "score": eval_result["y_score"]
    })

    cm = confusion_matrix(valid_labels, eval_result["y_pred"])
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0_non_unstable", "actual_1_unstable"],
        columns=["predicted_0_non_unstable", "predicted_1_unstable"]
    )

    if model_name == "logistic_regression":
        predictions_df.to_csv(pred_lr_file, index=False)
        cm_df.to_csv(cm_lr_file)
    elif model_name == "linear_svm":
        predictions_df.to_csv(pred_svm_file, index=False)
        cm_df.to_csv(cm_svm_file)
    elif model_name == "multinomial_nb":
        predictions_df.to_csv(pred_nb_file, index=False)
        cm_df.to_csv(cm_nb_file)


# -----------------------------------------------------------------------------
# Run the full tuning search
# -----------------------------------------------------------------------------
all_results = []
run_counter = 0

# Logistic Regression tuning
for tfidf_params in tfidf_grid:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=tfidf_params["ngram_range"],
        min_df=tfidf_params["min_df"]
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_valid_tfidf = vectorizer.transform(X_valid_text)

    for model_params in lr_grid:
        run_counter += 1
        print(f"[{run_counter}/{total_combinations}] logistic_regression | TF-IDF={tfidf_params} | Model={model_params}")

        model = get_model("logistic_regression", model_params)
        result = evaluate_candidate(
            model_name="logistic_regression",
            model=model,
            X_train=X_train_tfidf,
            y_train=y_train,
            X_valid=X_valid_tfidf,
            y_valid=y_valid
        )

        all_results.append({
            "model": "logistic_regression",
            "ngram_range": tfidf_params["ngram_range"],
            "min_df": tfidf_params["min_df"],
            "C": model_params["C"],
            "class_weight": model_params["class_weight"],
            "alpha": None,
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "f1": round(result["f1"], 4),
            "roc_auc": round(result["roc_auc"], 4),
            "pr_auc": round(result["pr_auc"], 4)
        })

# Linear SVM tuning
for tfidf_params in tfidf_grid:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=tfidf_params["ngram_range"],
        min_df=tfidf_params["min_df"]
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_valid_tfidf = vectorizer.transform(X_valid_text)

    for model_params in svm_grid:
        run_counter += 1
        print(f"[{run_counter}/{total_combinations}] linear_svm | TF-IDF={tfidf_params} | Model={model_params}")

        model = get_model("linear_svm", model_params)
        result = evaluate_candidate(
            model_name="linear_svm",
            model=model,
            X_train=X_train_tfidf,
            y_train=y_train,
            X_valid=X_valid_tfidf,
            y_valid=y_valid
        )

        all_results.append({
            "model": "linear_svm",
            "ngram_range": tfidf_params["ngram_range"],
            "min_df": tfidf_params["min_df"],
            "C": model_params["C"],
            "class_weight": model_params["class_weight"],
            "alpha": None,
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "f1": round(result["f1"], 4),
            "roc_auc": round(result["roc_auc"], 4),
            "pr_auc": round(result["pr_auc"], 4)
        })

# Multinomial Naive Bayes tuning
for tfidf_params in tfidf_grid:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=tfidf_params["ngram_range"],
        min_df=tfidf_params["min_df"]
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_valid_tfidf = vectorizer.transform(X_valid_text)

    for model_params in nb_grid:
        run_counter += 1
        print(f"[{run_counter}/{total_combinations}] multinomial_nb | TF-IDF={tfidf_params} | Model={model_params}")

        model = get_model("multinomial_nb", model_params)
        result = evaluate_candidate(
            model_name="multinomial_nb",
            model=model,
            X_train=X_train_tfidf,
            y_train=y_train,
            X_valid=X_valid_tfidf,
            y_valid=y_valid
        )

        all_results.append({
            "model": "multinomial_nb",
            "ngram_range": tfidf_params["ngram_range"],
            "min_df": tfidf_params["min_df"],
            "C": None,
            "class_weight": None,
            "alpha": model_params["alpha"],
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

# Sort globally for inspection
results_df = results_df.sort_values(
    by=["pr_auc", "f1", "recall", "roc_auc"],
    ascending=False
)

results_df.to_csv(all_results_file, index=False)


# -----------------------------------------------------------------------------
# Identify the best setting for each model
# -----------------------------------------------------------------------------
# We choose the best row within each model using:
# 1. PR-AUC (primary ranking metric)
# 2. F1
# 3. Recall
# 4. ROC-AUC
# -----------------------------------------------------------------------------
best_rows = []

for model_name in ["logistic_regression", "linear_svm", "multinomial_nb"]:
    model_results = results_df[results_df["model"] == model_name].copy()
    model_results = model_results.sort_values(
        by=["pr_auc", "f1", "recall", "roc_auc"],
        ascending=False
    )
    best_row = model_results.iloc[0]
    best_rows.append(best_row)

best_df = pd.DataFrame(best_rows)
best_df.to_csv(best_settings_file, index=False)


# -----------------------------------------------------------------------------
# Save predictions and confusion matrices for the best tuned version of each model
# -----------------------------------------------------------------------------
for _, row in best_df.iterrows():
    save_best_model_outputs(
        model_name=row["model"],
        best_row=row,
        train_text=X_train_text,
        train_labels=y_train,
        valid_text=X_valid_text,
        valid_labels=y_valid,
        valid_source_df=validation_df
    )


# -----------------------------------------------------------------------------
# Print the key outputs
# -----------------------------------------------------------------------------
print()
print("=" * 100)
print("BEST TUNED SETTING FOR EACH MODEL")
print("=" * 100)
print(best_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(all_results_file.name)
print(best_settings_file.name)
print(pred_lr_file.name)
print(pred_svm_file.name)
print(pred_nb_file.name)
print(cm_lr_file.name)
print(cm_svm_file.name)
print(cm_nb_file.name)
print()

print("=" * 100)
print("STEP 9 FINISHED")
print("Systematic tuning of the text-only models has been completed on the validation set.")
print("The test set has still NOT been used.")
print("=" * 100)