from pathlib import Path
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
# STEP 22: FINAL DOC2VEC TEST EVALUATION
# =============================================================================
# Purpose of this step:
# The previous steps established the best Method 2 candidate as:
# - Doc2Vec document representation
# - Logistic Regression classifier
# - C = 10.0
# - class_weight = balanced
#
# In this final Method 2 evaluation, the model-selection stage is complete.
# Therefore, the train and validation data can now be combined into one larger
# development dataset. The Doc2Vec representation is retrained on that combined
# pre-test corpus, and the selected Logistic Regression model is then evaluated
# once on the untouched 2025 test set.
#
# This mirrors the same logic used earlier for the final Method 1 evaluation.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_input = project_folder / "step18_doc2vec_train_prepared.csv"
validation_input = project_folder / "step18_doc2vec_validation_prepared.csv"
test_input = project_folder / "step18_doc2vec_test_prepared.csv"

model_output = project_folder / "step22_doc2vec_final_model.model"
metrics_output = project_folder / "step22_doc2vec_final_test_metrics.csv"
predictions_output = project_folder / "step22_doc2vec_final_test_predictions.csv"
cm_output = project_folder / "step22_doc2vec_final_test_confusion_matrix.csv"


# -----------------------------------------------------------------------------
# Load prepared text files
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_input, low_memory=False)
validation_df = pd.read_csv(validation_input, low_memory=False)
test_df = pd.read_csv(test_input, low_memory=False)

print("=" * 100)
print("STEP 22: FINAL DOC2VEC TEST EVALUATION")
print("=" * 100)
print(f"Train rows loaded:      {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print(f"Test rows loaded:       {len(test_df)}")
print()


# -----------------------------------------------------------------------------
# Recreate token lists from stored token strings
# -----------------------------------------------------------------------------
def split_tokenised_text(text):
    """
    Convert a space-joined token string back into a token list.
    """
    if pd.isna(text):
        return []

    text = str(text).strip()
    if text == "":
        return []

    return text.split(" ")


train_df["token_list"] = train_df["tokenised_text"].apply(split_tokenised_text)
validation_df["token_list"] = validation_df["tokenised_text"].apply(split_tokenised_text)
test_df["token_list"] = test_df["tokenised_text"].apply(split_tokenised_text)


# -----------------------------------------------------------------------------
# Combine train + validation into one development dataset
# -----------------------------------------------------------------------------
development_df = pd.concat([train_df, validation_df], ignore_index=True)

print("DEVELOPMENT DATASET CREATED")
print("-" * 100)
print(f"Rows in development set: {len(development_df)}")
print()


# -----------------------------------------------------------------------------
# Create tagged documents for Doc2Vec training
# -----------------------------------------------------------------------------
tagged_development_documents = [
    TaggedDocument(words=row["token_list"], tags=[f"DEV_{i}"])
    for i, row in development_df.iterrows()
]

print("Tagged development documents created.")
print(f"Number of tagged development documents: {len(tagged_development_documents)}")
print()


# -----------------------------------------------------------------------------
# Train the final Doc2Vec model on development data
# -----------------------------------------------------------------------------
# We keep the same document-embedding settings used in Step 19:
# - vector_size = 100
# - window = 5
# - min_count = 3
# - dm = 1
# - epochs = 40
# -----------------------------------------------------------------------------
doc2vec_model = Doc2Vec(
    vector_size=100,
    window=5,
    min_count=3,
    workers=1,
    dm=1,
    epochs=40,
    seed=42
)

doc2vec_model.build_vocab(tagged_development_documents)

print("Doc2Vec vocabulary built on development data.")
print(f"Vocabulary size: {len(doc2vec_model.wv)}")
print()

doc2vec_model.train(
    tagged_development_documents,
    total_examples=doc2vec_model.corpus_count,
    epochs=doc2vec_model.epochs
)

doc2vec_model.save(str(model_output))

print("Final Doc2Vec model training completed.")
print()


# -----------------------------------------------------------------------------
# Infer vectors for development and test sets
# -----------------------------------------------------------------------------
def infer_vectors(df, model):
    """
    Infer dense document vectors for a dataframe of token lists.
    """
    vectors = []
    for tokens in df["token_list"]:
        vec = model.infer_vector(tokens, epochs=30)
        vectors.append(vec)

    vectors_df = pd.DataFrame(vectors, columns=[f"vec_{i}" for i in range(model.vector_size)])
    return vectors_df


X_dev = infer_vectors(development_df, doc2vec_model)
X_test = infer_vectors(test_df, doc2vec_model)

y_dev = development_df["label_unstable"].astype(int)
y_test = test_df["label_unstable"].astype(int)

print("VECTOR INFERENCE COMPLETED")
print("-" * 100)
print(f"Development vector shape: {X_dev.shape}")
print(f"Test vector shape:        {X_test.shape}")
print()


# -----------------------------------------------------------------------------
# Fit the final selected Method 2 classifier
# -----------------------------------------------------------------------------
# Best Method 2 classifier from Step 21:
# - Logistic Regression
# - C = 10.0
# - class_weight = balanced
# -----------------------------------------------------------------------------
final_model = LogisticRegression(
    C=10.0,
    class_weight="balanced",
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

final_model.fit(X_dev, y_dev)


# -----------------------------------------------------------------------------
# Generate final test predictions and scores
# -----------------------------------------------------------------------------
y_test_pred = final_model.predict(X_test)
y_test_score = final_model.predict_proba(X_test)[:, 1]


# -----------------------------------------------------------------------------
# Compute final test metrics
# -----------------------------------------------------------------------------
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_score)
pr_auc = average_precision_score(y_test, y_test_score)

metrics_df = pd.DataFrame([{
    "model": "final_doc2vec_logistic_regression",
    "development_rows": len(development_df),
    "test_rows": len(test_df),
    "test_unstable_cases": int(y_test.sum()),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4)
}])

metrics_df.to_csv(metrics_output, index=False)


# -----------------------------------------------------------------------------
# Save final test predictions
# -----------------------------------------------------------------------------
predictions_df = pd.DataFrame({
    "ACN": test_df["ACN"],
    "incident_year": test_df["incident_year"],
    "true_label": y_test,
    "predicted_label": y_test_pred,
    "score": y_test_score
})

predictions_df.to_csv(predictions_output, index=False)


# -----------------------------------------------------------------------------
# Save confusion matrix
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)

cm_df = pd.DataFrame(
    cm,
    index=["actual_0_non_unstable", "actual_1_unstable"],
    columns=["predicted_0_non_unstable", "predicted_1_unstable"]
)

cm_df.to_csv(cm_output)


# -----------------------------------------------------------------------------
# Print final outputs
# -----------------------------------------------------------------------------
print("=" * 100)
print("FINAL DOC2VEC TEST METRICS")
print("=" * 100)
print(metrics_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(model_output.name)
print(metrics_output.name)
print(predictions_output.name)
print(cm_output.name)
print()

print("=" * 100)
print("STEP 22 FINISHED")
print("The final selected Doc2Vec-based model has now been evaluated on the")
print("untouched test set.")
print("=" * 100)