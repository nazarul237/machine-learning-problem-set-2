from pathlib import Path
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =============================================================================
# STEP 25: FINAL DISTILBERT TEST EVALUATION
# =============================================================================
# Purpose of this step:
# In Step 24, DistilBERT was fine-tuned on the training data and evaluated on the
# validation data. The best checkpoint was selected on validation performance.
#
# The purpose of the present step is now to evaluate that selected DistilBERT
# checkpoint once on the untouched 2025 test set.
#
# Important methodological note:
# - No further tuning is done here.
# - The test set is used only once at this stage.
# - This mirrors the final locked evaluation logic used for Methods 1 and 2.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

test_file = project_folder / "data" / "processed" / "step23_distilbert_test.csv"
checkpoint_file = project_folder / "results" / "tables" / "step24_distilbert_best_checkpoint_summary.csv"

metrics_output = project_folder / "results" / "tables" / "step25_distilbert_final_test_metrics.csv"
predictions_output = project_folder / "results" / "tables" / "step25_distilbert_final_test_predictions.csv"
cm_output = project_folder / "results" / "tables" / "step25_distilbert_final_test_confusion_matrix.csv"


# -----------------------------------------------------------------------------
# Load test data and best checkpoint summary
# -----------------------------------------------------------------------------
test_df = pd.read_csv(test_file, low_memory=False)
checkpoint_df = pd.read_csv(checkpoint_file, low_memory=False)

best_checkpoint = checkpoint_df.loc[0, "best_checkpoint"]
model_name = checkpoint_df.loc[0, "model_name"]

print("=" * 100)
print("STEP 25: FINAL DISTILBERT TEST EVALUATION")
print("=" * 100)
print(f"Test rows loaded: {len(test_df)}")
print(f"Best checkpoint: {best_checkpoint}")
print(f"Model name: {model_name}")
print()


# -----------------------------------------------------------------------------
# Prepare Hugging Face dataset
# -----------------------------------------------------------------------------
test_hf = test_df[["ACN", "incident_year", "cleaned_text", "label_unstable"]].copy()
test_hf = test_hf.rename(columns={"label_unstable": "labels"})

test_dataset = Dataset.from_pandas(test_hf, preserve_index=False)


# -----------------------------------------------------------------------------
# Load tokenizer from the original base model
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Tokenizer loaded.")
print()


# -----------------------------------------------------------------------------
# Tokenisation
# -----------------------------------------------------------------------------
def tokenize_batch(batch):
    return tokenizer(
        batch["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_test = test_dataset.map(tokenize_batch, batched=True)
tokenized_test = tokenized_test.remove_columns(["ACN", "incident_year", "cleaned_text"])
tokenized_test.set_format("torch")


# -----------------------------------------------------------------------------
# Load the best saved checkpoint
# -----------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

print("Best checkpoint model loaded.")
print()


# -----------------------------------------------------------------------------
# Create a simple Trainer for prediction/evaluation
# -----------------------------------------------------------------------------
trainer = Trainer(
    model=model
)


# -----------------------------------------------------------------------------
# Predict on the test set
# -----------------------------------------------------------------------------
pred_output = trainer.predict(tokenized_test)
logits = pred_output.predictions
labels = test_hf["labels"].to_numpy()

# Convert logits to probabilities
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
positive_scores = probs[:, 1]
predicted_labels = np.argmax(logits, axis=1)


# -----------------------------------------------------------------------------
# Compute final test metrics
# -----------------------------------------------------------------------------
precision = precision_score(labels, predicted_labels, zero_division=0)
recall = recall_score(labels, predicted_labels, zero_division=0)
f1 = f1_score(labels, predicted_labels, zero_division=0)
roc_auc = roc_auc_score(labels, positive_scores)
pr_auc = average_precision_score(labels, positive_scores)

metrics_df = pd.DataFrame([{
    "model": "final_distilbert",
    "test_rows": len(test_hf),
    "test_unstable_cases": int(test_hf["labels"].sum()),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(pr_auc, 4)
}])

metrics_df.to_csv(metrics_output, index=False)


# -----------------------------------------------------------------------------
# Save test predictions
# -----------------------------------------------------------------------------
predictions_df = pd.DataFrame({
    "ACN": test_hf["ACN"],
    "incident_year": test_hf["incident_year"],
    "true_label": test_hf["labels"],
    "predicted_label": predicted_labels,
    "score": positive_scores
})

predictions_df.to_csv(predictions_output, index=False)


# -----------------------------------------------------------------------------
# Save confusion matrix
# -----------------------------------------------------------------------------
cm = confusion_matrix(labels, predicted_labels)
cm_df = pd.DataFrame(
    cm,
    index=["actual_0_non_unstable", "actual_1_unstable"],
    columns=["predicted_0_non_unstable", "predicted_1_unstable"]
)
cm_df.to_csv(cm_output)


# -----------------------------------------------------------------------------
# Final output
# -----------------------------------------------------------------------------
print("=" * 100)
print("FINAL DISTILBERT TEST METRICS")
print("=" * 100)
print(metrics_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(metrics_output.name)
print(predictions_output.name)
print(cm_output.name)
print()

print("=" * 100)
print("STEP 25 FINISHED")
print("The selected DistilBERT model has now been evaluated on the untouched")
print("2025 test set.")
print("=" * 100)