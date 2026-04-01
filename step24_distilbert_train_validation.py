from pathlib import Path
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# =============================================================================
# STEP 24: TRAIN AND VALIDATE DISTILBERT
# =============================================================================
# Purpose of this step:
# The previous step prepared the text splits for a transformer-based model.
# The present step now fine-tunes DistilBERT for the binary classification task:
#
# unstable approach (1) vs non-unstable case (0)
#
# This is Method 3 in the project. It differs from the previous methods because:
# - Method 1 used TF-IDF features with classical classifiers
# - Method 2 used Doc2Vec document embeddings with classical classifiers
# - Method 3 uses transformer-based transfer learning
#
# The aim here is to evaluate DistilBERT fairly on the same time-based setup:
# - train = 2018-2023
# - validation = 2024
#
# Important methodological note:
# This step uses ONLY the train and validation sets.
# The 2025 test set remains untouched until the final evaluation step.
#
# Another important point:
# The unstable class is smaller than the non-unstable class. To account for that,
# the training loss is weighted so that unstable cases receive higher attention
# during optimisation.
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
project_folder = Path(__file__).resolve().parent

train_file = project_folder / "data" / "processed" / "step23_distilbert_train.csv"
validation_file = project_folder / "data" / "processed" / "step23_distilbert_validation.csv"

output_dir = project_folder / "models" / "step24_distilbert_outputs"

metrics_output = project_folder / "results" / "tables" / "step24_distilbert_validation_metrics.csv"
checkpoint_output = project_folder / "results" / "tables" / "step24_distilbert_best_checkpoint_summary.csv"
predictions_output = project_folder / "results" / "tables" / "step24_distilbert_validation_predictions.csv"


# -----------------------------------------------------------------------------
# Load prepared train and validation data
# -----------------------------------------------------------------------------
train_df = pd.read_csv(train_file, low_memory=False)
validation_df = pd.read_csv(validation_file, low_memory=False)

print("=" * 100)
print("STEP 24: TRAIN AND VALIDATE DISTILBERT")
print("=" * 100)
print(f"Train rows loaded:      {len(train_df)}")
print(f"Validation rows loaded: {len(validation_df)}")
print()


# -----------------------------------------------------------------------------
# Prepare Hugging Face datasets
# -----------------------------------------------------------------------------
# The transformer expects:
# - one text column
# - one numeric label column
#
# Here we rename the target to the standard name 'labels' because Hugging Face
# models and trainers expect that convention.
# -----------------------------------------------------------------------------
train_hf = train_df[["ACN", "incident_year", "cleaned_text", "label_unstable"]].copy()
validation_hf = validation_df[["ACN", "incident_year", "cleaned_text", "label_unstable"]].copy()

train_hf = train_hf.rename(columns={"label_unstable": "labels"})
validation_hf = validation_hf.rename(columns={"label_unstable": "labels"})

train_dataset = Dataset.from_pandas(train_hf, preserve_index=False)
validation_dataset = Dataset.from_pandas(validation_hf, preserve_index=False)


# -----------------------------------------------------------------------------
# Load tokenizer
# -----------------------------------------------------------------------------
# DistilBERT is used as a lighter transformer model that is still strong enough
# to test whether contextual language modelling improves on the earlier methods.
# -----------------------------------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Tokenizer loaded: {model_name}")
print()


# -----------------------------------------------------------------------------
# Tokenisation function
# -----------------------------------------------------------------------------
# max_length=256 is used as a practical compromise:
# - long enough to retain substantial narrative content
# - short enough to keep training manageable
# -----------------------------------------------------------------------------
def tokenize_batch(batch):
    return tokenizer(
        batch["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


tokenized_train = train_dataset.map(tokenize_batch, batched=True)
tokenized_validation = validation_dataset.map(tokenize_batch, batched=True)

# Keep only the columns required by the Trainer
tokenized_train = tokenized_train.remove_columns(["ACN", "incident_year", "cleaned_text"])
tokenized_validation = tokenized_validation.remove_columns(["ACN", "incident_year", "cleaned_text"])

tokenized_train.set_format("torch")
tokenized_validation.set_format("torch")


# -----------------------------------------------------------------------------
# Compute class weights for imbalanced learning
# -----------------------------------------------------------------------------
# The unstable class is the minority class, so it should receive a larger weight
# in the training loss.
# -----------------------------------------------------------------------------
class_counts = train_hf["labels"].value_counts().sort_index()
count_0 = int(class_counts.get(0, 0))
count_1 = int(class_counts.get(1, 0))

# Standard inverse-frequency style weighting
total_count = count_0 + count_1
weight_0 = total_count / (2 * count_0)
weight_1 = total_count / (2 * count_1)

class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

print("CLASS DISTRIBUTION AND LOSS WEIGHTS")
print("-" * 100)
print(f"Class 0 count (non-unstable): {count_0}")
print(f"Class 1 count (unstable):     {count_1}")
print(f"Weight for class 0: {weight_0:.4f}")
print(f"Weight for class 1: {weight_1:.4f}")
print()


# -----------------------------------------------------------------------------
# Load DistilBERT classification model
# -----------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# -----------------------------------------------------------------------------
# Custom Trainer with weighted loss
# -----------------------------------------------------------------------------
# This overrides the default loss so that the unstable class receives greater
# importance during optimisation.
# -----------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=None
        )

        logits = outputs.get("logits")
        weights = class_weights.to(logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Metric function
# -----------------------------------------------------------------------------
# This function computes the same style of metrics used throughout the rest of
# the project so that Method 3 remains directly comparable with Methods 1 and 2.
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to probabilities for the positive class
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    positive_scores = probs[:, 1]

    preds = np.argmax(logits, axis=1)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    roc_auc = roc_auc_score(labels, positive_scores)
    pr_auc = average_precision_score(labels, positive_scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# -----------------------------------------------------------------------------
# Training arguments
# -----------------------------------------------------------------------------
# These settings are intentionally moderate for a student project. The objective
# is to obtain a fair transformer benchmark without turning the workflow into an
# excessively large experiment.
# -----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(output_dir),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    seed=42
)


# -----------------------------------------------------------------------------
# Build trainer
# -----------------------------------------------------------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics
)


# -----------------------------------------------------------------------------
# Train the model
# -----------------------------------------------------------------------------
print("Starting DistilBERT fine-tuning...")
print()

trainer.train()

print()
print("DistilBERT training completed.")
print()


# -----------------------------------------------------------------------------
# Evaluate the best checkpoint on validation
# -----------------------------------------------------------------------------
eval_results = trainer.evaluate()

# Save evaluation metrics in the same style as earlier methods
metrics_df = pd.DataFrame([{
    "model": "distilbert_validation",
    "validation_rows": len(validation_hf),
    "validation_unstable_cases": int(validation_hf["labels"].sum()),
    "precision": round(eval_results["eval_precision"], 4),
    "recall": round(eval_results["eval_recall"], 4),
    "f1": round(eval_results["eval_f1"], 4),
    "roc_auc": round(eval_results["eval_roc_auc"], 4),
    "pr_auc": round(eval_results["eval_pr_auc"], 4)
}])

metrics_df.to_csv(metrics_output, index=False)


# -----------------------------------------------------------------------------
# Save best checkpoint summary
# -----------------------------------------------------------------------------
checkpoint_df = pd.DataFrame([{
    "model_name": model_name,
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_metric": trainer.state.best_metric,
    "selection_metric": training_args.metric_for_best_model,
    "num_train_epochs": training_args.num_train_epochs,
    "max_length": 256,
    "learning_rate": training_args.learning_rate,
    "train_batch_size": training_args.per_device_train_batch_size,
    "eval_batch_size": training_args.per_device_eval_batch_size
}])

checkpoint_df.to_csv(checkpoint_output, index=False)


# -----------------------------------------------------------------------------
# Save validation predictions
# -----------------------------------------------------------------------------
pred_output = trainer.predict(tokenized_validation)
logits = pred_output.predictions

exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
positive_scores = probs[:, 1]
predicted_labels = np.argmax(logits, axis=1)

predictions_df = pd.DataFrame({
    "ACN": validation_hf["ACN"],
    "incident_year": validation_hf["incident_year"],
    "true_label": validation_hf["labels"],
    "predicted_label": predicted_labels,
    "score": positive_scores
})

predictions_df.to_csv(predictions_output, index=False)


# -----------------------------------------------------------------------------
# Final printout
# -----------------------------------------------------------------------------
print("=" * 100)
print("DISTILBERT VALIDATION METRICS")
print("=" * 100)
print(metrics_df.to_string(index=False))
print()

print("FILES CREATED")
print("-" * 100)
print(metrics_output.name)
print(checkpoint_output.name)
print(predictions_output.name)
print()

print("=" * 100)
print("STEP 24 FINISHED")
print("DistilBERT has been trained and evaluated on the validation set.")
print("The test set has NOT been used yet.")
print("=" * 100)