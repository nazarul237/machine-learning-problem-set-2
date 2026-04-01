# Machine Learning for Social Data Science — Unstable Approach Classification from Aviation Safety Narratives

**GitHub repository (replication link):**  
<https://github.com/nazarul237/machine-learning-problem-set-2>

---

## Purpose and Scope

This repository contains the full reproducible workflow for my **Machine Learning for Social Data Science** assessment project. The study investigates whether **aviation safety narratives** from the NASA Aviation Safety Reporting System (ASRS) can be used to classify whether an approach-and-landing event involved an **unstable approach**.

The project is designed as a supervised text-classification study and compares three method families:

1. **TF-IDF text classification** with classical supervised models  
2. **Doc2Vec document embedding classification**  
3. **DistilBERT transformer-based transfer learning**

The workflow also tests whether adding **environmental and operational context** improves classification beyond narrative text alone.

The repository is organised as a **step-based pipeline** so that the full workflow can be followed clearly from raw-data inspection to final cross-method comparison.

---

## Research Questions

**RQ1 (Main):**  
To what extent can aviation safety narratives support the classification of whether an approach-and-landing event involved an unstable approach?

**RQ2:**  
Which supervised text-classification approach provides the most reliable held-out performance for identifying unstable-approach events: **TF-IDF with Logistic Regression, TF-IDF with Linear SVM, TF-IDF with Multinomial Naive Bayes, Doc2Vec-based classification, or DistilBERT transfer learning**?

**RQ3:**  
Does incorporating environmental and operational context improve the classification of unstable-approach events beyond what can be achieved from narrative text alone?

**RQ4:**  
Which textual signals and environmental or operational factors are most strongly associated with unstable-approach events, and what do these associations suggest about operational risk in the approach-and-landing phase?

---

## Project Overview

The corpus of texts is drawn from the **NASA Aviation Safety Reporting System (ASRS)**. The main modelling text field is based on the **Narrative** field, with **Synopsis** used only as a fallback where required.

The final modelling dataset was scoped to:

- **Years:** 2018–2025  
- **Flight phases:** Initial Approach, Final Approach, and Landing  
- **Target:** binary unstable-approach classification derived from anomaly coding  

The final modelling dataset contains:

- **11,165 total reports**
- **1,470 unstable reports**
- **9,695 non-unstable reports**

A strict **time-based split** was used:

- **Train:** 2018–2023  
- **Validation:** 2024  
- **Test:** 2025  

This design was chosen to reflect a more realistic out-of-time evaluation and to reduce information leakage from future reports into model development.

---

## Final Main Finding

Across the three method families, the strongest overall held-out performance came from:

**Method 1 — TF-IDF text classification with tuned Logistic Regression**

This method achieved the best balance across the final test metrics and outperformed both the Doc2Vec-based approach and the DistilBERT transformer benchmark.

A key substantive finding of the project is that the **narrative text itself** carries the strongest predictive signal, while environmental and operational context provides only **limited additional value** in this dataset.

---

## Repository Structure

This repository has been reorganised into a cleaner assessment-style structure.

```text
machine-learning-problem-set-2/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── results/
│   ├── figures/
│   └── tables/
│
├── models/
│   ├── step19_doc2vec_model.model
│   ├── step22_doc2vec_final_model.model
│   └── step24_distilbert_outputs/
│
├── scripts/
│   └── figure_builders/
│
├── step0_inspect_raw_data.py
├── step2_rebuild_headers.py
├── step3_combine_and_scope_check.py
├── step4_scope_cleaning.py
├── step5_prepare_base_dataset.py
├── step6_eda.py
├── step7_time_based_split.py
├── step8_text_only_baseline_models.py
├── step9_tune_text_only_models.py
├── step10_text_plus_context_models.py
├── step11_engineered_environment_check.py
├── step12_final_test_evaluation.py
├── step13_environment_only_leakage_check.py
├── step14_threshold_analysis.py
├── step15_error_analysis.py
├── step16_prepare_case_review_files.py
├── step17_complete_case_environment_check.py
├── step18_doc2vec_prepare_text.py
├── step19_doc2vec_train_vectors.py
├── step20_doc2vec_validation_models.py
├── step21_doc2vec_tuning.py
├── step22_doc2vec_final_test.py
├── step23_distilbert_prepare_data.py
├── step24_distilbert_train_validation.py
├── step25_distilbert_final_test.py
├── step26_cross_method_comparison.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes

Raw Data Used

The raw data actually used in the pipeline are the following ASRS CSV files inside data/raw/:

ASRS_DBOnline.csv
ASRS_DBOnline (1).csv
ASRS_DBOnline (2).csv
ASRS_DBOnline (3).csv
ASRS_DBOnline (4).csv
ASRS_DBOnline (5).csv
ASRS_DBOnline (6).csv
ASRS_DBOnline (7).csv

These are the files read in the early raw-data steps of the workflow.

Data Folders
data/raw/

Contains the original ASRS CSV files used for the project.

data/processed/

Contains processed intermediate datasets created during the workflow, including:

cleaned and scoped datasets
prepared Doc2Vec text files
inferred Doc2Vec vector files
prepared DistilBERT datasets
data/splits/

Contains the time-based split datasets:

step7_train_dataset.csv
step7_validation_dataset.csv
step7_test_dataset.csv
Results and Model Output Folders
results/tables/

Contains saved CSV outputs from the pipeline, including:

summary tables
validation metrics
final test metrics
confusion matrices
saved prediction files
threshold-analysis results
cross-method comparison tables
results/figures/

Contains figure outputs generated for reporting, including:

EDA figures
threshold trade-off figures
ROC comparison figures
confusion matrix figures
validation-comparison figures
models/

Contains saved trained model artefacts, including:

Doc2Vec model files
final Doc2Vec model
DistilBERT checkpoint outputs
Full Pipeline and Script Purpose
Step 0 to Step 7 — Data inspection, scoping, preparation, and splitting
step0_inspect_raw_data.py
Inspects the raw ASRS files and checks their structure.
step2_rebuild_headers.py
Reconstructs the grouped headers from the ASRS CSV files.
step3_combine_and_scope_check.py
Combines the raw ASRS files and performs initial scope checks.
step4_scope_cleaning.py
Restricts the study to the selected years and flight phases.
step5_prepare_base_dataset.py
Creates the modelling-ready base dataset and target variable.
step6_eda.py
Performs exploratory data analysis on the base dataset.
step7_time_based_split.py
Creates the time-based train / validation / test split.
Step 8 to Step 17 — Method 1, context checks, threshold analysis, and error analysis
step8_text_only_baseline_models.py
Runs baseline TF-IDF text-only models on the validation set.
step9_tune_text_only_models.py
Performs systematic hyperparameter tuning for the text-only models.
step10_text_plus_context_models.py
Tests text plus safe contextual variables on the validation set.
step11_engineered_environment_check.py
Tests text plus engineered environmental features.
step12_final_test_evaluation.py
Final locked test evaluation for the selected Method 1 model.
step13_environment_only_leakage_check.py
Runs an environment-only diagnostic and feature-removal check.
step14_threshold_analysis.py
Examines precision / recall / F1 trade-offs across thresholds.
step15_error_analysis.py
Analyses final test-set false positives, false negatives, and true positives.
step16_prepare_case_review_files.py
Prepares readable case-review files for manual interpretation.
step17_complete_case_environment_check.py
Runs a complete-case sensitivity analysis for environmental features.
Step 18 to Step 22 — Method 2, Doc2Vec pipeline
step18_doc2vec_prepare_text.py
Prepares tokenised text for Doc2Vec.
step19_doc2vec_train_vectors.py
Trains the Doc2Vec model and generates dense document vectors.
step20_doc2vec_validation_models.py
Runs baseline classifiers on validation Doc2Vec vectors.
step21_doc2vec_tuning.py
Tunes the classifiers built on top of Doc2Vec vectors.
step22_doc2vec_final_test.py
Final locked test evaluation for the selected Method 2 model.
Step 23 to Step 26 — Method 3, DistilBERT pipeline, and final comparison
step23_distilbert_prepare_data.py
Prepares the train / validation / test text for DistilBERT.
step24_distilbert_train_validation.py
Fine-tunes DistilBERT and selects the best validation checkpoint.
step25_distilbert_final_test.py
Final locked test evaluation for the selected DistilBERT checkpoint.
step26_cross_method_comparison.py
Combines the final Method 1, Method 2, and Method 3 results into one final comparison.
Figure Builder Scripts

The folder scripts/figure_builders/ contains helper scripts used to generate report figures and summary visuals, for example:

validation comparison figure
environmental summary figure
threshold trade-off figure
ROC comparison figure
confusion matrix figure

These scripts support presentation and reporting, but they are separate from the main numbered pipeline.

Environment Setup

This repository is intended to run inside a clean Python virtual environment.

Recommended Python version

For this project, Python 3.12 is recommended.

This is especially important because gensim / Doc2Vec installation can fail on Python 3.13 on some systems.

Installation
1. Create a virtual environment

macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Upgrade pip
python -m pip install --upgrade pip

3. Install dependencies
pip install -r requirements.txt

requirements.txt

The repository expects the following main packages:

pandas
numpy
matplotlib
scikit-learn
gensim
datasets
transformers
torch
accelerate
sentencepiece
Recommended Run Order

From the repository root, run the scripts in this order:
python step0_inspect_raw_data.py
python step2_rebuild_headers.py
python step3_combine_and_scope_check.py
python step4_scope_cleaning.py
python step5_prepare_base_dataset.py
python step6_eda.py
python step7_time_based_split.py
python step8_text_only_baseline_models.py
python step9_tune_text_only_models.py
python step10_text_plus_context_models.py
python step11_engineered_environment_check.py
python step12_final_test_evaluation.py
python step13_environment_only_leakage_check.py
python step14_threshold_analysis.py
python step15_error_analysis.py
python step16_prepare_case_review_files.py
python step17_complete_case_environment_check.py
python step18_doc2vec_prepare_text.py
python step19_doc2vec_train_vectors.py
python step20_doc2vec_validation_models.py
python step21_doc2vec_tuning.py
python step22_doc2vec_final_test.py
python step23_distilbert_prepare_data.py
python step24_distilbert_train_validation.py
python step25_distilbert_final_test.py
python step26_cross_method_comparison.py

Expected Outputs

If the pipeline runs successfully, the repository should produce:

In results/tables/
EDA summary tables
validation metrics
final test metrics
confusion matrices
threshold-analysis outputs
error-analysis tables
Doc2Vec tuning results
DistilBERT validation and final test outputs
final cross-method comparison table
In results/figures/
EDA plots
threshold trade-off plot
confusion-matrix figure
ROC comparison figure
validation-comparison figures
In models/
trained Doc2Vec model
final Doc2Vec model
DistilBERT checkpoint directory
Methodological Notes
Time-based evaluation

A strict out-of-time design was used:

Train: 2018–2023
Validation: 2024
Test: 2025

This is more realistic than a random split and reduces temporal leakage.

Imbalanced classification

Because unstable-approach cases are less frequent than non-unstable cases, the project emphasises:

Precision
Recall
F1-score
ROC-AUC
PR-AUC

rather than plain accuracy.

Leakage control

The project applies leakage-aware modelling practice throughout:

TF-IDF fit only on training or development text
Doc2Vec trained only on pre-test data
DistilBERT checkpoint selection based on validation only
test set used once at the final locked evaluation stage
leakage-prone coded variables excluded from predictive features
Context and environment checks

The repository includes multiple robustness checks to test whether environmental and operational variables provide real added value beyond narrative text.

Main Conclusion

The final results show that aviation safety narratives do contain meaningful predictive signal for unstable-approach classification.

Across the three method families, the strongest held-out performance came from:

TF-IDF + tuned Logistic Regression

This is an important methodological finding because it shows that, for this domain-specific aviation corpus, a strong classical text-classification pipeline outperformed both the Doc2Vec method and the DistilBERT transformer benchmark.

Author

SULTAN NAZARUL ISLAM
MSc Business Analytics
University of Exeter


Paste this into `README.md`, save it, then run:

```bash
git add README.md
git commit -m "add polished project README"
git push
