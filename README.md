# Machine Learning for Social Data Science — Unstable-Approach Classification from Aviation Safety Narratives

## GitHub Repository (Replication Link)
**[Add your GitHub repository link here after creating the repo]**

---

## Project Overview

This repository contains the full reproducible workflow for a Machine Learning for Social Data Science assessment focused on **classifying unstable-approach events from aviation safety narratives**. The project uses text reports from the **NASA Aviation Safety Reporting System (ASRS)** and compares three text-analysis method families:

- **Method 1:** TF-IDF + supervised classifiers  
- **Method 2:** Doc2Vec + supervised classifiers  
- **Method 3:** DistilBERT transfer learning  

The project also tests whether **environmental and operational context** improves classification beyond narrative text alone.

Because the dataset is imbalanced, evaluation focuses on:

- **PR-AUC**
- **F1-score**
- **ROC-AUC**
- **Precision**
- **Recall**

rather than accuracy alone.

---

## Research Questions

**RQ1:** To what extent can aviation safety narratives support the classification of whether an approach-and-landing event involved an unstable approach?

**RQ2:** Which supervised text-classification approach provides the most reliable held-out performance for identifying unstable-approach events: TF-IDF with Logistic Regression, TF-IDF with Linear SVM, TF-IDF with Multinomial Naive Bayes, Doc2Vec-based classification, or DistilBERT transfer learning?

**RQ3:** Does incorporating environmental and operational context improve the classification of unstable-approach events beyond what can be achieved from narrative text alone?

**RQ4:** Which textual signals and environmental or operational factors are most strongly associated with unstable-approach events, and what do these associations suggest about operational risk in the approach-and-landing phase?

---

## Repository Structure

This repository is organised as a **step-by-step script-based pipeline** to support straightforward replication.

```text
.
├── data/
│   ├── raw/
│   └── prepared/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── method1_tfidf/
│   ├── method2_doc2vec/
│   ├── method3_distilbert/
│   └── comparison/
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
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
├── make_figure3_validation_comparison.py
├── make_figure4_environment_summary.py
├── make_figure5_top_predictive_ngrams_refined.py
├── make_figure6_threshold_tradeoff.py
├── make_figure7_confusion_matrix.py
└── make_figure8_roc_comparison.py

Main Folders
data/raw/

Contains the original ASRS CSV files.

data/prepared/

Contains prepared modelling datasets used in the pipeline.

results/figures/

Contains report-ready figures.

results/tables/

Contains CSV outputs used for report tables and summaries.

results/method1_tfidf/

Contains outputs related to the TF-IDF method family.

results/method2_doc2vec/

Contains outputs related to the Doc2Vec method family.

results/method3_distilbert/

Contains outputs related to the DistilBERT method family.

results/comparison/

Contains cross-method comparison outputs and summary files.

Main Scripts and Workflow
Core Data Preparation and Text-Only Workflow
step0_inspect_raw_data.py — inspects raw ASRS files
step2_rebuild_headers.py — reconstructs grouped ASRS headers
step3_combine_and_scope_check.py — combines files and checks scope
step4_scope_cleaning.py — cleans and filters the scoped dataset
step5_prepare_base_dataset.py — creates the modelling-ready base dataset
step6_eda.py — performs exploratory data analysis
step7_time_based_split.py — creates time-based train/validation/test split
step8_text_only_baseline_models.py — runs text-only baseline models
step9_tune_text_only_models.py — tunes text-only models
step10_text_plus_context_models.py — evaluates text plus raw context
step11_engineered_environment_check.py — evaluates engineered environmental features
step12_final_test_evaluation.py — evaluates the selected final text-only model on the untouched 2025 test set
step13_environment_only_leakage_check.py — environment-only leakage diagnostic
step14_threshold_analysis.py — threshold trade-off analysis
step15_error_analysis.py — final test error analysis
step16_prepare_case_review_files.py — prepares case-review files for qualitative analysis
step17_complete_case_environment_check.py — complete-case sensitivity analysis
Doc2Vec Workflow
step18_doc2vec_prepare_text.py — prepares text for Doc2Vec
step19_doc2vec_train_vectors.py — trains and infers Doc2Vec vectors
step20_doc2vec_validation_models.py — runs Doc2Vec baseline validation models
step21_doc2vec_tuning.py — tunes Doc2Vec models
step22_doc2vec_final_test.py — final Doc2Vec test evaluation
DistilBERT Workflow
step23_distilbert_prepare_data.py — prepares data for DistilBERT
step24_distilbert_train_validation.py — trains and validates DistilBERT
step25_distilbert_final_test.py — final DistilBERT test evaluation
Cross-Method Comparison
step26_cross_method_comparison.py — compares final method-family performance
Figure Generation Scripts
make_figure3_validation_comparison.py
make_figure4_environment_summary.py
make_figure5_top_predictive_ngrams_refined.py
make_figure6_threshold_tradeoff.py
make_figure7_confusion_matrix.py
make_figure8_roc_comparison.py
Data Requirements
data/raw/
data/prepared/
Large model and checkpoint files are tracked using Git LFS so that the full project structure, saved outputs, and model artefacts can be uploaded while preserving repository usability.
