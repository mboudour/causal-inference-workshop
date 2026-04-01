Day 3: High-Dimensional Estimation, Measurement Correction, and LLM Auditing
=============================================================================

OVERVIEW
--------
Day 3 covers three topics from the slides:
  Part A: Double Machine Learning (DML) with high-dimensional text covariates
  Part B: Design-Based Supervised Learning (DSL) for measurement correction
  Part C: Auditing LLMs as causal systems (Average Causal Bias)

DATA
----
All scripts read from: day1_package/data/speeches_sample.csv
Output files are written to: day3/data/

PYTHON EXECUTION ORDER
----------------------
Run from the seminar_computations/ directory:

  python3 day3/python_app/partA_dml.py
  python3 day3/python_app/partB_dsl.py
  python3 day3/python_app/partC_auditing.py
  python3 day3/python_app/compare_days.py

R EXECUTION ORDER (fully independent of Python)
------------------------------------------------
Run from the seminar_computations/ directory:

  Rscript day3/r_app/partA_dml.R
  Rscript day3/r_app/partB_dsl.R
  Rscript day3/r_app/partC_auditing.R

R PACKAGE DEPENDENCIES (auto-installed on first run)
-----------------------------------------------------
  readr, dplyr, text2vec, glmnet, Matrix

PYTHON PACKAGE DEPENDENCIES
----------------------------
  pandas, numpy, statsmodels, scikit-learn
  (all included in requirements.txt)

NOTES
-----
- The R scripts are fully self-contained and do NOT require Python to be run first.
- All random seeds are fixed for reproducibility (seed 2026 for outcome simulation,
  seed 42 for cross-fitting folds).
- The DML estimator uses 5-fold cross-fitting with Ridge regression for the
  outcome model and Logistic regression for the propensity score model.
- The DSL estimator uses 10% of the data as a labeled sample.
- The auditing script simulates a biased LLM model and computes ACB via
  counterfactual interventions on the party signal.
