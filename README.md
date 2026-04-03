# Causal Inference with LLMs — Workshop

Three-day academic workshop materials on applying Large Language Models (LLMs) to causal inference.

## Interactive App

**[https://causal-inference-workshop-odxipb5ycay2fkgr9xezet.streamlit.app/](https://causal-inference-workshop-odxipb5ycay2fkgr9xezet.streamlit.app/)**

The interactive companion app runs entirely in the browser — no local installation required. Use the sidebar to navigate between days. Each day's page walks through the core computations step by step, with interactive sliders and buttons that let you adjust parameters (number of cross-fitting folds, noise levels, labeled sample size, and more) and immediately see how the results change. The app uses the same U.S. Congressional speeches dataset (Hein Bound 111) as the local scripts, so the numbers match what you obtain by running the scripts on your own machine.

<!--
The interactive companion app runs entirely in the browser — no local installation required. It mirrors the workshop's three-day structure: use the sidebar to navigate between days. Each day's page walks through the core computations step by step, with interactive sliders and buttons that let you adjust parameters (number of folds, noise levels, labeled sample size, etc.) and immediately see how the results change. The app uses the same U.S. Congressional speeches dataset (Hein Bound 111) as the local scripts, so the numbers you see in the app match what you would obtain by running the scripts on your own machine.
-->

| Day | Topic |
|-----|-------|
| [day1/](day1/) | Standard causal inference, LLM measurement, adjustment methods |
| [day2/](day2/) | Causal estimators (DiM, Regression, G-formula, Matching, IPW, AIPW), measurement error (MCAR vs MNAR) |
| [day3/](day3/) | Estimation and Auditing with LLMs: DML, DSL, Causal Auditing |

## Replicability and Reproducibility

This repository is designed to be fully reproducible. The scripts demonstrate causal inference concepts using a sample of U.S. Congressional speeches from the 111th Congress (Hein Bound).

### Environment Requirements

**Python:**
- Python 3.10+
- Install dependencies: `pip install -r day1/requirements.txt` (Day 1) or `pip install -r day2/requirements.txt` (Day 2)
- Key packages: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `sentence-transformers`

**R:**
- R 4.0+
- Required packages are auto-installed by each script on first run
- Day 1 R scripts: `readr`, `dplyr`, `broom`, `reticulate`
- Day 2 R scripts: `readr`, `dplyr`, `text2vec`, `MatchIt` — **no Python dependency**

### Execution Order

Run all scripts from the repository root directory.

**Day 1 — Python:**
1. `python3 day1/python_app/partA_standard_ci.py`
2. `python3 day1/python_app/partB_llm_measurement.py`
3. `python3 day1/python_app/partC_adjustment.py`
4. `python3 day1/python_app/compare_A_B.py`

**Day 1 — R:**
1. `Rscript day1/r_app/partA_standard_ci.R`
2. `Rscript day1/r_app/partB_llm_measurement.R`
3. `Rscript day1/r_app/partC_adjustment.R`

**Day 2 — Python:**
1. `python3 day2/python_app/partA_sentiment.py`
2. `python3 day2/python_app/partB_estimators.py`
3. `python3 day2/python_app/partC_measurement_error.py`
4. `python3 day2/python_app/compare_days.py`

**Day 2 — R (fully self-contained, no Python required):**
1. `Rscript day2/r_app/partA_sentiment.R`
2. `Rscript day2/r_app/partB_estimators.R`
3. `Rscript day2/r_app/partC_measurement_error.R`


**Day 3 — Python:**
1. `python3 day3/python_app/partA_dml.py`
2. `python3 day3/python_app/partB_dsl.py`
3. `python3 day3/python_app/partC_auditing.py`
4. `python3 day3/python_app/compare_days.py`

**Day 3 — R (fully self-contained, no Python required):**
1. `Rscript day3/r_app/partA_dml.R`
2. `Rscript day3/r_app/partB_dsl.R`
3. `Rscript day3/r_app/partC_auditing.R`

### Determinism and Caching

- **Python:** Embeddings are cached in `day2/data/embeddings_cache.npy` after the first run to avoid recomputation.
- **R:** Day 2 R scripts use `text2vec` TF-IDF embeddings computed natively in R with a fixed seed (2026), ensuring deterministic results without any Python dependency.
- **Random Seeds:** Fixed seeds are set throughout to guarantee reproducible outputs.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code for academic and educational purposes, provided you include the original copyright notice.

Copyright (c) 2026 Moses Boudourides
