# Causal Inference with LLMs — Workshop

Three-day academic workshop materials on applying Large Language Models (LLMs) to causal inference.

| Day | Topic |
|-----|-------|
| [day1/](day1/) | Standard causal inference, LLM measurement, adjustment methods |
| day2/ | Coming soon |
| day3/ | Coming soon |

## Replicability and Reproducibility

This repository is designed to be fully reproducible. The scripts demonstrate causal inference concepts using a sample of 8,000 U.S. Congressional speeches from the 111th Congress (Hein Bound).

### Environment Requirements

**Python:**
- Python 3.10+
- Install dependencies: `pip install -r day1/requirements.txt`
- Key packages: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `sentence-transformers`

**R:**
- R 4.5+
- Required packages: `readr`, `dplyr`, `broom`, `reticulate`
- *Note:* The R scripts use `reticulate` to call Python's `sentence-transformers` library to ensure exact parity with the Python implementation. You must have a Python environment with `sentence-transformers` installed and accessible to R.

### Execution Order

To reproduce the results for Day 1, run the scripts in the following order from the root of the repository:

**Python:**
1. `python day1/python_app/partA_standard_ci.py`
2. `python day1/python_app/partB_llm_measurement.py`
3. `python day1/python_app/partC_adjustment.py`
4. `python day1/python_app/compare_A_B.py`

**R:**
1. `Rscript day1/r_app/partA_standard_ci.R`
2. `Rscript day1/r_app/partB_llm_measurement.R`
3. `Rscript day1/r_app/partC_adjustment.R`

### Determinism and Caching

- **LLM Embeddings:** The `sentence-transformers` embeddings generation in Part B can be computationally expensive. The scripts implement a caching mechanism (`embeddings_cache.npy`) that is shared between Python and R. This ensures that the exact same embeddings are used across both languages without recomputation.
- **Random Seeds:** Random seeds are set where applicable (e.g., generating the random direction vector for stance projection) to guarantee deterministic outputs.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code for academic and educational purposes, provided you include the original copyright notice.

Copyright (c) 2026 Moses Boudourides
