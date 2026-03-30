# Causal Inference with LLMs — Day 1 Workshop App

Interactive companion app for the **Causal Inference with LLMs** workshop (Day 1).

Built with [Streamlit](https://streamlit.io). No coding knowledge required — just click the buttons.

## What it covers

| Section | Slide topic |
|---|---|
| 0 · Data | Load built-in dataset or upload your own CSV |
| 1 · Naive Estimator | Naive difference in means + ATT/selection bias decomposition |
| 2 · Overlap Check | Propensity score estimation and positivity verification |
| 3 · Adjustment Formula & IPW | G-formula and Inverse Probability Weighting |
| 4 · LLM Measurement | Stance proxy construction, non-classical measurement error |
| 5 · Summary | All estimators compared side by side |

## Dataset

The built-in dataset is a sample of 8,000 U.S. Congressional speeches from the **111th Congress**
(Hein Bound 111). Treatment: party (Republican = 1). Outcome: text length (Part A) or
LLM-derived stance score (Part B).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Select this repository, branch `main`, and file `app.py`.
5. Click **Deploy**.

## Author

Moses Boudourides · Northwestern University
