"""
Day 3 - Compare Days
This script compares the causal estimates across Day 1, Day 2, and Day 3
to show how the choice of estimator and measurement affects conclusions.

Day 1: Stance score, simple OLS with year dummies
Day 2: Sentiment score, simple OLS with year dummies
Day 3: Sentiment score, DML with high-dimensional text features

Run from seminar_computations/ root:
    python3 day3/python_app/compare_days.py

Note: Run partA_dml.py first to generate day3/data/dml_results.csv.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm


def estimate_ols(df, y_col):
    """OLS with year dummies. Explicitly names the treatment column 'D'."""
    Y = df[y_col]
    # Rename to 'D' so params key is always 'D' regardless of source column name
    D = (df['party'] == 'Republican').astype(int).rename('D')
    X = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)
    X_reg = sm.add_constant(pd.concat([D, X], axis=1))
    ols = sm.OLS(Y, X_reg).fit()
    return ols.params['D'], ols.bse['D']


def main():
    print("--- Comparing Day 1, Day 2, and Day 3 Estimates ---")

    # Correct paths relative to seminar_computations/
    day1_path = 'day1/data/speeches_with_stance.csv'
    day2_path = 'day2/data/speeches_with_sentiment.csv'
    day3_path = 'day3/data/dml_results.csv'

    rows = []

    # ── Day 1 ──────────────────────────────────────────────────────────────────
    if os.path.exists(day1_path):
        df1 = pd.read_csv(day1_path)
        ate1, se1 = estimate_ols(df1, 'Y_tilde')
        rows.append({'Day': 'Day 1', 'Outcome': 'Stance (LLM)',
                     'Estimator': 'OLS (year dummies)', 'ATE': ate1, 'SE': se1})
        print(f"\nDay 1 (Stance, OLS):      ATE = {ate1:.4f}  SE = {se1:.4f}")
    else:
        print(f"\nDay 1 data not found at {day1_path}.")
        print("  Run: python3 day1/python_app/partA_stance.py")

    # ── Day 2 ──────────────────────────────────────────────────────────────────
    if os.path.exists(day2_path):
        df2 = pd.read_csv(day2_path)
        ate2, se2 = estimate_ols(df2, 'Y_tilde')
        rows.append({'Day': 'Day 2', 'Outcome': 'Sentiment (LLM)',
                     'Estimator': 'OLS (year dummies)', 'ATE': ate2, 'SE': se2})
        print(f"Day 2 (Sentiment, OLS):   ATE = {ate2:.4f}  SE = {se2:.4f}")
    else:
        print(f"Day 2 data not found at {day2_path}.")
        print("  Run: python3 day2/python_app/partA_sentiment.py")

    # ── Day 3 DML ──────────────────────────────────────────────────────────────
    if os.path.exists(day3_path):
        df3 = pd.read_csv(day3_path)
        dml_row = df3[df3['estimator'].str.contains('DML', case=False)].iloc[0]
        ate3, se3 = dml_row['estimate'], dml_row['se']
        rows.append({'Day': 'Day 3', 'Outcome': 'Sentiment (LLM)',
                     'Estimator': 'DML (text features, K=5)', 'ATE': ate3, 'SE': se3})
        print(f"Day 3 (Sentiment, DML):   ATE = {ate3:.4f}  SE = {se3:.4f}")
    else:
        print(f"Day 3 DML results not found at {day3_path}.")
        print("  Run: python3 day3/python_app/partA_dml.py")

    # ── Summary ────────────────────────────────────────────────────────────────
    if rows:
        print("\nSummary Table:")
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))

        os.makedirs('day3/data', exist_ok=True)
        out_path = 'day3/data/compare_days.csv'
        summary.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    print("\nConclusion:")
    print("Moving from simple OLS to DML with high-dimensional text features")
    print("changes the ATE estimate by controlling for text-based confounders")
    print("that year dummies alone cannot capture.")


if __name__ == "__main__":
    main()
