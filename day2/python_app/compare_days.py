"""
Day 2 - Compare Days
This script compares the causal estimates from Day 1 (Stance as outcome)
with Day 2 (Sentiment as outcome) to show that the choice of LLM measurement
changes the substantive conclusions.
"""

import os
import pandas as pd
import statsmodels.api as sm

def estimate_ate(df, y_col):
    """Estimate ATE using regression adjustment."""
    Y = df[y_col]
    D = df['D']
    X = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)

    X_reg = sm.add_constant(pd.concat([D, X], axis=1))
    ols = sm.OLS(Y, X_reg).fit()
    return ols.params['D'], ols.bse['D']

def main():
    print("--- Comparing Day 1 (Stance) vs Day 2 (Sentiment) ---")

    day1_path = 'day1/data/speeches_with_stance.csv'
    day2_path = 'day2/data/speeches_with_sentiment.csv'

    missing = []
    if not os.path.exists(day1_path):
        missing.append(day1_path)
    if not os.path.exists(day2_path):
        missing.append(day2_path)

    if missing:
        for p in missing:
            print(f"Error: {p} not found.")
        if day1_path in missing:
            print("  -> Run Day 1 Part B (LLM stance scoring) first.")
        if day2_path in missing:
            print("  -> Run Day 2 partA_sentiment.py first.")
        return

    df1 = pd.read_csv(day1_path)
    df2 = pd.read_csv(day2_path)

    df1['D'] = (df1['party'] == 'Republican').astype(int)
    df2['D'] = (df2['party'] == 'Republican').astype(int)

    ate_stance,    se_stance    = estimate_ate(df1, 'Y_tilde')
    ate_sentiment, se_sentiment = estimate_ate(df2, 'Y_tilde')

    print("\nOutcome: LLM Stance Score (Day 1)")
    print(f"ATE Estimate: {ate_stance:.4f} (SE: {se_stance:.4f})")

    print("\nOutcome: LLM Sentiment Score (Day 2)")
    print(f"ATE Estimate: {ate_sentiment:.4f} (SE: {se_sentiment:.4f})")

    print("\nConclusion:")
    print("The causal effect of party affiliation differs depending on")
    print("which LLM measurement is used as the outcome variable.")

if __name__ == "__main__":
    main()
