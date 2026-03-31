"""
Day 2 - Part C: Measurement Error and Bias
This script demonstrates the difference between Classical (MCAR) and
Non-classical (MNAR) measurement error, as discussed in the slides.
Since we don't observe the "true" latent sentiment Y_i, we simulate it
here to show how different types of error affect the ATE estimate.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

def estimate_ate(y, d, x):
    """Simple regression adjustment ATE."""
    X_reg = sm.add_constant(pd.concat([d, x], axis=1))
    ols = sm.OLS(y, X_reg).fit()
    return ols.params['D'], ols.bse['D']

def main():
    print("--- Day 2 Part C: Measurement Error Diagnostics ---")

    data_path = 'day2/data/speeches_with_sentiment.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run partA_sentiment.py first.")
        return

    df = pd.read_csv(data_path)
    D = df['D']
    X = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)

    # We treat the computed sentiment as the "true" latent outcome Y for this simulation
    # so we can inject known errors and see the effect.
    Y_true = df['Y_tilde']

    # 1. Baseline: Estimate ATE with "true" Y
    ate_true, se_true = estimate_ate(Y_true, D, X)
    print("\n1. Baseline (No Measurement Error)")
    print(f"ATE using true Y: {ate_true:.4f} (SE: {se_true:.4f})")

    # 2. Classical Measurement Error (MCAR)
    # Error is purely random noise, independent of D and X
    rng = np.random.default_rng(42)
    epsilon_mcar = rng.normal(loc=0, scale=1.0, size=len(df))
    Y_mcar = Y_true + epsilon_mcar

    ate_mcar, se_mcar = estimate_ate(Y_mcar, D, X)
    print("\n2. Classical Measurement Error (MCAR)")
    print("Error epsilon ~ N(0, 1) independent of D")
    print(f"ATE using Y_MCAR: {ate_mcar:.4f} (SE: {se_mcar:.4f})")
    print(f"Bias: {ate_mcar - ate_true:.4f}")
    print("Note: Estimate is unbiased, but standard error increases.")

    # 3. Non-classical Measurement Error (MNAR)
    # Error depends systematically on treatment D (e.g., LLM prompt sensitivity)
    # Suppose the LLM systematically overestimates sentiment for Republicans (D=1) by 0.5
    bias_term = 0.5
    epsilon_mnar = rng.normal(loc=0, scale=0.5, size=len(df)) + (bias_term * D)
    Y_mnar = Y_true + epsilon_mnar

    ate_mnar, se_mnar = estimate_ate(Y_mnar, D, X)
    print("\n3. Non-classical Measurement Error (MNAR)")
    print(f"Error epsilon ~ N(0, 0.5) + {bias_term}*D")
    print(f"ATE using Y_MNAR: {ate_mnar:.4f} (SE: {se_mnar:.4f})")
    print(f"Bias: {ate_mnar - ate_true:.4f}")
    print("Note: Estimate is heavily biased because E[epsilon | D=1] != E[epsilon | D=0].")

    print("\nConclusion:")
    print("If LLM measurement error is correlated with treatment (MNAR),")
    print("standard causal estimators will yield biased results.")

if __name__ == "__main__":
    main()
