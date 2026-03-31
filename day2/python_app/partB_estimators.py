"""
Day 2 - Part B: Causal Estimators
This script implements the core estimators from the Day 2 slides:
1. Difference-in-Means (with standard errors)
2. Regression Adjustment
3. G-formula (Plug-in)
4. Nearest-Neighbour Matching
5. Inverse Probability Weighting (IPW)
6. Augmented IPW (Doubly Robust)
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats

def compute_diff_in_means(df, y_col, d_col):
    """Difference in means with standard error."""
    treated = df[df[d_col] == 1][y_col]
    control = df[df[d_col] == 0][y_col]

    n1, n0 = len(treated), len(control)
    mean1, mean0 = treated.mean(), control.mean()
    var1, var0 = treated.var(ddof=1), control.var(ddof=1)

    estimate = mean1 - mean0
    se = np.sqrt(var1/n1 + var0/n0)

    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    return estimate, se, (ci_lower, ci_upper)

def main():
    print("--- Day 2 Part B: Causal Estimators ---")

    data_path = 'day2/data/speeches_with_sentiment.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run partA_sentiment.py first.")
        return

    df = pd.read_csv(data_path)

    Y = df['Y_tilde']  # LLM sentiment score
    D = df['D']        # Party (1=Rep, 0=Dem)

    # Confounders X: year dummies
    year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)
    X = year_dummies

    print("\n1. Difference-in-Means Estimator")
    est_diff, se_diff, ci_diff = compute_diff_in_means(df, 'Y_tilde', 'D')
    print(f"ATE Estimate: {est_diff:.4f}")
    print(f"Standard Error: {se_diff:.4f}")
    print(f"95% CI: [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")

    print("\n2. Regression Adjustment")
    X_reg = sm.add_constant(pd.concat([D, X], axis=1))
    ols = sm.OLS(Y, X_reg).fit()
    est_reg = ols.params['D']
    se_reg  = ols.bse['D']
    ci_reg  = ols.conf_int().loc['D'].values
    print(f"ATE Estimate: {est_reg:.4f}")
    print(f"Standard Error: {se_reg:.4f}")
    print(f"95% CI: [{ci_reg[0]:.4f}, {ci_reg[1]:.4f}]")

    print("\n3. G-formula (Plug-in Estimator)")
    X_gform = sm.add_constant(pd.concat([D, X], axis=1))
    model_m = sm.OLS(Y, X_gform).fit()

    df1 = X_gform.copy(); df1['D'] = 1
    df0 = X_gform.copy(); df0['D'] = 0

    m1_hat = model_m.predict(df1)
    m0_hat = model_m.predict(df0)

    est_gform = np.mean(m1_hat - m0_hat)
    print(f"ATE Estimate: {est_gform:.4f}")

    print("\n4. Nearest-Neighbour Matching (ATT)")
    treated_X = X[D == 1].values
    control_X = X[D == 0].values

    nn = NearestNeighbors(n_neighbors=1).fit(control_X)
    distances, indices = nn.kneighbors(treated_X)

    treated_Y = Y[D == 1].values
    control_Y = Y[D == 0].values
    matched_control_Y = control_Y[indices.flatten()]

    est_match = np.mean(treated_Y - matched_control_Y)
    print(f"ATT Estimate: {est_match:.4f}")
    print("Note: Matching is performed on year dummies only. With a single weak")
    print("covariate, matched pairs may be poorly balanced. In practice, matching")
    print("should be performed on richer covariates (e.g., text-based features).")

    print("\n5. Inverse Probability Weighting (IPW)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logit = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    logit.fit(X_scaled, D)
    e_hat = logit.predict_proba(X_scaled)[:, 1]
    e_hat = np.clip(e_hat, 0.05, 0.95)

    w1 = D / e_hat
    w0 = (1 - D) / (1 - e_hat)
    est_ipw = np.sum(Y * w1) / np.sum(w1) - np.sum(Y * w0) / np.sum(w0)
    print(f"ATE Estimate: {est_ipw:.4f}")

    print("\n6. Augmented IPW (Doubly Robust)")
    term1 = m1_hat - m0_hat
    term2 = (D * (Y - m1_hat)) / e_hat
    term3 = ((1 - D) * (Y - m0_hat)) / (1 - e_hat)

    aipw_scores = term1 + term2 - term3
    est_aipw = np.mean(aipw_scores)
    se_aipw  = np.std(aipw_scores, ddof=1) / np.sqrt(len(Y))
    ci_aipw  = (est_aipw - 1.96 * se_aipw, est_aipw + 1.96 * se_aipw)

    print(f"ATE Estimate: {est_aipw:.4f}")
    print(f"Standard Error: {se_aipw:.4f}")
    print(f"95% CI: [{ci_aipw[0]:.4f}, {ci_aipw[1]:.4f}]")

    print("\nSummary of ATE Estimates:")
    print("-" * 30)
    print(f"Diff-in-Means : {est_diff:8.4f}")
    print(f"Regression    : {est_reg:8.4f}")
    print(f"G-formula     : {est_gform:8.4f}")
    print(f"Matching (ATT): {est_match:8.4f}")
    print(f"IPW           : {est_ipw:8.4f}")
    print(f"AIPW          : {est_aipw:8.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
