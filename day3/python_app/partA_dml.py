"""
Day 3 - Part A: Double Machine Learning (DML)
This script implements the DML estimator from the Day 3 slides.
We use the speeches dataset, treating the LLM sentiment score as the outcome
and party affiliation as the treatment. High-dimensional covariates are
constructed from TF-IDF text features (simulating text embeddings).

The DML procedure:
  1. Estimate nuisance functions g(X) = E[Y|X] and e(X) = P(D=1|X) via ML
  2. Compute residuals: Y_tilde = Y - g_hat(X),  D_tilde = D - e_hat(X)
  3. Regress Y_tilde on D_tilde to get the DML estimate of theta
  4. Apply K-fold cross-fitting to avoid overfitting bias
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_features(df, n_components=50):
    """Build high-dimensional text features via TF-IDF + SVD (simulating embeddings)."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    tfidf = vectorizer.fit_transform(df['text'].fillna(''))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X = svd.fit_transform(tfidf)
    return X

def dml_estimate(Y, D, X, n_splits=5, random_state=42):
    """
    Double Machine Learning estimator with cross-fitting.
    Returns: theta_hat, se_hat, (ci_lower, ci_upper)
    """
    n = len(Y)
    Y = np.array(Y)
    D = np.array(D)

    Y_res = np.zeros(n)
    D_res = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scaler = StandardScaler()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        D_train = D[train_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Estimate g(X) = E[Y | X] via Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, Y_train)
        g_hat = ridge.predict(X_test_s)

        # Estimate e(X) = P(D=1 | X) via Logistic regression
        logit = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)
        logit.fit(X_train_s, D_train)
        e_hat = logit.predict_proba(X_test_s)[:, 1]
        e_hat = np.clip(e_hat, 0.05, 0.95)

        Y_res[test_idx] = Y[test_idx] - g_hat
        D_res[test_idx] = D[test_idx] - e_hat

    # Final DML estimate: regress Y_res on D_res
    theta_hat = np.sum(D_res * Y_res) / np.sum(D_res ** 2)

    # Influence-function standard error
    psi = D_res * (Y_res - theta_hat * D_res)
    se_hat = np.sqrt(np.mean(psi ** 2) / (np.mean(D_res ** 2) ** 2) / n)

    ci_lower = theta_hat - 1.96 * se_hat
    ci_upper = theta_hat + 1.96 * se_hat

    return theta_hat, se_hat, (ci_lower, ci_upper)


def main():
    print("--- Day 3 Part A: Double Machine Learning (DML) ---")

    data_path = 'day1/data/speeches_sample.csv'
    out_path  = 'day3/data/dml_results.csv'

    os.makedirs('day3/data', exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} speeches.")

    # Construct outcome Y (LLM sentiment proxy via deterministic projection)
    rng = np.random.default_rng(2026)
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    tv = TV(max_features=500, min_df=5)
    tfidf_small = tv.fit_transform(df['text'].fillna('')).toarray()
    proj = rng.standard_normal(tfidf_small.shape[1])
    raw = tfidf_small @ proj
    Y = (raw - raw.mean()) / raw.std()

    D = (df['party'] == 'Republican').astype(int).values

    print("\nBuilding high-dimensional text features (TF-IDF + SVD)...")
    X = build_features(df, n_components=50)
    print(f"Feature matrix shape: {X.shape}")

    # Naive difference-in-means (no adjustment)
    naive_ate = Y[D == 1].mean() - Y[D == 0].mean()
    print(f"\nNaive Difference-in-Means: {naive_ate:.4f}")

    # Simple OLS with year dummies (baseline)
    year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)
    X_ols = sm.add_constant(pd.concat([pd.Series(D, name='D'), year_dummies], axis=1))
    ols = sm.OLS(Y, X_ols).fit()
    ols_ate = ols.params['D']
    ols_se  = ols.bse['D']
    print(f"OLS (year dummies only): {ols_ate:.4f} (SE: {ols_se:.4f})")

    # DML with K=5 cross-fitting
    print("\nRunning DML with 5-fold cross-fitting...")
    theta_dml, se_dml, ci_dml = dml_estimate(Y, D, X, n_splits=5)
    print(f"DML Estimate (theta): {theta_dml:.4f}")
    print(f"Standard Error:       {se_dml:.4f}")
    print(f"95% CI:               [{ci_dml[0]:.4f}, {ci_dml[1]:.4f}]")

    # Save results
    results = pd.DataFrame({
        'estimator': ['Naive DiM', 'OLS (year dummies)', 'DML (text features, K=5)'],
        'estimate':  [naive_ate, ols_ate, theta_dml],
        'se':        [np.nan, ols_se, se_dml],
        'ci_lower':  [np.nan, ols_ate - 1.96*ols_se, ci_dml[0]],
        'ci_upper':  [np.nan, ols_ate + 1.96*ols_se, ci_dml[1]],
    })
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print("\nSummary:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
