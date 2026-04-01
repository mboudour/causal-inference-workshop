"""
Day 3 - Part B: Design-Based Supervised Learning (DSL)
This script implements the DSL measurement-correction estimator from the Day 3 slides.

Setup:
  - We simulate a "true" outcome Y_i and a "measured" outcome Y_tilde_i = Y_i + epsilon_i
    where epsilon_i is non-classical (depends on treatment D_i).
  - A small labeled sample L has (Y_i, Y_tilde_i, D_i, X_i).
  - A large unlabeled sample U has only (Y_tilde_i, D_i, X_i).

DSL procedure:
  1. On L, learn the correction function m(X, D, Y_tilde) = E[Y | X, D, Y_tilde]
  2. Apply m to U to get corrected outcomes Y_hat_corr
  3. Estimate ATE using Y_hat_corr on U
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_features(df, n_components=30):
    """Build text features via TF-IDF + SVD."""
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5)
    tfidf = vectorizer.fit_transform(df['text'].fillna(''))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X = svd.fit_transform(tfidf)
    return X

def main():
    print("--- Day 3 Part B: Design-Based Supervised Learning (DSL) ---")

    data_path = 'day1/data/speeches_sample.csv'
    out_path  = 'day3/data/dsl_results.csv'

    os.makedirs('day3/data', exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} speeches.")

    # -----------------------------------------------------------------------
    # 1. Simulate true Y and measured Y_tilde with non-classical error
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(2026)
    D = (df['party'] == 'Republican').astype(int).values
    n = len(df)

    # Build text features for simulation
    X_feat = build_features(df, n_components=30)

    # True latent outcome: Y = 0.3*D + f(X) + noise
    proj = rng.standard_normal(X_feat.shape[1])
    f_X  = X_feat @ proj
    f_X  = (f_X - f_X.mean()) / f_X.std()
    Y_true = 0.3 * D + f_X + rng.normal(0, 0.5, n)

    # Non-classical measurement error: epsilon depends on D
    # LLM over-scores Republicans by 0.5 on average
    epsilon = rng.normal(0, 0.5, n) + 0.5 * D
    Y_tilde = Y_true + epsilon

    # -----------------------------------------------------------------------
    # 2. Split into labeled (L) and unlabeled (U) samples
    # -----------------------------------------------------------------------
    label_frac = 0.1  # 10% labeled
    labeled_idx = rng.choice(n, size=int(n * label_frac), replace=False)
    unlabeled_idx = np.setdiff1d(np.arange(n), labeled_idx)

    print(f"\nLabeled sample size:   {len(labeled_idx)}")
    print(f"Unlabeled sample size: {len(unlabeled_idx)}")

    # -----------------------------------------------------------------------
    # 3. Naive ATE on U (using biased Y_tilde, no correction)
    # -----------------------------------------------------------------------
    Y_u = Y_tilde[unlabeled_idx]
    D_u = D[unlabeled_idx]
    naive_ate = Y_u[D_u == 1].mean() - Y_u[D_u == 0].mean()
    print(f"\nNaive ATE on U (biased Y_tilde): {naive_ate:.4f}")
    print(f"True ATE (known in simulation):  0.3000")

    # -----------------------------------------------------------------------
    # 4. DSL: Learn correction m(X, D, Y_tilde) on L, apply to U
    # -----------------------------------------------------------------------
    # Features for correction model: [X_text, D, Y_tilde]
    Y_L = Y_true[labeled_idx]
    Y_tilde_L = Y_tilde[labeled_idx].reshape(-1, 1)
    D_L = D[labeled_idx].reshape(-1, 1)
    X_L = X_feat[labeled_idx]

    X_tilde_L = np.hstack([X_L, D_L, Y_tilde_L])

    scaler = StandardScaler()
    X_tilde_L_s = scaler.fit_transform(X_tilde_L)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tilde_L_s, Y_L)

    # Apply correction to U
    Y_tilde_U = Y_tilde[unlabeled_idx].reshape(-1, 1)
    D_U = D[unlabeled_idx].reshape(-1, 1)
    X_U = X_feat[unlabeled_idx]
    X_tilde_U = np.hstack([X_U, D_U, Y_tilde_U])
    X_tilde_U_s = scaler.transform(X_tilde_U)

    Y_corr_U = ridge.predict(X_tilde_U_s)

    # ATE using corrected outcomes
    dsl_ate = Y_corr_U[D_u == 1].mean() - Y_corr_U[D_u == 0].mean()
    print(f"DSL-corrected ATE on U:          {dsl_ate:.4f}")

    # -----------------------------------------------------------------------
    # 5. Oracle ATE (using true Y on U — for reference only)
    # -----------------------------------------------------------------------
    Y_true_U = Y_true[unlabeled_idx]
    oracle_ate = Y_true_U[D_u == 1].mean() - Y_true_U[D_u == 0].mean()
    print(f"Oracle ATE on U (true Y):        {oracle_ate:.4f}")

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    print("\nSummary:")
    print(f"{'Estimator':<35} {'ATE':>8} {'Bias vs Oracle':>15}")
    print("-" * 60)
    for name, ate in [
        ("Naive (biased Y_tilde)", naive_ate),
        ("DSL-corrected", dsl_ate),
        ("Oracle (true Y)", oracle_ate),
    ]:
        print(f"{name:<35} {ate:>8.4f} {ate - oracle_ate:>15.4f}")

    # Save
    results = pd.DataFrame({
        'estimator': ['Naive (biased Y_tilde)', 'DSL-corrected', 'Oracle (true Y)'],
        'ate':       [naive_ate, dsl_ate, oracle_ate],
        'bias':      [naive_ate - oracle_ate, dsl_ate - oracle_ate, 0.0],
    })
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
