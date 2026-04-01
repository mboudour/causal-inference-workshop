"""
Day 3 - Part C: Auditing LLMs as Causal Systems
This script implements the causal auditing framework from the Day 3 slides.

We simulate an LLM that maps text inputs to predictions, and audit whether
the model's outputs depend causally on a sensitive attribute (party affiliation).

Procedure:
  1. For each speech, create a modified version with the party label swapped
  2. Compute model predictions on original and modified inputs
  3. Compute individual-level causal effects: Delta_i = Y_hat_i - Y_hat_i'
  4. Compute Average Causal Bias (ACB) and its distribution
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def simulate_llm_model(X_train, Y_train):
    """Simulate an LLM-like model trained on text features."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    model = Ridge(alpha=0.5)
    model.fit(X_s, Y_train)
    return model, scaler

def build_features(texts, vectorizer=None, svd=None, n_components=30):
    """Build TF-IDF + SVD features from texts."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5)
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X = svd.fit_transform(tfidf)
    else:
        X = svd.transform(tfidf)

    return X, vectorizer, svd

def main():
    print("--- Day 3 Part C: Auditing LLMs as Causal Systems ---")

    data_path = 'day1/data/speeches_sample.csv'
    out_path  = 'day3/data/audit_results.csv'

    os.makedirs('day3/data', exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path).reset_index(drop=True)
    print(f"Loaded {len(df)} speeches.")

    rng = np.random.default_rng(2026)
    D = (df['party'] == 'Republican').astype(int).values
    n = len(df)

    # -----------------------------------------------------------------------
    # 1. Build features and simulate a "biased" LLM model
    # -----------------------------------------------------------------------
    # The model is trained on text + a party-signal feature,
    # so it learns to associate party with the outcome (simulating LLM bias)
    X_text, vectorizer, svd = build_features(df['text'].fillna(''), n_components=30)

    # True latent outcome (party effect = 0.3, plus text confounding)
    proj = rng.standard_normal(X_text.shape[1])
    f_X  = X_text @ proj
    f_X  = (f_X - f_X.mean()) / f_X.std()
    Y_true = 0.3 * D + f_X + rng.normal(0, 0.3, n)

    # Add a party-signal column to features (simulates LLM picking up party cues)
    party_signal = D.reshape(-1, 1) * 0.8 + rng.normal(0, 0.1, (n, 1))
    X_full = np.hstack([X_text, party_signal])

    model, scaler = simulate_llm_model(X_full, Y_true)
    Y_hat = model.predict(scaler.transform(X_full))

    print(f"\nModel trained. Mean prediction: {Y_hat.mean():.4f}")

    # -----------------------------------------------------------------------
    # 2. Create counterfactual inputs (swap party signal)
    # -----------------------------------------------------------------------
    # Intervention: flip the party signal for each observation
    party_signal_flipped = (1 - D).reshape(-1, 1) * 0.8 + rng.normal(0, 0.1, (n, 1))
    X_full_flipped = np.hstack([X_text, party_signal_flipped])

    Y_hat_prime = model.predict(scaler.transform(X_full_flipped))

    # -----------------------------------------------------------------------
    # 3. Individual-level causal effects
    # -----------------------------------------------------------------------
    Delta = Y_hat - Y_hat_prime

    ACB = Delta.mean()
    ACB_var = Delta.var(ddof=1)
    ACB_se  = Delta.std(ddof=1) / np.sqrt(n)
    ACB_ci  = (ACB - 1.96 * ACB_se, ACB + 1.96 * ACB_se)

    print(f"\nAverage Causal Bias (ACB): {ACB:.4f}")
    print(f"Variance of Delta_i:       {ACB_var:.4f}")
    print(f"95% CI for ACB:            [{ACB_ci[0]:.4f}, {ACB_ci[1]:.4f}]")

    # -----------------------------------------------------------------------
    # 4. Subgroup ACB
    # -----------------------------------------------------------------------
    ACB_R = Delta[D == 1].mean()
    ACB_D = Delta[D == 0].mean()
    print(f"\nSubgroup ACB (Republicans): {ACB_R:.4f}")
    print(f"Subgroup ACB (Democrats):   {ACB_D:.4f}")

    # -----------------------------------------------------------------------
    # 5. Prompt sensitivity simulation
    # -----------------------------------------------------------------------
    print("\n--- Prompt Sensitivity Simulation ---")
    prompt_results = []
    for prompt_name, noise_std, bias_mult in [
        ("Zero-shot (high noise)", 0.5, 1.0),
        ("Few-shot balanced",      0.1, 0.5),
        ("Chain-of-Thought",       0.05, 1.5),
    ]:
        # Simulate different prompts by varying the party signal noise and bias
        ps = D.reshape(-1, 1) * 0.8 * bias_mult + rng.normal(0, noise_std, (n, 1))
        X_p = np.hstack([X_text, ps])
        Y_p = model.predict(scaler.transform(X_p))

        ps_flip = (1 - D).reshape(-1, 1) * 0.8 * bias_mult + rng.normal(0, noise_std, (n, 1))
        X_p_flip = np.hstack([X_text, ps_flip])
        Y_p_flip = model.predict(scaler.transform(X_p_flip))

        delta_p = Y_p - Y_p_flip
        acb_p   = delta_p.mean()
        print(f"  {prompt_name:<35}: ACB = {acb_p:.4f}")
        prompt_results.append({'prompt': prompt_name, 'acb': acb_p})

    # -----------------------------------------------------------------------
    # 6. Save results
    # -----------------------------------------------------------------------
    df_out = df[['speaker', 'party', 'year']].copy()
    df_out['Y_hat']       = Y_hat
    df_out['Y_hat_prime'] = Y_hat_prime
    df_out['Delta_i']     = Delta
    df_out.to_csv(out_path, index=False)
    print(f"\nIndividual-level results saved to {out_path}")

    # Summary table
    summary = pd.DataFrame({
        'metric': ['ACB', 'Var(Delta_i)', 'ACB_SE', 'CI_lower', 'CI_upper',
                   'ACB_Republicans', 'ACB_Democrats'],
        'value':  [ACB, ACB_var, ACB_se, ACB_ci[0], ACB_ci[1], ACB_R, ACB_D],
    })
    summary_path = 'day3/data/audit_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    main()
