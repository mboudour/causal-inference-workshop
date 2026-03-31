# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- LOAD DATA ---
df = pd.read_csv("day1/data/llm/speeches_sample.csv")

# --- OUTCOME AND TREATMENT ---
df["Y"] = df["text"].str.len()
df["D"] = (df["party"] == "Republican").astype(int)

print("--- PART C: CAUSAL ADJUSTMENT & DIAGNOSTICS ---\n")

# 1. NAIVE ESTIMATOR DECOMPOSITION
mu1 = df[df["D"] == 1]["Y"].mean()
mu0 = df[df["D"] == 0]["Y"].mean()
naive_ate = mu1 - mu0
print(f"Naive difference (Y|D=1 - Y|D=0): {naive_ate:.2f}")

# 2. PROPENSITY SCORES & OVERLAP CHECK
year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
X_ps = year_dummies.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ps)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, df["D"])
df["ps"] = lr.predict_proba(X_scaled)[:, 1]

print(f"\nPropensity score range: [{df['ps'].min():.4f}, {df['ps'].max():.4f}]")
if df["ps"].min() > 0.01 and df["ps"].max() < 0.99:
    print("Overlap assumption appears satisfied.")
else:
    print("Warning: Possible overlap violation (ps close to 0 or 1).")

# 3. ADJUSTMENT FORMULA (G-FORMULA)
# Build a single design matrix with the correct column structure,
# then create counterfactual copies by setting D=1 or D=0.
X_base = pd.concat([df[["D"]], year_dummies], axis=1).astype(float)
X_reg = sm.add_constant(X_base, has_constant="add")
ols = sm.OLS(df["Y"], X_reg).fit()

X_base_1 = X_base.copy(); X_base_1["D"] = 1.0
X_base_0 = X_base.copy(); X_base_0["D"] = 0.0
X1 = sm.add_constant(X_base_1, has_constant="add")
X0 = sm.add_constant(X_base_0, has_constant="add")

mu1_hat = ols.predict(X1)
mu0_hat = ols.predict(X0)
ate_gformula = (mu1_hat - mu0_hat).mean()
print(f"\nATE via Adjustment Formula (G-formula): {ate_gformula:.2f}")

# 4. INVERSE PROBABILITY WEIGHTING (IPW)
eps = 1e-6
ps_clipped = df["ps"].clip(eps, 1 - eps)
ate_ipw = (
    (df["D"] * df["Y"] / ps_clipped).mean() -
    ((1 - df["D"]) * df["Y"] / (1 - ps_clipped)).mean()
)
print(f"ATE via IPW: {ate_ipw:.2f}")

# 5. MEASUREMENT ERROR DIAGNOSTIC (requires Part B output)
stance_path = "day1/data/llm/speeches_with_stance.csv"
if os.path.exists(stance_path):
    df_llm = pd.read_csv(stance_path)
    if "Y_tilde" in df_llm.columns:
        df["Y_tilde"] = df_llm["Y_tilde"]
        df_clean = df.dropna(subset=["Y_tilde"])

        corr = np.corrcoef(df_clean["Y_tilde"], df_clean["D"])[0, 1]
        print(f"\nMeasurement Error Diagnostic:")
        print(f"Correlation between LLM stance proxy (Y_tilde) and Treatment (D): {corr:.4f}")
        if abs(corr) > 0.05:
            print("-> Evidence of non-classical measurement error (proxy is correlated with treatment).")

        ate_llm_naive = (
            df_clean[df_clean["D"] == 1]["Y_tilde"].mean() -
            df_clean[df_clean["D"] == 0]["Y_tilde"].mean()
        )
        X_llm = sm.add_constant(pd.concat(
            [df_clean[["D"]], pd.get_dummies(df_clean["year"], prefix="yr", drop_first=True)],
            axis=1
        ).astype(float))
        ate_llm_adj = sm.OLS(df_clean["Y_tilde"], X_llm).fit().params["D"]

        print(f"Naive LLM ATE: {ate_llm_naive:.4f}")
        print(f"Adjusted LLM ATE: {ate_llm_adj:.4f}")
    else:
        print("\n(speeches_with_stance.csv found but Y_tilde column missing — re-run partB first.)")
else:
    print("\n(Run partB_llm_measurement.py first to generate speeches_with_stance.csv)")
