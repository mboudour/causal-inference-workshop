# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Causal Inference with LLMs — Day 3 NZ Interactive App
Moses Boudourides | Northwestern University
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Causal Inference with LLMs — Day 3 NZ",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Causal Inference with LLMs — Day 3 NZ")
st.markdown(
    "**Workshop companion app.** This page mirrors Day 3 using a New Zealand parliamentary corpus. "
    "The intended built-in dataset comes from **ParlSpeech V2** and the app displays the original NZ "
    "party names: **National** and **Labour**. Internally, treatment is coded as `D=1` for National "
    "and `D=0` for Labour for comparability with the workshop estimators."
)
st.markdown("---")


@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    primary = os.path.join(base, "..", "day3_nz", "data", "llm", "speeches_sample.csv")
    fallback = os.path.join(base, "..", "day1_nz", "data", "llm", "speeches_sample.csv")

    if os.path.exists(primary):
        return pd.read_csv(primary), "day3_nz/data/llm/speeches_sample.csv"
    if os.path.exists(fallback):
        return pd.read_csv(fallback), "day1_nz/data/llm/speeches_sample.csv (fallback)"

    raise FileNotFoundError(
        "Could not find NZ speeches_sample.csv in day3_nz/data/llm/ or day1_nz/data/llm/."
    )


@st.cache_data
def compute_tfidf_features(texts, n_components=50):
    vec = TfidfVectorizer(max_features=5000, min_df=5, sublinear_tf=True)
    tfidf = vec.fit_transform(texts)
    max_components = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    n_components = max(1, min(n_components, max_components))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(tfidf)


@st.cache_data
def compute_outcome(texts):
    vec = TfidfVectorizer(max_features=5000, min_df=5)
    tfidf = vec.fit_transform(texts)
    rng = np.random.default_rng(2026)
    proj = rng.standard_normal(tfidf.shape[1])
    raw = np.asarray(tfidf @ proj).flatten()
    sd = raw.std()
    if sd == 0:
        return np.zeros(len(raw))
    return (raw - raw.mean()) / sd


def make_year_dummies(df):
    X = pd.get_dummies(df["year"].astype(str), prefix="yr", drop_first=True).astype(float)
    if X.shape[1] == 0:
        return pd.DataFrame({"yr_single": np.zeros(len(df))}, index=df.index)
    return X


party_to_internal = {
    "National": "Republican",
    "Labour": "Democrat",
    "Republican": "Republican",
    "Democrat": "Democrat",
}
party_to_display = {
    "Republican": "National",
    "Democrat": "Labour",
    "National": "National",
    "Labour": "Labour",
}

try:
    df_raw, source_path = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

required_cols = {"text", "party", "year"}
if not required_cols.issubset(df_raw.columns):
    st.error(f"Dataset must contain columns: {required_cols}. Found: {set(df_raw.columns)}")
    st.stop()

df_raw["party_internal"] = df_raw["party"].map(party_to_internal)
df_raw = df_raw[df_raw["party_internal"].isin(["Republican", "Democrat"])].copy()
df_raw["party"] = df_raw["party_internal"].map(party_to_display)
df_raw["D"] = (df_raw["party_internal"] == "Republican").astype(int)

st.header("0 · Data")
st.info(f"Using built-in NZ dataset: {len(df_raw):,} speeches from `{source_path}`.")
with st.expander("Preview data"):
    cols = [c for c in ["speech_id", "speaker", "party", "year", "text"] if c in df_raw.columns]
    st.dataframe(df_raw[cols].head(10))
col1, col2, col3 = st.columns(3)
col1.metric("Total speeches", f"{len(df_raw):,}")
col2.metric("National", f"{df_raw['D'].sum():,}")
col3.metric("Labour", f"{(1-df_raw['D']).sum():,}")
st.markdown("---")

# ── Section DAGs: Causal Structure Diagrams ──────────────────────────────────
st.header("DAGs · Causal Structure Diagrams")

st.markdown(
    """
These three directed acyclic graphs (DAGs) illustrate the causal structure underlying Day 3 NZ.

- **DAG 1 — DML:** high-dimensional covariates X confound both treatment D and outcome Ỹ.
  ML nuisance models partial out confounding and DML recovers the ATE from orthogonalized residuals.
- **DAG 2 — DSL:** a correction model is trained on a labeled subset where the true outcome is observed,
  then applied to the larger unlabeled sample to reduce LLM measurement error.
- **DAG 3 — Causal Auditing:** party enters the measurement system as a sensitive attribute, and
  counterfactual outputs under party-swapped inputs are compared to quantify Average Causal Bias (ACB).
"""
)

_dag_primary = os.path.join(os.path.dirname(__file__), "..", "day3_nz", "data", "dags")
_dag_fallback = os.path.join(os.path.dirname(__file__), "..", "day3", "data", "dags")
_dag_dir3 = _dag_primary if os.path.exists(_dag_primary) else _dag_fallback
_dag_files3 = {
    "DAG 1 — DML: High-Dim Confounding": os.path.join(_dag_dir3, "dag1_dml.png"),
    "DAG 2 — DSL: Measurement Bridge": os.path.join(_dag_dir3, "dag2_dsl.png"),
    "DAG 3 — Causal Auditing (ACB)": os.path.join(_dag_dir3, "dag3_auditing.png"),
}

for _title, _path in _dag_files3.items():
    st.markdown(f"**{_title}**")
    if os.path.exists(_path):
        st.image(_path, use_container_width=True)
    else:
        st.info(f"DAG image not found at {_path}.")

st.markdown("---")

# ── Section 1: DML ───────────────────────────────────────────────────────────
st.header("1 · Double Machine Learning (DML)")
st.markdown(
    r"""
**Slide reference:** *Partially Linear Model*, *DML: Orthogonalization*, *Cross-Fitting*, *Day 3*

DML estimates the causal parameter $\theta$ in the partially linear model:
$$Y_i = \theta D_i + g(X_i) + \varepsilon_i, \quad \mathbb{E}[\varepsilon_i \mid D_i, X_i] = 0$$

The procedure uses **cross-fitting** over $K$ folds:

1. For each fold, estimate nuisance functions $\hat{g}^{(-k)}$ and $\hat{e}^{(-k)}$ on the other folds.
2. Compute residuals $R^Y_i = Y_i - \hat{g}^{(-k)}(X_i)$ and $R^D_i = D_i - \hat{e}^{(-k)}(X_i)$.
3. Regress $R^Y_i$ on $R^D_i$ to obtain $\hat{\theta}$.

Covariates $X_i$: TF-IDF + SVD text features. Nuisance models: Ridge for $g$, Logistic for $e$.
"""
)

n_folds = st.slider("Number of cross-fitting folds (K)", 2, 10, 5)
n_components = st.slider("SVD components (text features)", 10, 100, 50, 10)

if st.button("Run Section 1 — DML Estimation"):
    with st.spinner("Computing TF-IDF features and running DML..."):
        df = df_raw.copy()
        Y = compute_outcome(df["text"].tolist())
        D = df["D"].values
        X = compute_tfidf_features(df["text"].tolist(), n_components=n_components)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        R_Y = np.zeros(len(Y))
        R_D = np.zeros(len(D))

        for train_idx, test_idx in kf.split(X):
            ridge_g = Ridge(alpha=1.0)
            ridge_g.fit(X[train_idx], Y[train_idx])
            R_Y[test_idx] = Y[test_idx] - ridge_g.predict(X[test_idx])

            scaler = StandardScaler()
            logit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            logit.fit(scaler.fit_transform(X[train_idx]), D[train_idx])
            e_hat = logit.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            e_hat = np.clip(e_hat, 0.05, 0.95)
            R_D[test_idx] = D[test_idx] - e_hat

        theta_dml = np.sum(R_D * R_Y) / np.sum(R_D ** 2)
        psi = R_D * (R_Y - theta_dml * R_D)
        se_dml = np.sqrt(np.mean(psi ** 2) / (np.mean(R_D ** 2) ** 2) / len(Y))
        ci_lo, ci_hi = theta_dml - 1.96 * se_dml, theta_dml + 1.96 * se_dml

        Xr = sm.add_constant(pd.concat([pd.Series(D, name="D"), make_year_dummies(df)], axis=1))
        ols = sm.OLS(Y, Xr).fit()
        theta_ols = ols.params["D"]
        se_ols = ols.bse["D"]

    col1, col2 = st.columns(2)
    col1.metric("DML estimate theta", f"{theta_dml:+.4f}", f"SE: {se_dml:.4f}")
    col2.metric("Naive OLS estimate", f"{theta_ols:+.4f}", f"SE: {se_ols:.4f}")
    st.markdown(f"**95% CI (DML):** [{ci_lo:.4f}, {ci_hi:.4f}]")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    estimators = ["Naive OLS\n(year dummies)", f"DML\n(K={n_folds}, d={n_components})"]
    ests = [theta_ols, theta_dml]
    ses = [se_ols, se_dml]
    ax.barh(estimators, ests, xerr=[1.96 * s for s in ses],
            color=["#e67e22", "#2980b9"], alpha=0.8, capsize=5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Estimated ATE")
    ax.set_title("DML vs Naive OLS")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.scatter(R_D, R_Y, alpha=0.15, s=8, color="#2980b9")
    x_line = np.linspace(R_D.min(), R_D.max(), 100)
    ax2.plot(x_line, theta_dml * x_line, color="#c0392b", linewidth=2,
             label=f"theta = {theta_dml:+.4f}")
    ax2.set_xlabel("R^D_i (residualized treatment)")
    ax2.set_ylabel("R^Y_i (residualized outcome)")
    ax2.set_title("DML: Residual-on-Residual Regression")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# ── Section 2: DSL ───────────────────────────────────────────────────────────
st.header("2 · Design-Based Supervised Learning (DSL)")
st.markdown(
    r"""
**Slide reference:** *Design-Based Supervised Learning (DSL)*, *DSL: Identification and Estimation*, *Day 3*

DSL corrects LLM measurement error using a small **labeled sample** $\mathcal{L}$ where the
true outcome is observed, together with a large **unlabeled sample** $\mathcal{U}$ where only
proxy measurements are available.

**Procedure:**
1. Fit a correction model on $\mathcal{L}$.
2. Apply it to $\mathcal{U}$ to obtain corrected outcomes.
3. Estimate the ATE using the corrected outcomes.

We simulate MNAR error through a treatment-dependent perturbation of the proxy outcome.
"""
)

labeled_frac = st.slider("Labeled sample fraction (|L| / n)", 0.05, 0.5, 0.1, 0.05)
mnar_delta = st.slider("MNAR bias delta (treatment-dependent error)", 0.0, 2.0, 0.5, 0.1)

if st.button("Run Section 2 — DSL Estimation"):
    with st.spinner("Running DSL..."):
        df = df_raw.copy()
        Y_tilde = compute_outcome(df["text"].tolist())
        D = df["D"].values
        X_feat = compute_tfidf_features(df["text"].tolist(), n_components=50)

        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.3, len(df))
        Y_true = Y_tilde + noise + mnar_delta * D

        n = len(df)
        n_labeled = max(1, int(n * labeled_frac))
        labeled_idx = rng.choice(n, size=n_labeled, replace=False)
        unlabeled_idx = np.setdiff1d(np.arange(n), labeled_idx)

        feat_L = np.column_stack([X_feat[labeled_idx], D[labeled_idx], Y_tilde[labeled_idx]])
        feat_U = np.column_stack([X_feat[unlabeled_idx], D[unlabeled_idx], Y_tilde[unlabeled_idx]])

        scaler = StandardScaler()
        ridge_m = Ridge(alpha=1.0)
        ridge_m.fit(scaler.fit_transform(feat_L), Y_true[labeled_idx])
        Y_corr = ridge_m.predict(scaler.transform(feat_U))

        D_U = D[unlabeled_idx]
        ate_dsl = Y_corr[D_U == 1].mean() - Y_corr[D_U == 0].mean()
        n1, n0 = D_U.sum(), (1 - D_U).sum()
        se_dsl = np.sqrt(Y_corr[D_U == 1].var(ddof=1) / n1 + Y_corr[D_U == 0].var(ddof=1) / n0)
        ate_naive = Y_tilde[D == 1].mean() - Y_tilde[D == 0].mean()
        ate_oracle = Y_true[D == 1].mean() - Y_true[D == 0].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Naive ATE (proxy Y-tilde)", f"{ate_naive:+.4f}")
    col2.metric("DSL-corrected ATE", f"{ate_dsl:+.4f}", f"SE: {se_dsl:.4f}")
    col3.metric("Oracle ATE (true Y)", f"{ate_oracle:+.4f}")
    st.markdown(
        f"With delta = {mnar_delta}, naive bias = **{ate_naive - ate_oracle:+.4f}**. "
        f"DSL residual bias = **{ate_dsl - ate_oracle:+.4f}** using {n_labeled:,} labeled observations."
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Naive\n(proxy Y-tilde)", "DSL\n(corrected)", "Oracle\n(true Y)"],
           [ate_naive, ate_dsl, ate_oracle],
           color=["#e67e22", "#2980b9", "#27ae60"], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Estimated ATE")
    ax.set_title("DSL correction under treatment-dependent measurement error")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Section 3: Causal Auditing ───────────────────────────────────────────────
st.header("3 · Causal Auditing of LLMs")
st.markdown(
    r"""
**Slide reference:** *Auditing as a Causal Problem*, *Interventions on Inputs*,
*Individual-Level Causal Effect*, *Average Causal Bias*, *Day 3*

We audit the LLM measurement operator for **party bias**: does the model score a speech
differently when it detects **National** versus **Labour** signals?

**Procedure:**
1. For each speech $W_i$, construct a counterfactual $W_i'$ by swapping party-associated signals.
2. Compute the individual causal effect $\delta_i = f_{\text{LLM}}(W_i) - f_{\text{LLM}}(W_i')$.
3. Estimate the **Average Causal Bias (ACB)** as the mean of $\delta_i$.

We simulate the counterfactual by applying a treatment-sign shift to the TF-IDF-based proxy score.
"""
)

if st.button("Run Section 3 — Causal Auditing"):
    with st.spinner("Computing causal audit..."):
        df = df_raw.copy()
        Y_tilde = compute_outcome(df["text"].tolist())
        D = df["D"].values
        rng = np.random.default_rng(7)
        shift = rng.normal(0, 0.15, len(df))
        Y_counter = Y_tilde - 0.3 * (2 * D - 1) + shift
        delta_i = Y_tilde - Y_counter
        acb = delta_i.mean()
        acb_national = delta_i[D == 1].mean()
        acb_labour = delta_i[D == 0].mean()
        se_acb = delta_i.std(ddof=1) / np.sqrt(len(delta_i))

    col1, col2, col3 = st.columns(3)
    col1.metric("ACB (overall)", f"{acb:+.4f}", f"SE: {se_acb:.4f}")
    col2.metric("ACB — National", f"{acb_national:+.4f}")
    col3.metric("ACB — Labour", f"{acb_labour:+.4f}")
    st.markdown(
        f"**ACB = {acb:+.4f}** (SE = {se_acb:.4f}). "
        "A non-zero ACB indicates that the measurement system responds differently to party-associated signals."
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(delta_i, bins=40, color="#8e44ad", alpha=0.8, density=True)
    axes[0].axvline(acb, color="#c0392b", linewidth=2, linestyle="--",
                    label=f"ACB = {acb:+.4f}")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("delta_i = f_LLM(W_i) - f_LLM(W_i')")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Individual Causal Effects")
    axes[0].legend()
    axes[1].bar(["National", "Labour"], [acb_national, acb_labour],
                color=["#c0392b", "#2980b9"], alpha=0.85, width=0.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Subgroup ACB")
    axes[1].set_title("ACB by Party")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Section 4: Compare Days ──────────────────────────────────────────────────
st.header("4 · Compare ATE Estimates Across NZ Days")
st.markdown(
    r"""
**Slide reference:** *Key Takeaways*, *Day 3*

Comparing OLS-style adjustment from Day 1 NZ / Day 2 NZ with DML from Day 3 NZ shows how
controlling for high-dimensional text features can change the estimated causal effect.
"""
)

if st.button("Run Section 4 — Cross-Day Comparison"):
    with st.spinner("Computing all estimates..."):
        df = df_raw.copy()
        Y = compute_outcome(df["text"].tolist())
        D = df["D"].values
        X_feat = compute_tfidf_features(df["text"].tolist(), n_components=50)
        Xr = sm.add_constant(pd.concat([pd.Series(D, name="D"), make_year_dummies(df)], axis=1))
        ols = sm.OLS(Y, Xr).fit()
        ate_ols = ols.params["D"]
        se_ols = ols.bse["D"]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        R_Y = np.zeros(len(Y))
        R_D = np.zeros(len(D))
        for tr, te in kf.split(X_feat):
            ridge_g = Ridge(alpha=1.0)
            ridge_g.fit(X_feat[tr], Y[tr])
            R_Y[te] = Y[te] - ridge_g.predict(X_feat[te])
            scaler = StandardScaler()
            logit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            logit.fit(scaler.fit_transform(X_feat[tr]), D[tr])
            e_hat = np.clip(logit.predict_proba(scaler.transform(X_feat[te]))[:, 1], 0.05, 0.95)
            R_D[te] = D[te] - e_hat

        ate_dml = np.sum(R_D * R_Y) / np.sum(R_D ** 2)
        psi = R_D * (R_Y - ate_dml * R_D)
        se_dml = np.sqrt(np.mean(psi ** 2) / (np.mean(R_D ** 2) ** 2) / len(Y))

    rows = [
        {"Day": "Day 1/2 NZ — OLS (year dummies)", "ATE": f"{ate_ols:+.4f}", "SE": f"{se_ols:.4f}"},
        {"Day": "Day 3 NZ — DML (text features, K=5)", "ATE": f"{ate_dml:+.4f}", "SE": f"{se_dml:.4f}"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [r["Day"] for r in rows]
    ests = [float(r["ATE"]) for r in rows]
    ses = [float(r["SE"]) for r in rows]
    ax.barh(names, ests, xerr=[1.96 * s for s in ses],
            color=["#e67e22", "#2980b9"], alpha=0.85, capsize=5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Estimated ATE")
    ax.set_title("ATE Estimates: OLS vs DML")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.caption(
    "Moses Boudourides · Northwestern University · "
    "Causal Inference with LLMs Workshop · Day 3 NZ"
)
