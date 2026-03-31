# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Causal Inference with LLMs — Day 2 Interactive App
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Causal Inference with LLMs — Day 2",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Causal Inference with LLMs — Day 2")
st.markdown(
    "**Workshop companion app.** Day 2 covers causal estimators and LLM measurement error. "
    "Work through each section in order using the built-in dataset "
    "(U.S. Congressional speeches, Hein Bound 111)."
)
st.markdown("---")

# ── Section 0: Data ──────────────────────────────────────────────────────────
st.header("0 · Data")

st.markdown(
    "The built-in dataset is a sample of **U.S. Congressional speeches** "
    "from the 111th Congress (Hein Bound 111). "
    "Each row is one speech with columns: `speech_id`, `text`, `speaker`, `party`, `date`, `year`."
)

@st.cache_data
def load_default():
    path = os.path.join(os.path.dirname(__file__), "..", "day1", "data", "speeches_sample.csv")
    return pd.read_csv(path)

@st.cache_data
def compute_sentiment(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(texts)
    rng = np.random.default_rng(2026)
    projection = rng.standard_normal(tfidf_matrix.shape[1])
    raw = np.asarray(tfidf_matrix @ projection).flatten()
    return (raw - raw.mean()) / raw.std()

def make_year_dummies(df):
    return pd.get_dummies(df["year"].astype(str), prefix="yr", drop_first=True).astype(float)

df_raw = load_default()
df_raw = df_raw[df_raw["party"].isin(["Republican", "Democrat"])].copy()
df_raw["D"] = (df_raw["party"] == "Republican").astype(int)

st.info(f"Using built-in dataset: {len(df_raw):,} speeches.")

with st.expander("Preview data"):
    cols = [c for c in ["speech_id", "speaker", "party", "year", "text"] if c in df_raw.columns]
    st.dataframe(df_raw[cols].head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Total speeches", f"{len(df_raw):,}")
col2.metric("Republicans", f"{df_raw['D'].sum():,}")
col3.metric("Democrats", f"{(1-df_raw['D']).sum():,}")

st.markdown("---")

# ── Section 1: LLM Sentiment Measurement ─────────────────────────────────────
st.header("1 · LLM Sentiment Measurement")

st.markdown(
    r"""
**Slide reference:** *LLMs as Measurement Operators*, *Day 2*

We construct a proxy outcome $\tilde{Y}_i = f_{\text{LLM}}(W_i)$ using TF-IDF embeddings
projected onto a fixed random direction (seed 2026), simulating an LLM-based **sentiment** score.

This is distinct from the Day 1 **stance** score (seed 42): the two measurements capture
different aspects of the text and will yield different causal estimates.
"""
)

if st.button("▶ Run Section 1 — Compute Sentiment Scores"):
    with st.spinner("Computing sentiment scores..."):
        sentiment = compute_sentiment(df_raw["text"].tolist())

    df = df_raw.copy()
    df["Y_tilde"] = sentiment

    mean_R = df[df["D"] == 1]["Y_tilde"].mean()
    mean_D = df[df["D"] == 0]["Y_tilde"].mean()
    naive  = mean_R - mean_D

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean sentiment — Republican", f"{mean_R:+.4f}")
    col2.metric("Mean sentiment — Democrat",   f"{mean_D:+.4f}")
    col3.metric("Naive Diff-in-Means",          f"{naive:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df[df["D"] == 1]["Y_tilde"], bins=40, alpha=0.6,
            color="#c0392b", label="Republican", density=True)
    ax.hist(df[df["D"] == 0]["Y_tilde"], bins=40, alpha=0.6,
            color="#2980b9", label="Democrat", density=True)
    ax.set_xlabel("Sentiment score $\\tilde{Y}$")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of LLM sentiment score by party")
    ax.legend()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Section 2: Causal Estimators ─────────────────────────────────────────────
st.header("2 · Causal Estimators")

st.markdown(
    r"""
**Slide reference:** *Estimators*, *Day 2*

We apply six estimators to the LLM sentiment outcome $\tilde{Y}$, conditioning on `year`:

1. **Difference-in-Means** — naive, unadjusted
2. **Regression Adjustment** — OLS with year dummies
3. **G-formula (Plug-in)** — predict potential outcomes, average the difference
4. **Nearest-Neighbour Matching (ATT)** — match treated to control on year
5. **IPW** — inverse probability weighting with propensity scores
6. **AIPW (Doubly Robust)** — combines outcome model and propensity model
"""
)

if st.button("▶ Run Section 2 — All Estimators"):
    with st.spinner("Computing estimators..."):
        df = df_raw.copy()
        df["Y_tilde"] = compute_sentiment(df["text"].tolist())

        Y = df["Y_tilde"]
        D = df["D"]
        X = make_year_dummies(df)

        # 1. Diff-in-Means
        n1, n0  = D.sum(), (1-D).sum()
        est_dim = Y[D==1].mean() - Y[D==0].mean()
        se_dim  = np.sqrt(Y[D==1].var(ddof=1)/n1 + Y[D==0].var(ddof=1)/n0)
        ci_dim  = (est_dim - 1.96*se_dim, est_dim + 1.96*se_dim)

        # 2. Regression
        X_reg   = sm.add_constant(pd.concat([D, X], axis=1))
        ols     = sm.OLS(Y, X_reg).fit()
        est_reg = ols.params["D"]
        se_reg  = ols.bse["D"]
        ci_reg  = tuple(ols.conf_int().loc["D"].values)

        # 3. G-formula
        df1 = X_reg.copy(); df1["D"] = 1
        df0 = X_reg.copy(); df0["D"] = 0
        m1_hat    = ols.predict(df1)
        m0_hat    = ols.predict(df0)
        est_gform = (m1_hat - m0_hat).mean()

        # 4. Matching (ATT)
        nn = NearestNeighbors(n_neighbors=1).fit(X[D==0].values)
        _, idx    = nn.kneighbors(X[D==1].values)
        est_match = (Y[D==1].values - Y[D==0].values[idx.flatten()]).mean()

        # 5. IPW
        scaler = StandardScaler()
        logit  = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        logit.fit(scaler.fit_transform(X), D)
        e_hat   = np.clip(logit.predict_proba(scaler.transform(X))[:, 1], 0.05, 0.95)
        w1, w0  = D/e_hat, (1-D)/(1-e_hat)
        est_ipw = np.sum(Y*w1)/np.sum(w1) - np.sum(Y*w0)/np.sum(w0)

        # 6. AIPW
        aipw_scores = (m1_hat - m0_hat) + (D*(Y-m1_hat))/e_hat - ((1-D)*(Y-m0_hat))/(1-e_hat)
        est_aipw    = aipw_scores.mean()
        se_aipw     = aipw_scores.std(ddof=1) / np.sqrt(len(Y))
        ci_aipw     = (est_aipw - 1.96*se_aipw, est_aipw + 1.96*se_aipw)

    results = {
        "Diff-in-Means":  (est_dim,   se_dim,  ci_dim),
        "Regression":     (est_reg,   se_reg,  ci_reg),
        "G-formula":      (est_gform, None,    None),
        "Matching (ATT)": (est_match, None,    None),
        "IPW":            (est_ipw,   None,    None),
        "AIPW (DR)":      (est_aipw,  se_aipw, ci_aipw),
    }

    rows = []
    for name, (est, se, ci) in results.items():
        rows.append({
            "Estimator": name,
            "ATE":       f"{est:+.4f}",
            "SE":        f"{se:.4f}" if se is not None else "—",
            "95% CI":    f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "—",
        })
    st.subheader("Estimator Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    names  = [r["Estimator"] for r in rows]
    ests   = [float(r["ATE"]) for r in rows]
    colors = ["#c0392b", "#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]
    ax.barh(names, ests, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Estimated ATE (sentiment score)")
    ax.set_title("Comparison of causal estimators — Day 2")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "Note: Matching and IPW are based on `year` only — a weak covariate. "
        "In practice, richer covariates (e.g., text-based features) would improve balance."
    )

    with st.expander("OLS regression summary"):
        st.text(ols.summary().as_text())

st.markdown("---")

# ── Section 3: Measurement Error ─────────────────────────────────────────────
st.header("3 · Measurement Error: MCAR vs MNAR")

st.markdown(
    r"""
**Slide reference:** *Measurement Error*, *MCAR vs MNAR*, *Day 2*

We simulate two types of measurement error on the LLM sentiment score $\tilde{Y}$:

- **MCAR (Classical):** $\epsilon \sim \mathcal{N}(0, \sigma^2)$, independent of $D$.
  The ATE estimate remains unbiased but variance increases.
- **MNAR (Non-classical):** $\epsilon \sim \mathcal{N}(0, \sigma^2) + \delta \cdot D$.
  Error is correlated with treatment — the ATE estimate is **biased**.
"""
)

noise_sd  = st.slider("MCAR noise standard deviation (σ)", 0.1, 3.0, 1.0, 0.1)
mnar_bias = st.slider("MNAR bias term (δ)", 0.0, 2.0, 0.5, 0.1)

if st.button("▶ Run Section 3 — Measurement Error Simulation"):
    with st.spinner("Running simulation..."):
        df = df_raw.copy()
        df["Y_tilde"] = compute_sentiment(df["text"].tolist())
        D = df["D"]
        X = make_year_dummies(df)

        def reg_ate(y, d, x):
            Xr = sm.add_constant(pd.concat([d, x], axis=1))
            m  = sm.OLS(y, Xr).fit()
            return m.params["D"], m.bse["D"]

        ate_true, se_true = reg_ate(df["Y_tilde"], D, X)

        rng    = np.random.default_rng(42)
        Y_mcar = df["Y_tilde"] + rng.normal(0, noise_sd, len(df))
        ate_mcar, se_mcar = reg_ate(Y_mcar, D, X)

        Y_mnar = df["Y_tilde"] + rng.normal(0, noise_sd/2, len(df)) + mnar_bias * D
        ate_mnar, se_mnar = reg_ate(Y_mnar, D, X)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline ATE (no error)", f"{ate_true:+.4f}", f"SE: {se_true:.4f}")
    col2.metric("MCAR ATE",                f"{ate_mcar:+.4f}", f"Bias: {ate_mcar-ate_true:+.4f}")
    col3.metric("MNAR ATE",                f"{ate_mnar:+.4f}", f"Bias: {ate_mnar-ate_true:+.4f}")

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Baseline\n(no error)", f"MCAR\n(σ={noise_sd})", f"MNAR\n(δ={mnar_bias})"]
    ates   = [ate_true, ate_mcar, ate_mnar]
    colors = ["#27ae60", "#2980b9", "#c0392b"]
    ax.bar(labels, ates, color=colors, alpha=0.85, width=0.5)
    ax.axhline(ate_true, color="black", linestyle="--", linewidth=0.8, label="Baseline")
    ax.set_ylabel("ATE estimate")
    ax.set_title("Effect of measurement error on ATE")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if abs(ate_mnar - ate_true) > abs(ate_mcar - ate_true):
        st.warning(
            f"MNAR introduces a bias of **{ate_mnar-ate_true:+.4f}** vs "
            f"MCAR bias of **{ate_mcar-ate_true:+.4f}**. "
            "Non-classical measurement error is substantially more harmful."
        )
    else:
        st.info("With these parameters, MCAR and MNAR produce similar bias.")

st.markdown("---")

# ── Section 4: Compare Day 1 vs Day 2 ────────────────────────────────────────
st.header("4 · Compare Day 1 (Stance) vs Day 2 (Sentiment)")

st.markdown(
    r"""
**Slide reference:** *Measurement Choices Matter*

The same causal question — does party affiliation affect speech content? — yields different
ATE estimates depending on whether we use the Day 1 **stance** score or the Day 2 **sentiment** score.
This illustrates that the choice of LLM measurement operator is a substantive research decision.
"""
)

if st.button("▶ Run Section 4 — Compare Measurements"):
    with st.spinner("Computing..."):
        df = df_raw.copy()
        X  = make_year_dummies(df)
        D  = df["D"]

        df["sentiment"] = compute_sentiment(df["text"].tolist())
        Xr = sm.add_constant(pd.concat([D, X], axis=1))
        ate_sentiment = sm.OLS(df["sentiment"], Xr).fit().params["D"]

        stance_path = os.path.join(os.path.dirname(__file__), "..", "day1", "data", "speeches_with_stance.csv")
        ate_stance = None
        if os.path.exists(stance_path):
            df_stance = pd.read_csv(stance_path)
            if "speech_id" in df.columns and "speech_id" in df_stance.columns:
                df = df.merge(df_stance[["speech_id", "Y_tilde"]], on="speech_id", how="left")
            else:
                df["Y_tilde"] = df_stance["Y_tilde"].values[:len(df)]
            df_s = df.dropna(subset=["Y_tilde"])
            Xs   = make_year_dummies(df_s)
            Xrs  = sm.add_constant(pd.concat([df_s["D"], Xs], axis=1))
            ate_stance = sm.OLS(df_s["Y_tilde"], Xrs).fit().params["D"]

    col1, col2 = st.columns(2)
    col1.metric("ATE — Sentiment (Day 2)", f"{ate_sentiment:+.4f}")
    if ate_stance is not None:
        col2.metric("ATE — Stance (Day 1)", f"{ate_stance:+.4f}")
    else:
        col2.info("Day 1 stance scores not found. Run Day 1 scripts first.")

    st.markdown(
        "The two measurements capture different latent constructs. "
        "The choice of measurement operator changes the substantive conclusion — "
        "a key lesson of the workshop."
    )

st.markdown("---")
st.caption(
    "Moses Boudourides · Northwestern University · "
    "Causal Inference with LLMs Workshop · Day 2"
)
