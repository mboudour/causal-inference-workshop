# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Causal Inference with LLMs — Day 2 NZ Interactive App
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
    page_title="Causal Inference with LLMs — Day 2 NZ",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Causal Inference with LLMs — Day 2 NZ")
st.markdown(
    "**Workshop companion app.** This page mirrors Day 2 using a New Zealand parliamentary corpus. "
    "The intended built-in dataset comes from **ParlSpeech V2** and the app displays the original NZ "
    "party names: **National** and **Labour**. Internally, treatment is coded as `D=1` for National "
    "and `D=0` for Labour for comparability with the workshop estimators."
)
st.markdown("---")

# ── Section 0: Data ──────────────────────────────────────────────────────────
st.header("0 · Data")

st.markdown(
    "The intended built-in dataset is a sample of **New Zealand House of Representatives speeches** "
    "from **ParlSpeech V2**. Each row should contain columns such as `speech_id`, `text`, `speaker`, "
    "`party`, `date`, and `year`."
)


@st.cache_data
def load_default():
    base = os.path.dirname(__file__)
    primary = os.path.join(base, "..", "day2_nz", "data", "llm", "speeches_sample.csv")
    fallback = os.path.join(base, "..", "day1_nz", "data", "llm", "speeches_sample.csv")

    if os.path.exists(primary):
        return pd.read_csv(primary), "day2_nz/data/llm/speeches_sample.csv"
    if os.path.exists(fallback):
        return pd.read_csv(fallback), "day1_nz/data/llm/speeches_sample.csv (fallback)"

    raise FileNotFoundError(
        "Could not find NZ speeches_sample.csv in day2_nz/data/llm/ or day1_nz/data/llm/."
    )


@st.cache_data
def compute_sentiment(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(texts)
    rng = np.random.default_rng(2026)
    projection = rng.standard_normal(tfidf_matrix.shape[1])
    raw = np.asarray(tfidf_matrix @ projection).flatten()
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
    df_raw, source_path = load_default()
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
These three directed acyclic graphs (DAGs) illustrate the causal structure underlying Day 2 NZ.

- **DAG 1 — MCAR:** error ε is independent of treatment D and covariates X (classical / MCAR).
  This increases noise without introducing systematic treatment-linked bias.
- **DAG 2 — MNAR:** a D → ε path indicates that the LLM measurement error depends on treatment
  (non-classical / MNAR), biasing causal estimates even after adjustment.
- **DAG 3 — Full DAG:** the complete data-generating process with treatment D, true outcome Y,
  proxy outcome Ỹ, observed covariate X (year), and treatment-dependent measurement error.
"""
)

_dag_primary = os.path.join(os.path.dirname(__file__), "..", "day2_nz", "data", "dags")
_dag_fallback = os.path.join(os.path.dirname(__file__), "..", "day2", "data", "dags")
_dag_dir2 = _dag_primary if os.path.exists(_dag_primary) else _dag_fallback
_dag_files2 = {
    "DAG 1 — Classical Error (MCAR)": os.path.join(_dag_dir2, "dag1_mcar.png"),
    "DAG 2 — Non-Classical Error (MNAR)": os.path.join(_dag_dir2, "dag2_mnar.png"),
    "DAG 3 — Full Causal DAG": os.path.join(_dag_dir2, "dag3_full.png"),
}

for _title, _path in _dag_files2.items():
    st.markdown(f"**{_title}**")
    if os.path.exists(_path):
        st.image(_path, use_container_width=True)
    else:
        st.info(f"DAG image not found at {_path}.")

st.markdown("---")

# ── Section 1: LLM Sentiment Measurement ─────────────────────────────────────
st.header("1 · LLM Sentiment Measurement")

st.markdown(
    r"""
**Slide reference:** *LLMs as Measurement Operators*, *Day 2*

We construct a proxy outcome $\tilde{Y}_i = f_{\text{LLM}}(W_i)$ using TF-IDF embeddings
projected onto a fixed random direction (seed 2026), simulating an LLM-based **sentiment** score.

This is distinct from the Day 1 NZ **stance** score. The two measurements capture different
aspects of the speeches and can therefore yield different causal estimates.
"""
)

if st.button("▶ Run Section 1 — Compute Sentiment Scores"):
    with st.spinner("Computing sentiment scores..."):
        sentiment = compute_sentiment(df_raw["text"].tolist())

    df = df_raw.copy()
    df["Y_tilde"] = sentiment

    mean_national = df[df["D"] == 1]["Y_tilde"].mean()
    mean_labour = df[df["D"] == 0]["Y_tilde"].mean()
    naive = mean_national - mean_labour

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean sentiment — National", f"{mean_national:+.4f}")
    col2.metric("Mean sentiment — Labour", f"{mean_labour:+.4f}")
    col3.metric("Naive Diff-in-Means", f"{naive:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df[df["D"] == 1]["Y_tilde"], bins=40, alpha=0.6,
            color="#c0392b", label="National", density=True)
    ax.hist(df[df["D"] == 0]["Y_tilde"], bins=40, alpha=0.6,
            color="#2980b9", label="Labour", density=True)
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

        n1, n0 = D.sum(), (1 - D).sum()
        est_dim = Y[D == 1].mean() - Y[D == 0].mean()
        se_dim = np.sqrt(Y[D == 1].var(ddof=1) / n1 + Y[D == 0].var(ddof=1) / n0)
        ci_dim = (est_dim - 1.96 * se_dim, est_dim + 1.96 * se_dim)

        X_reg = sm.add_constant(pd.concat([D, X], axis=1))
        ols = sm.OLS(Y, X_reg).fit()
        est_reg = ols.params["D"]
        se_reg = ols.bse["D"]
        ci_reg = tuple(ols.conf_int().loc["D"].values)

        df1 = X_reg.copy()
        df1["D"] = 1
        df0 = X_reg.copy()
        df0["D"] = 0
        m1_hat = ols.predict(df1)
        m0_hat = ols.predict(df0)
        est_gform = (m1_hat - m0_hat).mean()

        nn = NearestNeighbors(n_neighbors=1).fit(X[D == 0].values)
        _, idx = nn.kneighbors(X[D == 1].values)
        est_match = (Y[D == 1].values - Y[D == 0].values[idx.flatten()]).mean()

        scaler = StandardScaler()
        logit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        logit.fit(scaler.fit_transform(X), D)
        e_hat = np.clip(logit.predict_proba(scaler.transform(X))[:, 1], 0.05, 0.95)
        w1, w0 = D / e_hat, (1 - D) / (1 - e_hat)
        est_ipw = np.sum(Y * w1) / np.sum(w1) - np.sum(Y * w0) / np.sum(w0)

        aipw_scores = (m1_hat - m0_hat) + (D * (Y - m1_hat)) / e_hat - ((1 - D) * (Y - m0_hat)) / (1 - e_hat)
        est_aipw = aipw_scores.mean()
        se_aipw = aipw_scores.std(ddof=1) / np.sqrt(len(Y))
        ci_aipw = (est_aipw - 1.96 * se_aipw, est_aipw + 1.96 * se_aipw)

    results = {
        "Diff-in-Means": (est_dim, se_dim, ci_dim),
        "Regression": (est_reg, se_reg, ci_reg),
        "G-formula": (est_gform, None, None),
        "Matching (ATT)": (est_match, None, None),
        "IPW": (est_ipw, None, None),
        "AIPW (DR)": (est_aipw, se_aipw, ci_aipw),
    }

    rows = []
    for name, (est, se, ci) in results.items():
        rows.append({
            "Estimator": name,
            "ATE": f"{est:+.4f}",
            "SE": f"{se:.4f}" if se is not None else "—",
            "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "—",
        })
    st.subheader("Estimator Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["Estimator"] for r in rows]
    ests = [float(r["ATE"]) for r in rows]
    colors = ["#c0392b", "#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]
    ax.barh(names, ests, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Estimated ATE (sentiment score)")
    ax.set_title("Comparison of causal estimators — Day 2 NZ")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "Note: Matching and IPW are based on `year` only — a weak covariate. "
        "In practice, richer covariates (for example text-based features) would improve balance."
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
  The ATE estimate remains unbiased in expectation, but variance increases.
- **MNAR (Non-classical):** $\epsilon \sim \mathcal{N}(0, \sigma^2) + \delta \cdot D$.
  Error is correlated with treatment, so the ATE estimate becomes **biased**.
"""
)

noise_sd = st.slider("MCAR noise standard deviation (σ)", 0.1, 3.0, 1.0, 0.1)
mnar_bias = st.slider("MNAR bias term (δ)", 0.0, 2.0, 0.5, 0.1)

if st.button("▶ Run Section 3 — Measurement Error Simulation"):
    with st.spinner("Running simulation..."):
        df = df_raw.copy()
        df["Y_tilde"] = compute_sentiment(df["text"].tolist())
        D = df["D"]
        X = make_year_dummies(df)

        def reg_ate(y, d, x):
            Xr = sm.add_constant(pd.concat([d, x], axis=1))
            m = sm.OLS(y, Xr).fit()
            return m.params["D"], m.bse["D"]

        ate_true, se_true = reg_ate(df["Y_tilde"], D, X)

        rng = np.random.default_rng(42)
        Y_mcar = df["Y_tilde"] + rng.normal(0, noise_sd, len(df))
        ate_mcar, se_mcar = reg_ate(Y_mcar, D, X)

        Y_mnar = df["Y_tilde"] + rng.normal(0, noise_sd / 2, len(df)) + mnar_bias * D
        ate_mnar, se_mnar = reg_ate(Y_mnar, D, X)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline ATE (no error)", f"{ate_true:+.4f}", f"SE: {se_true:.4f}")
    col2.metric("MCAR ATE", f"{ate_mcar:+.4f}", f"Bias: {ate_mcar - ate_true:+.4f}")
    col3.metric("MNAR ATE", f"{ate_mnar:+.4f}", f"Bias: {ate_mnar - ate_true:+.4f}")

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Baseline\n(no error)", f"MCAR\n(σ={noise_sd})", f"MNAR\n(δ={mnar_bias})"]
    ates = [ate_true, ate_mcar, ate_mnar]
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
            f"MNAR introduces a bias of **{ate_mnar - ate_true:+.4f}** versus "
            f"MCAR bias of **{ate_mcar - ate_true:+.4f}**. "
            "Non-classical measurement error is substantially more harmful."
        )
    else:
        st.info("With these parameters, MCAR and MNAR produce similar bias.")

st.markdown("---")

# ── Section 4: Compare Day 1 NZ vs Day 2 NZ ──────────────────────────────────
st.header("4 · Compare Day 1 NZ (Stance) vs Day 2 NZ (Sentiment)")

st.markdown(
    r"""
**Slide reference:** *Measurement Choices Matter*

The same causal question — does party affiliation affect speech content? — can yield different
ATE estimates depending on whether we use the Day 1 NZ **stance** score or the Day 2 NZ
**sentiment** score. This illustrates that the choice of measurement operator is itself a
substantive research decision.
"""
)

if st.button("▶ Run Section 4 — Compare Measurements"):
    with st.spinner("Computing..."):
        df = df_raw.copy()
        X = make_year_dummies(df)
        D = df["D"]

        df["sentiment"] = compute_sentiment(df["text"].tolist())
        Xr = sm.add_constant(pd.concat([D, X], axis=1))
        ate_sentiment = sm.OLS(df["sentiment"], Xr).fit().params["D"]

        stance_path = os.path.join(os.path.dirname(__file__), "..", "day1_nz", "data", "llm", "speeches_with_stance.csv")
        ate_stance = None
        if os.path.exists(stance_path):
            df_stance = pd.read_csv(stance_path)
            if "speech_id" in df.columns and "speech_id" in df_stance.columns:
                df = df.merge(df_stance[["speech_id", "Y_tilde"]], on="speech_id", how="left")
            else:
                df["Y_tilde"] = df_stance["Y_tilde"].values[:len(df)]
            df_s = df.dropna(subset=["Y_tilde"])
            Xs = make_year_dummies(df_s)
            Xrs = sm.add_constant(pd.concat([df_s["D"], Xs], axis=1))
            ate_stance = sm.OLS(df_s["Y_tilde"], Xrs).fit().params["D"]

    col1, col2 = st.columns(2)
    col1.metric("ATE — Sentiment (Day 2 NZ)", f"{ate_sentiment:+.4f}")
    if ate_stance is not None:
        col2.metric("ATE — Stance (Day 1 NZ)", f"{ate_stance:+.4f}")
    else:
        col2.info("Day 1 NZ stance scores not found. Add day1_nz/data/llm/speeches_with_stance.csv first.")

    st.markdown(
        "The two measurements capture different latent constructs. The choice of measurement "
        "operator changes the substantive conclusion, which is a central lesson of the workshop."
    )

st.markdown("---")
st.caption(
    "Moses Boudourides · Northwestern University · "
    "Causal Inference with LLMs Workshop · Day 2 NZ"
)
