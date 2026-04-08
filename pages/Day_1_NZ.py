# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Causal Inference with LLMs — Day 1 NZ Interactive App
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
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Causal Inference with LLMs — Day 1 NZ",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Causal Inference with LLMs — Day 1 NZ")
st.markdown(
    "**Workshop companion app.** This page mirrors Day 1 using a New Zealand parliamentary corpus. "
    "The built-in dataset comes from ParlSpeech V2 and has been adapted for comparability with the Day 1 workflow. "
    "Party labels are mapped as **National → Republican** and **Labour → Democrat**."
)
st.markdown("---")

st.header("0 · Data")

st.markdown(
    "The built-in dataset is a sample of **New Zealand House of Representatives speeches** "
    "from the **ParlSpeech V2** collection. "
    "Each row is one speech with columns such as `speech_id`, `text`, `speaker`, `party`, `date`, and `year`."
)

uploaded = st.file_uploader(
    "Upload your own CSV (optional — must have columns: text, party, year)",
    type="csv",
)

@st.cache_data
def load_default():
    path = os.path.join(
        os.path.dirname(__file__), "..", "day1_nz", "data", "llm", "speeches_sample.csv"
    )
    return pd.read_csv(path)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.success(f"Uploaded dataset loaded: {len(df_raw):,} rows.")
else:
    df_raw = load_default()
    st.info(f"Using built-in NZ dataset: {len(df_raw):,} speeches.")

required_cols = {"text", "party", "year"}
if not required_cols.issubset(df_raw.columns):
    st.error(f"Dataset must contain columns: {required_cols}. Found: {set(df_raw.columns)}")
    st.stop()

party_map = {
    "National": "Republican",
    "Labour": "Democrat",
}
if set(df_raw["party"].dropna().unique()) & set(party_map.keys()):
    df_raw["party"] = df_raw["party"].replace(party_map)

df_raw = df_raw[df_raw["party"].isin(["Republican", "Democrat"])].copy()
df_raw["D"] = (df_raw["party"] == "Republican").astype(int)

with st.expander("Preview data"):
    cols = [
        "speech_id" if "speech_id" in df_raw.columns else "text",
        "speaker" if "speaker" in df_raw.columns else "party",
        "party",
        "year",
        "text",
    ]
    st.dataframe(df_raw[cols].head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Total speeches", f"{len(df_raw):,}")
col2.metric("Republican / National", f"{df_raw['D'].sum():,}")
col3.metric("Democrat / Labour", f"{(1 - df_raw['D']).sum():,}")

st.markdown("---")

def make_year_dummies(df):
    return pd.get_dummies(df["year"].astype(str), prefix="yr", drop_first=True).astype(float)

st.header("DAGs · Causal Structure Diagrams")

st.markdown(
    """
These directed acyclic graphs (DAGs) illustrate the Day 1 NZ causal structure.

- **DAG 1 — Baseline:** treatment D (party) affects outcome Y, with observed confounder X (year)
  and unobserved confounder U creating backdoor paths.
- **DAG 2 — Adjustment:** conditioning on X blocks the backdoor path through X.
- **DAG 3 — Measurement Error:** the LLM proxy Ỹ is a noisy measurement of Y, with error that may depend on D.
"""
)

_dag_dir = os.path.join(os.path.dirname(__file__), "..", "day1_nz", "data", "dags")
_dag_files = {
    "DAG 1 — Baseline Causal Structure": os.path.join(_dag_dir, "dag1_baseline.png"),
    "DAG 2 — Backdoor Adjustment": os.path.join(_dag_dir, "dag2_adjustment.png"),
    "DAG 3 — LLM Measurement Error": os.path.join(_dag_dir, "dag3_measurement.png"),
}

for _title, _path in _dag_files.items():
    st.markdown(f"**{_title}**")
    if os.path.exists(_path):
        st.image(_path, use_container_width=True)
    else:
        st.info(f"DAG image not found at {_path}.")

st.markdown("---")
st.header("1 · Naive Estimator and Bias Decomposition")

st.markdown(
    r"""
We use **text length** as a simple proxy outcome $Y_i = |W_i|$ and party as treatment $D_i$.

The naive estimator is:
$$\mathbb{E}[Y \mid D=1] - \mathbb{E}[Y \mid D=0]$$

This equals the **ATT** plus a **selection bias** term:
$$= \underbrace{\mathbb{E}[Y(1)-Y(0)\mid D=1]}_{\tau_{\text{ATT}}} + \underbrace{\mathbb{E}[Y(0)\mid D=1] - \mathbb{E}[Y(0)\mid D=0]}_{\text{selection bias}}$$
"""
)

if st.button("▶ Run Section 1 — Naive Estimator"):
    df = df_raw.copy()
    df["Y"] = df["text"].astype(str).str.len()

    mu1 = df[df["D"] == 1]["Y"].mean()
    mu0 = df[df["D"] == 0]["Y"].mean()
    naive = mu1 - mu0

    year_dummies = make_year_dummies(df)
    X_reg = sm.add_constant(pd.concat([df[["D"]], year_dummies], axis=1))
    ols = sm.OLS(df["Y"], X_reg).fit()
    att_reg = ols.params["D"]
    sel_bias = naive - att_reg

    st.subheader("Results")
    a, b, c = st.columns(3)
    a.metric("Naive estimator", f"{naive:+.1f} chars")
    b.metric("ATT (regression-adjusted)", f"{att_reg:+.1f} chars")
    c.metric("Selection bias (approx.)", f"{sel_bias:+.1f} chars")

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Naive estimator", "ATT (adj.)", "Selection bias"]
    values = [naive, att_reg, sel_bias]
    colors = ["#c0392b", "#2980b9", "#7f8c8d"]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Characters")
    ax.set_title("Decomposition of the naive estimator")
    st.pyplot(fig)
    plt.close()

    with st.expander("Full OLS regression summary"):
        st.text(ols.summary().as_text())

st.markdown("---")
st.header("2 · Overlap (Positivity) Check")

st.markdown(
    r"""
$$0 < P(D=1 \mid X=x) < 1 \quad \forall\, x$$

We estimate the **propensity score** $\hat{e}(X) = P(D=1 \mid X)$ using logistic regression on `year`.
Overlap requires that both parties appear at every covariate value.
"""
)

if st.button("▶ Run Section 2 — Overlap Check"):
    df = df_raw.copy()
    df["Y"] = df["text"].astype(str).str.len()

    year_dummies = make_year_dummies(df)
    X_ps = year_dummies.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ps)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_scaled, df["D"])
    ps = lr.predict_proba(X_scaled)[:, 1]
    df["propensity_score"] = ps

    st.subheader("Propensity Score Distribution by Party")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(ps[df["D"] == 1], bins=30, alpha=0.6, label="Republican / National", color="#c0392b", density=True)
    ax.hist(ps[df["D"] == 0], bins=30, alpha=0.6, label="Democrat / Labour", color="#2980b9", density=True)
    ax.set_xlabel("Propensity score $\\hat{e}(X)$")
    ax.set_ylabel("Density")
    ax.set_title("Overlap check: propensity score distributions")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    col1.metric("Min propensity score", f"{ps.min():.3f}")
    col2.metric("Max propensity score", f"{ps.max():.3f}")

    if ps.min() > 0.05 and ps.max() < 0.95:
        st.success("Overlap appears satisfied: propensity scores are bounded away from 0 and 1.")
    else:
        st.warning("Some propensity scores are near 0 or 1, so overlap may be violated for some units.")

    year_tab = df.groupby("year")["D"].agg(["mean", "count"]).rename(
        columns={"mean": "Prop. Republican / National", "count": "N speeches"}
    )
    year_tab["Prop. Republican / National"] = year_tab["Prop. Republican / National"].round(3)
    st.subheader("Treatment share by year")
    st.dataframe(year_tab)

st.markdown("---")
st.header("3 · Identification: Adjustment Formula and IPW")

st.markdown(
    r"""
Under SUTVA, ignorability, and overlap, the ATE is identified by the **adjustment formula**:
$$\tau = \mathbb{E}_X\!\left[\mathbb{E}[Y \mid D=1, X] - \mathbb{E}[Y \mid D=0, X]\right]$$

We compute two estimators:

1. **G-formula (regression adjustment)**.
2. **Inverse Probability Weighting (IPW)**.
"""
)

if st.button("▶ Run Section 3 — Adjustment Formula & IPW"):
    df = df_raw.copy()
    df["Y"] = df["text"].astype(str).str.len()

    year_dummies = make_year_dummies(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(year_dummies)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, df["D"])
    df["ps"] = lr.predict_proba(X_scaled)[:, 1]

    X_reg = sm.add_constant(pd.concat([df[["D"]], year_dummies], axis=1))
    ols = sm.OLS(df["Y"], X_reg).fit()

    X1 = pd.concat([df[["D"]].assign(D=1), year_dummies], axis=1)
    X0 = pd.concat([df[["D"]].assign(D=0), year_dummies], axis=1)
    X1 = sm.add_constant(X1, has_constant="add").reindex(columns=X_reg.columns, fill_value=0)
    X0 = sm.add_constant(X0, has_constant="add").reindex(columns=X_reg.columns, fill_value=0)
    mu1_hat = ols.predict(X1)
    mu0_hat = ols.predict(X0)
    ate_gformula = (mu1_hat - mu0_hat).mean()

    eps = 1e-6
    ps = df["ps"].clip(eps, 1 - eps)
    ipw_treated = (df["D"] * df["Y"] / ps).mean()
    ipw_control = ((1 - df["D"]) * df["Y"] / (1 - ps)).mean()
    ate_ipw = ipw_treated - ipw_control

    naive = df[df["D"] == 1]["Y"].mean() - df[df["D"] == 0]["Y"].mean()

    st.subheader("ATE Estimates")
    a, b, c = st.columns(3)
    a.metric("Naive (unadjusted)", f"{naive:+.1f} chars")
    b.metric("G-formula (adjustment)", f"{ate_gformula:+.1f} chars")
    c.metric("IPW", f"{ate_ipw:+.1f} chars")

    fig, ax = plt.subplots(figsize=(6, 3))
    estimators = ["Naive", "G-formula", "IPW"]
    ates = [naive, ate_gformula, ate_ipw]
    colors = ["#7f8c8d", "#27ae60", "#8e44ad"]
    ax.bar(estimators, ates, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Estimated ATE (characters)")
    ax.set_title("Comparison of ATE estimators")
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.header("4 · LLM as Measurement Operator")

st.markdown(
    r"""
We construct a proxy outcome $\tilde{Y}_i = f_{\text{LLM}}(W_i)$ using sentence embeddings
projected onto a direction vector, simulating an LLM-based stance measurement.

The key question is whether the measurement error is **classical** or **non-classical**.
"""
)

use_precomputed = st.checkbox(
    "Use pre-computed stance scores (faster — skips embedding step)",
    value=True,
)

if st.button("▶ Run Section 4 — LLM Measurement"):
    df = df_raw.copy()
    df["Y"] = df["text"].astype(str).str.len()

    if use_precomputed:
        stance_path = os.path.join(
            os.path.dirname(__file__), "..", "day1_nz", "data", "llm", "speeches_with_stance.csv"
        )
        if os.path.exists(stance_path):
            df_stance = pd.read_csv(stance_path)
            if "speech_id" in df.columns and "speech_id" in df_stance.columns:
                df = df.merge(df_stance[["speech_id", "Y_tilde"]], on="speech_id", how="left")
            else:
                df = df.assign(Y_tilde=df_stance["Y_tilde"].values[:len(df)])
            st.info("Pre-computed stance scores loaded.")
        else:
            st.warning("Pre-computed file not found. Computing embeddings now.")
            use_precomputed = False

    if not use_precomputed:
        with st.spinner("Computing sentence embeddings..."):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            embeddings = model.encode(df["text"].tolist(), show_progress_bar=False, batch_size=8)
            np.random.seed(42)
            direction = np.random.normal(size=embeddings.shape[1])
            stance = embeddings @ direction
            stance = (stance - stance.mean()) / stance.std()
            df["Y_tilde"] = stance

    df = df.dropna(subset=["Y_tilde"])
    year_dummies = make_year_dummies(df)

    ate_llm_naive = df[df["D"] == 1]["Y_tilde"].mean() - df[df["D"] == 0]["Y_tilde"].mean()

    X_reg = sm.add_constant(pd.concat([df[["D"]], year_dummies], axis=1))
    ols_llm = sm.OLS(df["Y_tilde"], X_reg).fit()
    ate_llm_adj = ols_llm.params["D"]

    corr_error_D = np.corrcoef(df["Y_tilde"], df["D"])[0, 1]

    st.subheader("LLM-Based ATE Estimates")
    a, b, c = st.columns(3)
    a.metric("Naive LLM ATE", f"{ate_llm_naive:+.4f}")
    b.metric("Adjusted LLM ATE", f"{ate_llm_adj:+.4f}")
    c.metric("Corr(Ỹ, D)", f"{corr_error_D:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(df[df["D"] == 1]["Y_tilde"], bins=40, alpha=0.6, color="#c0392b", label="Republican / National", density=True)
    axes[0].hist(df[df["D"] == 0]["Y_tilde"], bins=40, alpha=0.6, color="#2980b9", label="Democrat / Labour", density=True)
    axes[0].set_xlabel("Stance score $\\tilde{Y}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of LLM stance by party")
    axes[0].legend()

    sample = df.sample(min(500, len(df)), random_state=1)
    axes[1].scatter(
        sample["Y"],
        sample["Y_tilde"],
        c=sample["D"].map({1: "#c0392b", 0: "#2980b9"}),
        alpha=0.3,
        s=10,
    )
    axes[1].set_xlabel("Text length $Y$ (chars)")
    axes[1].set_ylabel("Stance $\\tilde{Y}$")
    axes[1].set_title("Text length vs LLM stance")
    st.pyplot(fig)
    plt.close()

    with st.expander("OLS regression summary (LLM outcome)"):
        st.text(ols_llm.summary().as_text())

st.markdown("---")
st.header("5 · Summary: All Estimators")

st.markdown(
    "This section brings together all estimators computed above so you can compare them side by side."
)

if st.button("▶ Run Section 5 — Full Summary"):
    df = df_raw.copy()
    df["Y"] = df["text"].astype(str).str.len()
    year_dummies = make_year_dummies(df)

    naive = df[df["D"] == 1]["Y"].mean() - df[df["D"] == 0]["Y"].mean()

    X_reg = sm.add_constant(pd.concat([df[["D"]], year_dummies], axis=1))
    ols = sm.OLS(df["Y"], X_reg).fit()
    ate_reg = ols.params["D"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(year_dummies)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, df["D"])
    ps = lr.predict_proba(X_scaled)[:, 1].clip(1e-6, 1 - 1e-6)
    ate_ipw = (df["D"] * df["Y"] / ps).mean() - ((1 - df["D"]) * df["Y"] / (1 - ps)).mean()

    stance_path = os.path.join(
        os.path.dirname(__file__), "..", "day1_nz", "data", "llm", "speeches_with_stance.csv"
    )
    if os.path.exists(stance_path):
        df_stance = pd.read_csv(stance_path)
        if "speech_id" in df.columns and "speech_id" in df_stance.columns:
            df = df.merge(df_stance[["speech_id", "Y_tilde"]], on="speech_id", how="left")
        else:
            df["Y_tilde"] = df_stance["Y_tilde"].values[:len(df)]
        df_llm = df.dropna(subset=["Y_tilde"])
        year_dummies_llm = make_year_dummies(df_llm)
        ate_llm_naive = df_llm[df_llm["D"] == 1]["Y_tilde"].mean() - df_llm[df_llm["D"] == 0]["Y_tilde"].mean()
        X_llm = sm.add_constant(pd.concat([df_llm[["D"]], year_dummies_llm], axis=1))
        ate_llm_adj = sm.OLS(df_llm["Y_tilde"], X_llm).fit().params["D"]
        llm_rows = [
            {"Estimator": "LLM stance — Naive", "Outcome": "Ỹ (stance)", "ATE": f"{ate_llm_naive:+.4f}", "Adjusted": "No"},
            {"Estimator": "LLM stance — Regression adj.", "Outcome": "Ỹ (stance)", "ATE": f"{ate_llm_adj:+.4f}", "Adjusted": "Yes"},
        ]
    else:
        llm_rows = []

    summary_rows = [
        {"Estimator": "Naive difference in means", "Outcome": "Y (text length)", "ATE": f"{naive:+.1f}", "Adjusted": "No"},
        {"Estimator": "Regression adjustment (OLS)", "Outcome": "Y (text length)", "ATE": f"{ate_reg:+.1f}", "Adjusted": "Yes"},
        {"Estimator": "IPW", "Outcome": "Y (text length)", "ATE": f"{ate_ipw:+.1f}", "Adjusted": "Yes"},
    ] + llm_rows

    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Estimator comparison table")
    st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Moses Boudourides · Northwestern University · Causal Inference with LLMs Workshop · Day 1 NZ"
)
