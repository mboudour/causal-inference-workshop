"""
Causal Inference with LLMs — Day 1 Interactive App
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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Inference with LLMs — Day 1",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Causal Inference with LLMs — Day 1")
st.markdown(
    "**Workshop companion app.** Work through each section in order. "
    "You can use the built-in dataset (U.S. Congressional speeches, Hein Bound 111) "
    "or upload your own CSV with the same column structure."
)
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 0 — DATA
# ─────────────────────────────────────────────
st.header("0 · Data")

st.markdown(
    "The built-in dataset is a sample of **8,000 U.S. Congressional speeches** "
    "from the 111th Congress (Hein Bound 111). "
    "Each row is one speech with columns: `speech_id`, `text`, `speaker`, `party`, `date`, `year`."
)

uploaded = st.file_uploader(
    "Upload your own CSV (optional — must have columns: text, party, year)",
    type="csv",
)

@st.cache_data
def load_default():
    path = os.path.join(os.path.dirname(__file__), "data", "speeches_sample.csv")
    return pd.read_csv(path)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.success(f"Uploaded dataset loaded: {len(df_raw):,} rows.")
else:
    df_raw = load_default()
    st.info(f"Using built-in dataset: {len(df_raw):,} speeches.")

required_cols = {"text", "party", "year"}
if not required_cols.issubset(df_raw.columns):
    st.error(f"Dataset must contain columns: {required_cols}. Found: {set(df_raw.columns)}")
    st.stop()

# Standardise party labels
df_raw = df_raw[df_raw["party"].isin(["Republican", "Democrat"])].copy()
df_raw["D"] = (df_raw["party"] == "Republican").astype(int)

with st.expander("Preview data"):
    st.dataframe(df_raw[["speech_id" if "speech_id" in df_raw.columns else "text",
                          "speaker" if "speaker" in df_raw.columns else "party",
                          "party", "year", "text"]].head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Total speeches", f"{len(df_raw):,}")
col2.metric("Republicans", f"{df_raw['D'].sum():,}")
col3.metric("Democrats", f"{(1-df_raw['D']).sum():,}")

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — NAIVE ESTIMATOR & BIAS DECOMPOSITION
# ─────────────────────────────────────────────
st.header("1 · Naive Estimator and Bias Decomposition")

st.markdown(
    r"""
**Slide reference:** *Naive Estimator and Bias*

We use **text length** as a simple proxy outcome $Y_i = |W_i|$ and party as treatment $D_i$.

The naive estimator is:
$$\mathbb{E}[Y \mid D=1] - \mathbb{E}[Y \mid D=0]$$

This equals the **ATT** plus a **selection bias** term:
$$= \underbrace{\mathbb{E}[Y(1)-Y(0)\mid D=1]}_{\tau_{\text{ATT}}} + \underbrace{\mathbb{E}[Y(0)\mid D=1] - \mathbb{E}[Y(0)\mid D=0]}_{\text{selection bias}}$$
"""
)

if st.button("▶ Run Section 1 — Naive Estimator"):
    df = df_raw.copy()
    df["Y"] = df["text"].str.len()

    mu1 = df[df["D"] == 1]["Y"].mean()
    mu0 = df[df["D"] == 0]["Y"].mean()
    naive = mu1 - mu0

    # Regression-based ATT proxy (using year as covariate)
    X_reg = sm.add_constant(df[["D", "year"]])
    ols = sm.OLS(df["Y"], X_reg).fit()
    att_reg = ols.params["D"]

    # Selection bias proxy = naive - ATT_reg
    sel_bias = naive - att_reg

    st.subheader("Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Naive estimator", f"{naive:+.1f} chars")
    res_col2.metric("ATT (regression-adjusted)", f"{att_reg:+.1f} chars")
    res_col3.metric("Selection bias (approx.)", f"{sel_bias:+.1f} chars")

    st.markdown(
        f"Republicans give speeches that are on average **{naive:+.0f} characters longer** "
        "than Democrats. But this naive difference mixes the true causal effect with selection bias."
    )

    # Bar chart
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

# ─────────────────────────────────────────────
# SECTION 2 — OVERLAP / POSITIVITY CHECK
# ─────────────────────────────────────────────
st.header("2 · Overlap (Positivity) Check")

st.markdown(
    r"""
**Slide reference:** *Overlap (Positivity)*

$$0 < P(D=1 \mid X=x) < 1 \quad \forall\, x$$

We estimate the **propensity score** $\hat{e}(X) = P(D=1 \mid X)$ using logistic regression on
`year` (and chamber/state if available). Overlap requires that both parties appear at every
covariate value — no subgroup is deterministically treated or untreated.
"""
)

if st.button("▶ Run Section 2 — Overlap Check"):
    df = df_raw.copy()
    df["Y"] = df["text"].str.len()

    # Build covariate matrix
    year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
    X_ps = year_dummies.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ps)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_scaled, df["D"])
    ps = lr.predict_proba(X_scaled)[:, 1]
    df["propensity_score"] = ps

    st.subheader("Propensity Score Distribution by Party")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(ps[df["D"] == 1], bins=30, alpha=0.6, label="Republican (D=1)", color="#c0392b", density=True)
    ax.hist(ps[df["D"] == 0], bins=30, alpha=0.6, label="Democrat (D=0)", color="#2980b9", density=True)
    ax.set_xlabel("Propensity score $\\hat{e}(X)$")
    ax.set_ylabel("Density")
    ax.set_title("Overlap check: propensity score distributions")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    ps_min = ps.min()
    ps_max = ps.max()
    ps_min_R = ps[df["D"] == 1].min()
    ps_max_D = ps[df["D"] == 0].max()

    col1, col2 = st.columns(2)
    col1.metric("Min propensity score", f"{ps_min:.3f}")
    col2.metric("Max propensity score", f"{ps_max:.3f}")

    if ps_min > 0.05 and ps_max < 0.95:
        st.success("Overlap appears satisfied: propensity scores are bounded away from 0 and 1.")
    else:
        st.warning("Some propensity scores are near 0 or 1 — overlap may be violated for some units.")

    # Year-level breakdown
    year_tab = df.groupby("year")["D"].agg(["mean", "count"]).rename(
        columns={"mean": "Prop. Republican", "count": "N speeches"}
    )
    year_tab["Prop. Republican"] = year_tab["Prop. Republican"].round(3)
    st.subheader("Treatment share by year")
    st.dataframe(year_tab)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 3 — ADJUSTMENT FORMULA (ATE via IPW and g-formula)
# ─────────────────────────────────────────────
st.header("3 · Identification: Adjustment Formula and IPW")

st.markdown(
    r"""
**Slide reference:** *Identification of the ATE*, *Adjustment Formula*

Under SUTVA, ignorability, and overlap, the ATE is identified by the **adjustment formula**:
$$\tau = \mathbb{E}_X\!\left[\mathbb{E}[Y \mid D=1, X] - \mathbb{E}[Y \mid D=0, X]\right]$$

We compute two estimators:

1. **G-formula (regression adjustment):** fit $\hat{\mu}(d, x) = \mathbb{E}[Y \mid D=d, X=x]$, predict potential outcomes for all units, average the difference.
2. **Inverse Probability Weighting (IPW):** weight each unit by $1/\hat{e}(X)$ (treated) or $1/(1-\hat{e}(X))$ (control).
"""
)

if st.button("▶ Run Section 3 — Adjustment Formula & IPW"):
    df = df_raw.copy()
    df["Y"] = df["text"].str.len()

    # Propensity scores
    year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
    X_ps = year_dummies.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ps)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, df["D"])
    df["ps"] = lr.predict_proba(X_scaled)[:, 1]

    # G-formula
    X_reg = sm.add_constant(pd.concat([df[["D", "year"]], year_dummies], axis=1))
    ols = sm.OLS(df["Y"], X_reg).fit()
    df_1 = df.copy(); df_1["D"] = 1
    df_0 = df.copy(); df_0["D"] = 0
    X1 = sm.add_constant(pd.concat([df_1[["D", "year"]], year_dummies], axis=1))
    X0 = sm.add_constant(pd.concat([df_0[["D", "year"]], year_dummies], axis=1))
    mu1_hat = ols.predict(X1)
    mu0_hat = ols.predict(X0)
    ate_gformula = (mu1_hat - mu0_hat).mean()

    # IPW
    eps = 1e-6
    ps = df["ps"].clip(eps, 1 - eps)
    ipw_treated = (df["D"] * df["Y"] / ps).mean()
    ipw_control = ((1 - df["D"]) * df["Y"] / (1 - ps)).mean()
    ate_ipw = ipw_treated - ipw_control

    # Naive
    naive = df[df["D"] == 1]["Y"].mean() - df[df["D"] == 0]["Y"].mean()

    st.subheader("ATE Estimates")
    col1, col2, col3 = st.columns(3)
    col1.metric("Naive (unadjusted)", f"{naive:+.1f} chars")
    col2.metric("G-formula (adjustment)", f"{ate_gformula:+.1f} chars")
    col3.metric("IPW", f"{ate_ipw:+.1f} chars")

    # Comparison chart
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

    st.markdown(
        "The g-formula and IPW estimators condition on `year`, removing confounding "
        "due to year-level differences in speech patterns. Compare these to the naive estimate."
    )

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 4 — LLM MEASUREMENT (PART B)
# ─────────────────────────────────────────────
st.header("4 · LLM as Measurement Operator")

st.markdown(
    r"""
**Slide reference:** *LLMs as Measurement Operators*, *Non-Classical Measurement Error*

We construct a proxy outcome $\tilde{Y}_i = f_{\text{LLM}}(W_i)$ using sentence embeddings
(SentenceTransformer `all-MiniLM-L6-v2`) projected onto a direction vector — simulating
an LLM-based stance measurement.

Key question: is the measurement error **classical** (independent of $D$) or **non-classical**
(correlated with $D$)?
"""
)

use_precomputed = st.checkbox(
    "Use pre-computed stance scores (faster — skips embedding step)",
    value=True,
)

if st.button("▶ Run Section 4 — LLM Measurement"):
    df = df_raw.copy()
    df["Y"] = df["text"].str.len()

    if use_precomputed:
        stance_path = os.path.join(os.path.dirname(__file__), "data", "speeches_with_stance.csv")
        if os.path.exists(stance_path):
            df_stance = pd.read_csv(stance_path)
            df = df.merge(df_stance[["speech_id", "Y_tilde"]], on="speech_id", how="left") \
                if "speech_id" in df.columns and "speech_id" in df_stance.columns \
                else df.assign(Y_tilde=df_stance["Y_tilde"].values[:len(df)])
            st.info("Pre-computed stance scores loaded.")
        else:
            st.warning("Pre-computed file not found. Computing embeddings now (may take a minute).")
            use_precomputed = False

    if not use_precomputed:
        with st.spinner("Computing sentence embeddings (this may take 1–2 minutes)..."):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            embeddings = model.encode(df["text"].tolist(), show_progress_bar=False, batch_size=8)
            np.random.seed(42)
            direction = np.random.normal(size=embeddings.shape[1])
            stance = embeddings @ direction
            stance = (stance - stance.mean()) / stance.std()
            df["Y_tilde"] = stance

    df = df.dropna(subset=["Y_tilde"])

    # Naive LLM-based ATE
    ate_llm_naive = df[df["D"] == 1]["Y_tilde"].mean() - df[df["D"] == 0]["Y_tilde"].mean()

    # Adjusted LLM-based ATE (regression on year)
    X_reg = sm.add_constant(df[["D", "year"]])
    ols_llm = sm.OLS(df["Y_tilde"], X_reg).fit()
    ate_llm_adj = ols_llm.params["D"]

    # Measurement error correlation with D
    corr_error_D = np.corrcoef(df["Y_tilde"], df["D"])[0, 1]

    st.subheader("LLM-Based ATE Estimates")
    col1, col2, col3 = st.columns(3)
    col1.metric("Naive LLM ATE", f"{ate_llm_naive:+.4f}")
    col2.metric("Adjusted LLM ATE", f"{ate_llm_adj:+.4f}")
    col3.metric("Corr(Ỹ, D)", f"{corr_error_D:+.4f}",
                help="Non-zero correlation indicates non-classical measurement error")

    # Distribution of stance by party
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].hist(df[df["D"] == 1]["Y_tilde"], bins=40, alpha=0.6,
                 color="#c0392b", label="Republican", density=True)
    axes[0].hist(df[df["D"] == 0]["Y_tilde"], bins=40, alpha=0.6,
                 color="#2980b9", label="Democrat", density=True)
    axes[0].set_xlabel("Stance score $\\tilde{Y}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of LLM stance by party")
    axes[0].legend()

    # Scatter: stance vs text length
    sample = df.sample(min(500, len(df)), random_state=1)
    axes[1].scatter(sample["Y"], sample["Y_tilde"],
                    c=sample["D"].map({1: "#c0392b", 0: "#2980b9"}),
                    alpha=0.3, s=10)
    axes[1].set_xlabel("Text length $Y$ (chars)")
    axes[1].set_ylabel("Stance $\\tilde{Y}$")
    axes[1].set_title("Text length vs LLM stance (red=Rep, blue=Dem)")

    st.pyplot(fig)
    plt.close()

    if abs(corr_error_D) > 0.05:
        st.warning(
            f"The LLM-derived outcome is correlated with treatment (r = {corr_error_D:.3f}). "
            "This is evidence of **non-classical measurement error**: the measurement is not "
            "independent of the treatment, which can bias causal estimates."
        )
    else:
        st.success("Measurement error appears approximately uncorrelated with treatment.")

    with st.expander("OLS regression summary (LLM outcome)"):
        st.text(ols_llm.summary().as_text())

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 5 — FULL COMPARISON SUMMARY
# ─────────────────────────────────────────────
st.header("5 · Summary: All Estimators")

st.markdown(
    """
**Slide reference:** *Why Causality Matters*, *Implications and Outlook*

This section brings together all estimators computed above so you can compare them side by side.
Click the button after running all previous sections.
"""
)

if st.button("▶ Run Section 5 — Full Summary"):
    df = df_raw.copy()
    df["Y"] = df["text"].str.len()

    # Naive
    naive = df[df["D"] == 1]["Y"].mean() - df[df["D"] == 0]["Y"].mean()

    # Regression adjusted
    X_reg = sm.add_constant(df[["D", "year"]])
    ols = sm.OLS(df["Y"], X_reg).fit()
    ate_reg = ols.params["D"]

    # Propensity score / IPW
    year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
    X_ps = year_dummies.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ps)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, df["D"])
    ps = lr.predict_proba(X_scaled)[:, 1].clip(1e-6, 1 - 1e-6)
    ate_ipw = (df["D"] * df["Y"] / ps).mean() - ((1 - df["D"]) * df["Y"] / (1 - ps)).mean()

    # LLM-based
    stance_path = os.path.join(os.path.dirname(__file__), "data", "speeches_with_stance.csv")
    if os.path.exists(stance_path):
        df_stance = pd.read_csv(stance_path)
        if "speech_id" in df.columns and "speech_id" in df_stance.columns:
            df = df.merge(df_stance[["speech_id", "Y_tilde"]], on="speech_id", how="left")
        else:
            df["Y_tilde"] = df_stance["Y_tilde"].values[:len(df)]
        df_llm = df.dropna(subset=["Y_tilde"])
        ate_llm_naive = df_llm[df_llm["D"] == 1]["Y_tilde"].mean() - df_llm[df_llm["D"] == 0]["Y_tilde"].mean()
        X_llm = sm.add_constant(df_llm[["D", "year"]])
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

    st.markdown(
        """
**Key takeaways:**
- The naive estimator conflates the causal effect with selection bias.
- Regression adjustment and IPW produce similar estimates when the model is correctly specified.
- The LLM-based outcome gives a completely different scale and potentially different conclusions — illustrating that *how you measure the outcome matters as much as how you estimate the effect*.
- Non-classical measurement error from the LLM can bias even the adjusted estimates.
"""
    )

st.markdown("---")
st.caption(
    "Moses Boudourides · Northwestern University · "
    "Causal Inference with LLMs Workshop · Day 1"
)
