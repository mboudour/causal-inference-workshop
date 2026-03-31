# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Causal Inference with LLMs — Workshop App
Moses Boudourides | Northwestern University
"""

import streamlit as st

st.set_page_config(
    page_title="Causal Inference with LLMs",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Causal Inference with LLMs")
st.subheader("Moses Boudourides · Northwestern University")
st.markdown("---")

st.markdown(
    """
This is the interactive companion app for the **Causal Inference with LLMs** workshop.
Use the sidebar to navigate to each day's materials.

| Day | Topic | Status |
|-----|-------|--------|
| [Day 1](Day_1) | Standard causal inference, LLM measurement, adjustment methods | ✅ Available |
| [Day 2](Day_2) | Causal estimators (DiM, Regression, G-formula, Matching, IPW, AIPW), measurement error (MCAR vs MNAR) | ✅ Available |
| [Day 3](Day_3) | Estimation \& Auditing with LLMs: DML, DSL, Causal Auditing | 🔜 |
"""
)

st.markdown("---")
st.caption("Source code: [github.com/mboudour/causal-inference-workshop](https://github.com/mboudour/causal-inference-workshop)")
