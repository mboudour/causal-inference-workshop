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
st.subheader("Moses Boudourides")
st.markdown(
    "[https://github.com/mboudour/causal-inference-workshop](https://github.com/mboudour/causal-inference-workshop )"
)
st.markdown("---")

st.markdown(
    """
This is the interactive companion app for the **Causal Inference with LLMs** workshop.
Use the sidebar to navigate to each day's materials.

| Day | Topic | Status |
|-----|-------|--------|
| [Day 1 Hein Bound 111](Day_1) | Standard causal inference, LLM measurement, adjustment methods | ✅ Available |
| [Day 1 NZ](Day_1_NZ) | New Zealand adaptation of Day 1 using ParlSpeech V2 speeches | ✅ Available |
| [Day 2 Hein Bound 111](Day_2) | Causal estimators (DiM, Regression, G-formula, Matching, IPW, AIPW), measurement error (MCAR vs MNAR) | ✅ Available |
| [Day 2 NZ](Day_2_NZ) | New Zealand adaptation of Day 2 using ParlSpeech V2 speeches | ✅ Available |
| [Day 3 Hein Bound 111](Day_3) | Estimation & Auditing with LLMs: DML, DSL, Causal Auditing | ✅ Available |
| [Day 3 NZ](Day_3_NZ) | New Zealand adaptation of Day 3 using ParlSpeech V2 speeches | ✅ Available |
"""
)

st.markdown("---")
st.caption("Source code: [github.com/mboudour/causal-inference-workshop](https://github.com/mboudour/causal-inference-workshop )")
