# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Day 2 - DAG Visualizations
Produces three DAGs illustrating the causal structure of Day 2.

  DAG 1: Classical measurement error (MCAR)
         Y (True) -> Y_tilde (LLM proxy)
         epsilon (Error) -> Y_tilde
         epsilon independent of (D, X)  [MCAR / classical]
         Shows that classical error attenuates but does not systematically bias estimates.

  DAG 2: Non-classical measurement error (MNAR)
         Same structure but D -> epsilon  [MNAR / non-classical]
         Shows that LLM measurement error depends on treatment, biasing causal estimates.

  DAG 3: Full observational DAG with LLM measurement
         D (Party) -> Y (True) -> Y_tilde (LLM proxy)
         X (Year) -> D, X -> Y
         U (Unobserved) -> D, U -> Y
         D -> epsilon -> Y_tilde  [MNAR path]
         Shows the complete data-generating process when LLMs are used as measurement operators.

Output: day2/data/dags/dag1_mcar.png
        day2/data/dags/dag2_mnar.png
        day2/data/dags/dag3_full.png

Requirements: graphviz (conda: python-graphviz)
"""

import os
import graphviz

os.makedirs("day2/data/dags", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared graph attributes (identical to Day 1)
# ---------------------------------------------------------------------------
GRAPH_ATTR = {
    "rankdir": "LR",
    "bgcolor": "white",
    "fontname": "Helvetica",
    "splines": "spline",
    "pad": "0.4",
    "nodesep": "0.7",
    "ranksep": "2.4",
}
NODE_ATTR = {
    "shape": "circle",
    "style": "filled",
    "fontname": "Helvetica",
    "fontsize": "13",
    "fontcolor": "white",
    "width": "1.1",
    "fixedsize": "true",
}
EDGE_ATTR = {
    "arrowsize": "0.9",
    "penwidth": "1.8",
    "color": "#444444",
}

# Colour palette (identical to Day 1)
COL_TREATMENT  = "#1565C0"   # dark blue
COL_OUTCOME    = "#1B5E20"   # dark green
COL_CONFOUNDER = "#FFA500"   # orange
COL_UNOBS      = "#6A1B9A"   # purple
COL_PROXY      = "#F785B1"   # dark pink
COL_ERROR      = "#B71C1C"   # dark red
COL_HIGHLIGHT  = "#FF8F00"   # amber

# ---------------------------------------------------------------------------
# DAG 1: Classical measurement error (MCAR)
# ---------------------------------------------------------------------------
g1 = graphviz.Digraph("dag1_mcar",
                       graph_attr={**GRAPH_ATTR,
                                   "label": "DAG 1 — Classical Measurement Error: MCAR (Day 2)",
                                   "labelloc": "t", "fontsize": "15"},
                       node_attr=NODE_ATTR,
                       edge_attr=EDGE_ATTR)

g1.node("D",      "D\n(Party)",      fillcolor=COL_TREATMENT)
g1.node("Y",      "Y\n(True)",       fillcolor=COL_OUTCOME)
g1.node("Ytilde", "Ỹ\n(LLM proxy)", fillcolor=COL_PROXY)
g1.node("eps",    "ε\n(Error)",      fillcolor=COL_ERROR)
g1.node("X",      "X\n(Year)",       fillcolor=COL_CONFOUNDER)

g1.edge("D",   "Y",      color=COL_TREATMENT, penwidth="2.5", xlabel="causal effect")
g1.edge("Y",   "Ytilde", color=COL_PROXY)
g1.edge("eps", "Ytilde", color=COL_ERROR)
g1.edge("X",   "D")
g1.edge("X",   "Y")
# Note: no edge from D to eps — error is independent of treatment (MCAR)
# Indicate independence with a label on the error node
g1.node("eps", "ε\n(Error)\n⊥ D,X", fillcolor=COL_ERROR, fontsize="11")

g1.render("day2/data/dags/dag1_mcar", format="png", cleanup=True)
print("Saved: day2/data/dags/dag1_mcar.png")

# ---------------------------------------------------------------------------
# DAG 2: Non-classical measurement error (MNAR)
# ---------------------------------------------------------------------------
g2 = graphviz.Digraph("dag2_mnar",
                       graph_attr={**GRAPH_ATTR,
                                   "label": "DAG 2 — Non-Classical Measurement Error: MNAR (Day 2)",
                                   "labelloc": "t", "fontsize": "15"},
                       node_attr=NODE_ATTR,
                       edge_attr=EDGE_ATTR)

g2.node("D",      "D\n(Party)",      fillcolor=COL_TREATMENT)
g2.node("Y",      "Y\n(True)",       fillcolor=COL_OUTCOME)
g2.node("Ytilde", "Ỹ\n(LLM proxy)", fillcolor=COL_PROXY)
g2.node("eps",    "ε\n(Error)",      fillcolor=COL_ERROR)
g2.node("X",      "X\n(Year)",       fillcolor=COL_CONFOUNDER)

g2.edge("D",   "Y",      color=COL_TREATMENT, penwidth="2.5", xlabel="causal effect")
g2.edge("Y",   "Ytilde", color=COL_PROXY)
g2.edge("eps", "Ytilde", color=COL_ERROR)
# Non-classical: D -> epsilon (MNAR — error depends on treatment)
g2.edge("D",   "eps",    style="dashed", color=COL_ERROR,
        xlabel="MNAR", fontcolor=COL_ERROR, fontsize="11")
g2.edge("X",   "D")
g2.edge("X",   "Y")

g2.render("day2/data/dags/dag2_mnar", format="png", cleanup=True)
print("Saved: day2/data/dags/dag2_mnar.png")

# ---------------------------------------------------------------------------
# DAG 3: Full observational DAG with LLM measurement
# ---------------------------------------------------------------------------
g3 = graphviz.Digraph("dag3_full",
                       graph_attr={**GRAPH_ATTR,
                                   "label": "DAG 3 — Full Causal DAG with LLM Measurement (Day 2)",
                                   "labelloc": "t", "fontsize": "15"},
                       node_attr=NODE_ATTR,
                       edge_attr=EDGE_ATTR)

g3.node("D",      "D\n(Party)",      fillcolor=COL_TREATMENT)
g3.node("Y",      "Y\n(True)",       fillcolor=COL_OUTCOME)
g3.node("Ytilde", "Ỹ\n(LLM proxy)", fillcolor=COL_PROXY)
g3.node("eps",    "ε\n(Error)",      fillcolor=COL_ERROR)
g3.node("X",      "X\n(Year)",       fillcolor=COL_CONFOUNDER)
g3.node("U",      "U\n(Unobs.)",     fillcolor=COL_UNOBS,
        style="filled,dashed", fontcolor="white", peripheries="2")

g3.edge("D",   "Y",      color=COL_TREATMENT, penwidth="2.5", xlabel="causal effect")
g3.edge("Y",   "Ytilde", color=COL_PROXY)
g3.edge("eps", "Ytilde", color=COL_ERROR)
g3.edge("D",   "eps",    style="dashed", color=COL_ERROR,
        xlabel="MNAR", fontcolor=COL_ERROR, fontsize="11")
g3.edge("X",   "D")
g3.edge("X",   "Y")
g3.edge("U",   "D",      style="dashed", color=COL_UNOBS)
g3.edge("U",   "Y",      style="dashed", color=COL_UNOBS)

g3.render("day2/data/dags/dag3_full", format="png", cleanup=True)
print("Saved: day2/data/dags/dag3_full.png")

print("\nAll Day 2 DAGs saved to day2/data/dags/")
