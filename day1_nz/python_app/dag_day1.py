# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Day 1 - DAG Visualizations
Produces three DAGs illustrating the causal structure of Day 1.

  DAG 1: Baseline causal DAG
         D (Party) -> Y (Outcome)
         X (Year/Covariates) -> D, X -> Y
         U (Unobserved) -> D, U -> Y
         Shows why naive DiM is biased.

  DAG 2: Backdoor adjustment DAG
         Same graph; X node highlighted to indicate conditioning.
         Shows that conditioning on X blocks the backdoor path.

  DAG 3: Measurement error DAG
         Y (True) -> Y_tilde (LLM proxy)
         epsilon (Error) -> Y_tilde
         D -> epsilon  [non-classical / MNAR]
         Shows why LLM-measured outcomes introduce additional bias.

Output: day1/data/dags/dag1_baseline.png
        day1/data/dags/dag2_adjustment.png
        day1/data/dags/dag3_measurement.png

Requirements: graphviz (conda: python-graphviz)
"""

import os
import graphviz

# os.makedirs("day1/data/dags", exist_ok=True)
os.makedirs("day1_nz/data/dags", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared graph attributes
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

# Colour palette
COL_TREATMENT  = "#1565C0"   # dark blue
COL_OUTCOME    = "#1B5E20"   # dark green
COL_CONFOUNDER = "#FFA500"   # orange "#E65100"
COL_UNOBS      = "#6A1B9A"   # purple
COL_PROXY      = "#F785B1"   # dark pink "#AD1457"
COL_ERROR      = "#B71C1C"   # dark red
COL_HIGHLIGHT  = "#FF8F00"   # amber (adjustment set border)

# ---------------------------------------------------------------------------
# DAG 1: Baseline causal DAG
# ---------------------------------------------------------------------------
g1 = graphviz.Digraph("dag1_baseline",
                       graph_attr={**GRAPH_ATTR, "label": "DAG 1 — Baseline Causal Structure (Day 1)",
                                   "labelloc": "t", "fontsize": "15"},
                       node_attr=NODE_ATTR,
                       edge_attr=EDGE_ATTR)

g1.node("D", "D\n(Party)",       fillcolor=COL_TREATMENT)
g1.node("Y", "Y\n(Outcome)",     fillcolor=COL_OUTCOME)
g1.node("X", "X\n(Year)",        fillcolor=COL_CONFOUNDER)
g1.node("U", "U\n(Unobs.)",      fillcolor=COL_UNOBS,
        style="filled,dashed", fontcolor="white", peripheries="2")

g1.edge("D", "Y", color=COL_TREATMENT, penwidth="2.5", xlabel="causal effect")
g1.edge("X", "D")
g1.edge("X", "Y")
g1.edge("U", "D", style="dashed", color=COL_UNOBS)
g1.edge("U", "Y", style="dashed", color=COL_UNOBS)

# g1.render("day1/data/dags/dag1_baseline", format="png", cleanup=True)
# print("Saved: day1/data/dags/dag1_baseline.png")
g1.render("day1_nz/data/dags/dag1_baseline", format="png", cleanup=True)
print("Saved: day1_nz/data/dags/dag1_baseline.png")

# ---------------------------------------------------------------------------
# DAG 2: Backdoor adjustment DAG
# ---------------------------------------------------------------------------
g2 = graphviz.Digraph("dag2_adjustment",
                       graph_attr={**GRAPH_ATTR,
                                   "label": "DAG 2 — Backdoor Adjustment: Conditioning on X (Day 1)",
                                   "labelloc": "t", "fontsize": "15"},
                       node_attr=NODE_ATTR,
                       edge_attr=EDGE_ATTR)

g2.node("D", "D\n(Party)",   fillcolor=COL_TREATMENT)
g2.node("Y", "Y\n(Outcome)", fillcolor=COL_OUTCOME)
# X highlighted with amber border to indicate it is in the adjustment set
g2.node("X", "X\n(Year)\n[conditioned]",
        fillcolor=COL_CONFOUNDER,
        color=COL_HIGHLIGHT, penwidth="3.5", fontsize="11")
g2.node("U", "U\n(Unobs.)", fillcolor=COL_UNOBS,
        style="filled,dashed", fontcolor="white", peripheries="2")

g2.edge("D", "Y", color=COL_TREATMENT, penwidth="2.5", xlabel="identified effect")
# Paths through X are blocked (shown in grey/dashed to indicate blocking)
g2.edge("X", "D", style="dashed", color="#AAAAAA")
g2.edge("X", "Y", style="dashed", color="#AAAAAA")
g2.edge("U", "D", style="dashed", color=COL_UNOBS)
g2.edge("U", "Y", style="dashed", color=COL_UNOBS)

# g2.render("day1/data/dags/dag2_adjustment", format="png", cleanup=True)
# print("Saved: day1/data/dags/dag2_adjustment.png")
g2.render("day1_nz/data/dags/dag2_adjustment", format="png", cleanup=True)
print("Saved: day1_nz/data/dags/dag2_adjustment.png")

# ---------------------------------------------------------------------------
# DAG 3: Measurement error DAG
# ---------------------------------------------------------------------------
g3 = graphviz.Digraph("dag3_measurement",
                       graph_attr={**GRAPH_ATTR,
                                   "label": "DAG 3 — LLM Measurement Error Structure (Day 1)",
                                   "labelloc": "t", "fontsize": "15"},
                       node_attr=NODE_ATTR,
                       edge_attr=EDGE_ATTR)

g3.node("D",      "D\n(Party)",      fillcolor=COL_TREATMENT)
g3.node("Y",      "Y\n(True)",       fillcolor=COL_OUTCOME)
g3.node("Ytilde", "Ỹ\n(LLM proxy)", fillcolor=COL_PROXY)
g3.node("eps",    "ε\n(Error)",      fillcolor=COL_ERROR)
g3.node("X",      "X\n(Year)",       fillcolor=COL_CONFOUNDER)

g3.edge("D",   "Y",      color=COL_TREATMENT, penwidth="2.5")
g3.edge("Y",   "Ytilde", color=COL_PROXY)
g3.edge("eps", "Ytilde", color=COL_ERROR)
# Non-classical error: D -> epsilon (MNAR)
g3.edge("D",   "eps",    style="dashed", color=COL_ERROR,
        xlabel="MNAR", fontcolor=COL_ERROR, fontsize="11")
g3.edge("X",   "D")
g3.edge("X",   "Y")

# g3.render("day1/data/dags/dag3_measurement", format="png", cleanup=True)
# print("Saved: day1/data/dags/dag3_measurement.png")
g3.render("day1_nz/data/dags/dag3_measurement", format="png", cleanup=True)
print("Saved: day1_nz/data/dags/dag3_measurement.png")

# print("\nAll Day 1 DAGs saved to day1/data/dags/")
print("\nAll Day 1 DAGs saved to day1_nz/data/dags/")
