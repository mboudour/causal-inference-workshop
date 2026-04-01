"""
Day 3 — DAG Visualizations
Generates three directed acyclic graphs (DAGs) for Day 3:
  1. DML — High-Dimensional Confounding (cross-fitting structure)
  2. DSL — Measurement Bridge (labeled/unlabeled sample correction)
  3. Causal Auditing — Sensitive Attribute Path (ACB framework)

Run from seminar_computations/ root:
    python3 day3/python_app/dag_day3.py
"""

import os
import graphviz

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
os.makedirs("day3/data/dags", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
GRAPH_ATTR = {
    "rankdir":  "LR",
    "bgcolor":  "white",
    "splines":  "spline",
    "pad":      "0.4",
    "nodesep":  "0.7",
    "ranksep":  "2.4",
}
NODE_ATTR = {
    "shape":     "circle",
    "style":     "filled",
    "fontname":  "Helvetica",
    "fontsize":  "13",
    "fontcolor": "white",
    "width":     "1.1",
    "fixedsize": "true",
}
EDGE_ATTR = {
    "arrowsize": "0.9",
    "penwidth":  "1.8",
    "color":     "#444444",
}

# Colour palette (identical to Days 1 & 2)
COL_TREATMENT  = "#1565C0"   # dark blue
COL_OUTCOME    = "#1B5E20"   # dark green
COL_CONFOUNDER = "#FFA500"   # orange
COL_UNOBS      = "#6A1B9A"   # purple
COL_PROXY      = "#F785B1"   # dark pink
COL_ERROR      = "#B71C1C"   # dark red
COL_HIGHLIGHT  = "#FF8F00"   # amber
COL_ML         = "#00695C"   # teal (ML nuisance)

# ---------------------------------------------------------------------------
# DAG 1: Double Machine Learning — High-Dimensional Confounding
# ---------------------------------------------------------------------------
g1 = graphviz.Digraph(
    "dag1_dml",
    graph_attr={**GRAPH_ATTR,
                "label": "DAG 1 — Double Machine Learning: High-Dimensional Confounding (Day 3)",
                "labelloc": "t", "fontsize": "15"},
    node_attr=NODE_ATTR,
    edge_attr=EDGE_ATTR,
)

# Nodes
g1.node("X",   "X\n(High-dim\ncovariates)", fillcolor=COL_CONFOUNDER)
g1.node("D",   "D\n(Party)",                fillcolor=COL_TREATMENT)
g1.node("Y",   "Ỹ\n(LLM proxy)",            fillcolor=COL_PROXY)
g1.node("gX",  "ĝ(X)\nE[Ỹ|X]",             fillcolor=COL_ML, shape="box",
        fontsize="11", width="1.3", fixedsize="true")
g1.node("eX",  "ê(X)\nP(D=1|X)",            fillcolor=COL_ML, shape="box",
        fontsize="11", width="1.3", fixedsize="true")
g1.node("Yr",  "Ỹ−ĝ(X)\n(residual)",        fillcolor=COL_OUTCOME,
        fontsize="11")
g1.node("Dr",  "D−ê(X)\n(residual)",         fillcolor=COL_TREATMENT,
        fontsize="11")
g1.node("th",  "θ̂\n(DML ATE)",              fillcolor=COL_HIGHLIGHT)

# Edges
g1.edge("X",  "D",   color=COL_CONFOUNDER, xlabel="confounds")
g1.edge("X",  "Y",   color=COL_CONFOUNDER)
g1.edge("D",  "Y",   color=COL_TREATMENT,  penwidth="2.5", xlabel="causal effect")
g1.edge("X",  "gX",  color=COL_ML,         style="dashed", xlabel="ML fit")
g1.edge("X",  "eX",  color=COL_ML,         style="dashed")
g1.edge("Y",  "Yr",  color=COL_OUTCOME,    xlabel="partial out")
g1.edge("gX", "Yr",  color=COL_ML,         style="dashed")
g1.edge("D",  "Dr",  color=COL_TREATMENT,  xlabel="partial out")
g1.edge("eX", "Dr",  color=COL_ML,         style="dashed")
g1.edge("Yr", "th",  color=COL_HIGHLIGHT,  penwidth="2.5", xlabel="regress")
g1.edge("Dr", "th",  color=COL_HIGHLIGHT,  penwidth="2.5")

g1.render("day3/data/dags/dag1_dml", format="png", cleanup=True)
print("Saved: day3/data/dags/dag1_dml.png")

# ---------------------------------------------------------------------------
# DAG 2: Design-Based Supervised Learning — Measurement Bridge
# ---------------------------------------------------------------------------
g2 = graphviz.Digraph(
    "dag2_dsl",
    graph_attr={**GRAPH_ATTR,
                "label": "DAG 2 — DSL: Measurement Bridge (Labeled → Unlabeled) (Day 3)",
                "labelloc": "t", "fontsize": "15"},
    node_attr=NODE_ATTR,
    edge_attr=EDGE_ATTR,
)

# Labeled sample subgraph
with g2.subgraph(name="cluster_labeled") as c:
    c.attr(label="Labeled sample L", style="dashed", color="#888888",
           fontsize="12", fontcolor="#555555")
    c.node("Y_L",  "Y\n(True,\nlabeled)",  fillcolor=COL_OUTCOME,  fontsize="11")
    c.node("Yt_L", "Ỹ\n(LLM proxy,\nlabeled)", fillcolor=COL_PROXY, fontsize="11")
    c.node("D_L",  "D\n(labeled)",          fillcolor=COL_TREATMENT, fontsize="11")
    c.node("X_L",  "X\n(labeled)",          fillcolor=COL_CONFOUNDER, fontsize="11")
    c.edge("Y_L",  "Yt_L", color=COL_PROXY)
    c.edge("D_L",  "Yt_L", style="dashed", color=COL_ERROR, xlabel="MNAR")
    c.edge("X_L",  "D_L")
    c.edge("X_L",  "Y_L")

# Correction model
g2.node("m",   "m(X,D,Ỹ)\ncorrection\nmodel", fillcolor=COL_ML, shape="box",
        fontsize="11", width="1.4", fixedsize="true")

# Unlabeled sample subgraph
with g2.subgraph(name="cluster_unlabeled") as c:
    c.attr(label="Unlabeled sample U", style="dashed", color="#888888",
           fontsize="12", fontcolor="#555555")
    c.node("Yt_U", "Ỹ\n(LLM proxy,\nunlabeled)", fillcolor=COL_PROXY, fontsize="11")
    c.node("D_U",  "D\n(unlabeled)",              fillcolor=COL_TREATMENT, fontsize="11")
    c.node("X_U",  "X\n(unlabeled)",              fillcolor=COL_CONFOUNDER, fontsize="11")
    c.node("Yc",   "Ŷ\n(corrected)",              fillcolor=COL_OUTCOME)

# Cross-sample edges
g2.edge("Y_L",  "m",   color=COL_ML, style="dashed", xlabel="train")
g2.edge("Yt_L", "m",   color=COL_ML, style="dashed")
g2.edge("D_L",  "m",   color=COL_ML, style="dashed")
g2.edge("X_L",  "m",   color=COL_ML, style="dashed")
g2.edge("m",    "Yc",  color=COL_OUTCOME, penwidth="2.5", xlabel="apply")
g2.edge("Yt_U", "Yc",  color=COL_PROXY)
g2.edge("D_U",  "Yc",  color=COL_TREATMENT)
g2.edge("X_U",  "Yc",  color=COL_CONFOUNDER)

# ATE node
g2.node("ATE", "ATE\n(DSL)", fillcolor=COL_HIGHLIGHT)
g2.edge("Yc",  "ATE", color=COL_HIGHLIGHT, penwidth="2.5")
g2.edge("D_U", "ATE", color=COL_TREATMENT)

g2.render("day3/data/dags/dag2_dsl", format="png", cleanup=True)
print("Saved: day3/data/dags/dag2_dsl.png")

# ---------------------------------------------------------------------------
# DAG 3: Causal Auditing — Sensitive Attribute Path (ACB)
# ---------------------------------------------------------------------------
g3 = graphviz.Digraph(
    "dag3_auditing",
    graph_attr={**GRAPH_ATTR,
                "label": "DAG 3 — Causal Auditing: Sensitive Attribute Path (Day 3)",
                "labelloc": "t", "fontsize": "15"},
    node_attr=NODE_ATTR,
    edge_attr=EDGE_ATTR,
)

g3.node("S",    "S\n(Sensitive\nattribute)",   fillcolor=COL_ERROR)
g3.node("W",    "W\n(Text /\ninput)",           fillcolor=COL_CONFOUNDER)
g3.node("LLM",  "f(W,S)\n(LLM\nmodel)",        fillcolor=COL_ML, shape="box",
        fontsize="11", width="1.3", fixedsize="true")
g3.node("Yhat", "Ŷ\n(LLM\noutput)",            fillcolor=COL_PROXY)
g3.node("Ys",   "Ŷ'\n(counterfact.\noutput)",  fillcolor=COL_PROXY,
        style="filled,dashed", fontsize="11")
g3.node("ACB",  "ACB\n(Avg. Causal\nBias)",    fillcolor=COL_HIGHLIGHT)
g3.node("U",    "U\n(Unobs.)",                 fillcolor=COL_UNOBS,
        style="filled,dashed", peripheries="2")

# Edges
g3.edge("W",   "LLM",  color=COL_CONFOUNDER, xlabel="input")
g3.edge("S",   "LLM",  color=COL_ERROR,       penwidth="2.5", xlabel="sensitive\npath")
g3.edge("LLM", "Yhat", color=COL_PROXY,       penwidth="2.5")
g3.edge("U",   "W",    style="dashed",         color=COL_UNOBS)
g3.edge("U",   "S",    style="dashed",         color=COL_UNOBS)
# Counterfactual: swap S → S' → LLM → Ŷ'
g3.edge("S",   "Ys",   style="dashed",         color=COL_ERROR,
        xlabel="do(S')", fontcolor=COL_ERROR, fontsize="11")
g3.edge("W",   "Ys",   style="dashed",         color=COL_CONFOUNDER)
# ACB = E[Ŷ - Ŷ']
g3.edge("Yhat", "ACB", color=COL_HIGHLIGHT, penwidth="2.5", xlabel="E[Δᵢ]")
g3.edge("Ys",   "ACB", color=COL_HIGHLIGHT, penwidth="2.5")

g3.render("day3/data/dags/dag3_auditing", format="png", cleanup=True)
print("Saved: day3/data/dags/dag3_auditing.png")

print("\nAll Day 3 DAGs saved to day3/data/dags/")
