#!/usr/bin/env Rscript
# Day 3 — DAG Visualizations (R version)
# Generates three directed acyclic graphs for Day 3 using system graphviz (dot).
# Fully self-contained: no Python or reticulate dependencies.
#
# Run from seminar_computations/ root:
#   Rscript day3/r_app/dag_day3.R

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
dir.create("day3/data/dags", recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Helper: write DOT string to temp file and render with system dot
# ---------------------------------------------------------------------------
render_dot <- function(dot_string, out_path) {
  tmp <- tempfile(fileext = ".dot")
  writeLines(dot_string, tmp)
  ret <- system(paste0("dot -Tpng \"", tmp, "\" -o \"", out_path, "\""))
  if (ret != 0) stop("dot rendering failed for: ", out_path)
  cat("Saved:", out_path, "\n")
}

# ---------------------------------------------------------------------------
# Colour palette (identical to Days 1 & 2)
# ---------------------------------------------------------------------------
COL_TREATMENT  <- "#1565C0"
COL_OUTCOME    <- "#1B5E20"
COL_CONFOUNDER <- "#FFA500"
COL_UNOBS      <- "#6A1B9A"
COL_PROXY      <- "#F785B1"
COL_ERROR      <- "#B71C1C"
COL_HIGHLIGHT  <- "#FF8F00"
COL_ML         <- "#00695C"

# ---------------------------------------------------------------------------
# DAG 1: Double Machine Learning — High-Dimensional Confounding
# ---------------------------------------------------------------------------
dag1 <- paste0('
digraph dag1_dml {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 1 \u2014 Double Machine Learning: High-Dimensional Confounding (Day 3)",
         labelloc=t, fontsize=15]
  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]
  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  X   [label="X\n(High-dim\ncovariates)", fillcolor="', COL_CONFOUNDER, '"]
  D   [label="D\n(Party)",                fillcolor="', COL_TREATMENT,  '"]
  Y   [label="Y~\n(LLM proxy)",           fillcolor="', COL_PROXY,      '"]
  gX  [label="g-hat(X)\nE[Y~|X]",         fillcolor="', COL_ML, '",
       shape=box, fontsize=11, width=1.3, fixedsize=true]
  eX  [label="e-hat(X)\nP(D=1|X)",        fillcolor="', COL_ML, '",
       shape=box, fontsize=11, width=1.3, fixedsize=true]
  Yr  [label="Y~-g-hat(X)\n(residual)",   fillcolor="', COL_OUTCOME,   '", fontsize=11]
  Dr  [label="D-e-hat(X)\n(residual)",    fillcolor="', COL_TREATMENT, '", fontsize=11]
  th  [label="theta-hat\n(DML ATE)",      fillcolor="', COL_HIGHLIGHT, '"]

  X  -> D   [color="', COL_CONFOUNDER, '", xlabel="confounds"]
  X  -> Y   [color="', COL_CONFOUNDER, '"]
  D  -> Y   [color="', COL_TREATMENT,  '", penwidth=2.5, xlabel="causal effect"]
  X  -> gX  [color="', COL_ML, '", style=dashed, xlabel="ML fit"]
  X  -> eX  [color="', COL_ML, '", style=dashed]
  Y  -> Yr  [color="', COL_OUTCOME,   '", xlabel="partial out"]
  gX -> Yr  [color="', COL_ML, '", style=dashed]
  D  -> Dr  [color="', COL_TREATMENT, '", xlabel="partial out"]
  eX -> Dr  [color="', COL_ML, '", style=dashed]
  Yr -> th  [color="', COL_HIGHLIGHT, '", penwidth=2.5, xlabel="regress"]
  Dr -> th  [color="', COL_HIGHLIGHT, '", penwidth=2.5]
}')
render_dot(dag1, "day3/data/dags/dag1_dml.png")

# ---------------------------------------------------------------------------
# DAG 2: Design-Based Supervised Learning — Measurement Bridge
# ---------------------------------------------------------------------------
dag2 <- paste0('
digraph dag2_dsl {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 2 \u2014 DSL: Measurement Bridge (Labeled to Unlabeled) (Day 3)",
         labelloc=t, fontsize=15]
  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]
  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  subgraph cluster_labeled {
    label="Labeled sample L"
    style=dashed
    color="#888888"
    fontsize=12
    fontcolor="#555555"
    Y_L  [label="Y\n(True,\nlabeled)",        fillcolor="', COL_OUTCOME,    '", fontsize=11]
    Yt_L [label="Y~\n(LLM proxy,\nlabeled)",  fillcolor="', COL_PROXY,      '", fontsize=11]
    D_L  [label="D\n(labeled)",               fillcolor="', COL_TREATMENT,  '", fontsize=11]
    X_L  [label="X\n(labeled)",               fillcolor="', COL_CONFOUNDER, '", fontsize=11]
    Y_L  -> Yt_L [color="', COL_PROXY, '"]
    D_L  -> Yt_L [style=dashed, color="', COL_ERROR, '", xlabel="MNAR"]
    X_L  -> D_L
    X_L  -> Y_L
  }

  m [label="m(X,D,Y~)\ncorrection\nmodel", fillcolor="', COL_ML, '",
     shape=box, fontsize=11, width=1.4, fixedsize=true]

  subgraph cluster_unlabeled {
    label="Unlabeled sample U"
    style=dashed
    color="#888888"
    fontsize=12
    fontcolor="#555555"
    Yt_U [label="Y~\n(LLM proxy,\nunlabeled)", fillcolor="', COL_PROXY,      '", fontsize=11]
    D_U  [label="D\n(unlabeled)",              fillcolor="', COL_TREATMENT,  '", fontsize=11]
    X_U  [label="X\n(unlabeled)",              fillcolor="', COL_CONFOUNDER, '", fontsize=11]
    Yc   [label="Y-hat\n(corrected)",          fillcolor="', COL_OUTCOME,    '"]
  }

  Y_L  -> m   [color="', COL_ML, '", style=dashed, xlabel="train"]
  Yt_L -> m   [color="', COL_ML, '", style=dashed]
  D_L  -> m   [color="', COL_ML, '", style=dashed]
  X_L  -> m   [color="', COL_ML, '", style=dashed]
  m    -> Yc  [color="', COL_OUTCOME,   '", penwidth=2.5, xlabel="apply"]
  Yt_U -> Yc  [color="', COL_PROXY,    '"]
  D_U  -> Yc  [color="', COL_TREATMENT,'"]
  X_U  -> Yc  [color="', COL_CONFOUNDER,'"]

  ATE [label="ATE\n(DSL)", fillcolor="', COL_HIGHLIGHT, '"]
  Yc  -> ATE [color="', COL_HIGHLIGHT, '", penwidth=2.5]
  D_U -> ATE [color="', COL_TREATMENT, '"]
}')
render_dot(dag2, "day3/data/dags/dag2_dsl.png")

# ---------------------------------------------------------------------------
# DAG 3: Causal Auditing — Sensitive Attribute Path (ACB)
# ---------------------------------------------------------------------------
dag3 <- paste0('
digraph dag3_auditing {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 3 \u2014 Causal Auditing: Sensitive Attribute Path (Day 3)",
         labelloc=t, fontsize=15]
  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]
  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  S    [label="S\n(Sensitive\nattribute)", fillcolor="', COL_ERROR,      '"]
  W    [label="W\n(Text /\ninput)",        fillcolor="', COL_CONFOUNDER, '"]
  LLM  [label="f(W,S)\n(LLM\nmodel)",     fillcolor="', COL_ML, '",
        shape=box, fontsize=11, width=1.3, fixedsize=true]
  Yhat [label="Y-hat\n(LLM\noutput)",     fillcolor="', COL_PROXY,      '"]
  Ys   [label="Y-hat-prime\n(counterfact.\noutput)", fillcolor="', COL_PROXY, '",
        style="filled,dashed", fontsize=11]
  ACB  [label="ACB\n(Avg. Causal\nBias)", fillcolor="', COL_HIGHLIGHT,  '"]
  U    [label="U\n(Unobs.)",              fillcolor="', COL_UNOBS,       '",
        style="filled,dashed", peripheries=2]

  W    -> LLM  [color="', COL_CONFOUNDER, '", xlabel="input"]
  S    -> LLM  [color="', COL_ERROR,      '", penwidth=2.5, xlabel="sensitive\npath"]
  LLM  -> Yhat [color="', COL_PROXY,      '", penwidth=2.5]
  U    -> W    [style=dashed, color="', COL_UNOBS, '"]
  U    -> S    [style=dashed, color="', COL_UNOBS, '"]
  S    -> Ys   [style=dashed, color="', COL_ERROR, '",
                xlabel="do(S-prime)", fontcolor="', COL_ERROR, '", fontsize=11]
  W    -> Ys   [style=dashed, color="', COL_CONFOUNDER, '"]
  Yhat -> ACB  [color="', COL_HIGHLIGHT, '", penwidth=2.5, xlabel="E[Delta_i]"]
  Ys   -> ACB  [color="', COL_HIGHLIGHT, '", penwidth=2.5]
}')
render_dot(dag3, "day3/data/dags/dag3_auditing.png")

cat("\nAll Day 3 DAGs saved to day3/data/dags/\n")
