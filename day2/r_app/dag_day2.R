# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

# Day 2 - DAG Visualizations (R)
# Produces three DAGs illustrating the causal structure of Day 2.
#
#   DAG 1: Classical measurement error (MCAR) — error independent of D and X
#   DAG 2: Non-classical measurement error (MNAR) — error depends on treatment D
#   DAG 3: Full causal DAG with LLM measurement, confounders, and MNAR path
#
# Output: day2/data/dags/dag1_mcar.png
#         day2/data/dags/dag2_mnar.png
#         day2/data/dags/dag3_full.png
#
# Requirements: graphviz (dot command must be on PATH)
#   Install on Mac: brew install graphviz
#   Install on Linux: sudo apt-get install graphviz

dir.create("day2/data/dags", showWarnings = FALSE, recursive = TRUE)

# Check that dot is available
if (Sys.which("dot") == "") {
  stop("graphviz 'dot' command not found. Install graphviz and ensure it is on your PATH.")
}

# Helper: write DOT string to temp file and render to PNG
render_dot <- function(dot_string, out_path) {
  tmp <- tempfile(fileext = ".dot")
  writeLines(dot_string, tmp)
  ret <- system(paste0("dot -Tpng \"", tmp, "\" -o \"", out_path, "\""))
  if (ret != 0) stop("dot rendering failed for: ", out_path)
  cat("Saved:", out_path, "\n")
}

# ---------------------------------------------------------------------------
# Colour palette (identical to Day 1)
# ---------------------------------------------------------------------------
COL_TREATMENT  <- "#1565C0"
COL_OUTCOME    <- "#1B5E20"
COL_CONFOUNDER <- "#FFA500"
COL_UNOBS      <- "#6A1B9A"
COL_PROXY      <- "#F785B1"
COL_ERROR      <- "#B71C1C"
COL_HIGHLIGHT  <- "#FF8F00"

# ---------------------------------------------------------------------------
# DAG 1: Classical measurement error (MCAR)
# ---------------------------------------------------------------------------
dag1 <- paste0('
digraph dag1_mcar {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 1 \u2014 Classical Measurement Error: MCAR (Day 2)",
         labelloc=t, fontsize=15]

  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]

  D      [label="D\n(Party)",         fillcolor="', COL_TREATMENT, '"]
  Y      [label="Y\n(True)",           fillcolor="', COL_OUTCOME,   '"]
  Ytilde [label="Y~\n(LLM proxy)",     fillcolor="', COL_PROXY,     '"]
  eps    [label="e\n(Error)\n\u22a5 D,X", fillcolor="', COL_ERROR, '", fontsize=11]
  X      [label="X\n(Year)",           fillcolor="', COL_CONFOUNDER,'"]

  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  D      -> Y      [color="', COL_TREATMENT, '", penwidth=2.5, xlabel="causal effect"]
  Y      -> Ytilde [color="', COL_PROXY, '"]
  eps    -> Ytilde [color="', COL_ERROR, '"]
  X      -> D
  X      -> Y
}')

render_dot(dag1, "day2/data/dags/dag1_mcar.png")

# ---------------------------------------------------------------------------
# DAG 2: Non-classical measurement error (MNAR)
# ---------------------------------------------------------------------------
dag2 <- paste0('
digraph dag2_mnar {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 2 \u2014 Non-Classical Measurement Error: MNAR (Day 2)",
         labelloc=t, fontsize=15]

  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]

  D      [label="D\n(Party)",      fillcolor="', COL_TREATMENT, '"]
  Y      [label="Y\n(True)",       fillcolor="', COL_OUTCOME,   '"]
  Ytilde [label="Y~\n(LLM proxy)", fillcolor="', COL_PROXY,     '"]
  eps    [label="e\n(Error)",      fillcolor="', COL_ERROR,     '"]
  X      [label="X\n(Year)",       fillcolor="', COL_CONFOUNDER,'"]

  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  D      -> Y      [color="', COL_TREATMENT, '", penwidth=2.5, xlabel="causal effect"]
  Y      -> Ytilde [color="', COL_PROXY, '"]
  eps    -> Ytilde [color="', COL_ERROR, '"]
  D      -> eps    [style=dashed, color="', COL_ERROR, '",
                    xlabel="MNAR", fontcolor="', COL_ERROR, '", fontsize=11]
  X      -> D
  X      -> Y
}')

render_dot(dag2, "day2/data/dags/dag2_mnar.png")

# ---------------------------------------------------------------------------
# DAG 3: Full causal DAG with LLM measurement
# ---------------------------------------------------------------------------
dag3 <- paste0('
digraph dag3_full {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 3 \u2014 Full Causal DAG with LLM Measurement (Day 2)",
         labelloc=t, fontsize=15]

  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]

  D      [label="D\n(Party)",      fillcolor="', COL_TREATMENT, '"]
  Y      [label="Y\n(True)",       fillcolor="', COL_OUTCOME,   '"]
  Ytilde [label="Y~\n(LLM proxy)", fillcolor="', COL_PROXY,     '"]
  eps    [label="e\n(Error)",      fillcolor="', COL_ERROR,     '"]
  X      [label="X\n(Year)",       fillcolor="', COL_CONFOUNDER,'"]
  U      [label="U\n(Unobs.)",     fillcolor="', COL_UNOBS,     '",
          style="filled,dashed", peripheries=2]

  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  D      -> Y      [color="', COL_TREATMENT, '", penwidth=2.5, xlabel="causal effect"]
  Y      -> Ytilde [color="', COL_PROXY, '"]
  eps    -> Ytilde [color="', COL_ERROR, '"]
  D      -> eps    [style=dashed, color="', COL_ERROR, '",
                    xlabel="MNAR", fontcolor="', COL_ERROR, '", fontsize=11]
  X      -> D
  X      -> Y
  U      -> D      [style=dashed, color="', COL_UNOBS, '"]
  U      -> Y      [style=dashed, color="', COL_UNOBS, '"]
}')

render_dot(dag3, "day2/data/dags/dag3_full.png")

cat("\nAll Day 2 DAGs saved to day2/data/dags/\n")
