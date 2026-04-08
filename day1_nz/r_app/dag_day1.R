# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

# Day 1 - DAG Visualizations (R)
# Produces three DAGs illustrating the causal structure of Day 1.
#
#   DAG 1: Baseline causal DAG
#   DAG 2: Backdoor adjustment DAG
#   DAG 3: LLM Measurement Error DAG
#
# Output: day1_nz/data/dags/dag1_baseline.png
#         day1_nz/data/dags/dag2_adjustment.png
#         day1_nz/data/dags/dag3_measurement.png
#
# Requirements: graphviz (dot command must be on PATH)
#   Install on Mac: brew install graphviz
#   Install on Linux: sudo apt-get install graphviz

dir.create("day1_nz/data/dags", showWarnings = FALSE, recursive = TRUE)

if (Sys.which("dot") == "") {
  stop("graphviz 'dot' command not found. Install graphviz and ensure it is on your PATH.")
}

render_dot <- function(dot_string, out_path) {
  tmp <- tempfile(fileext = ".dot")
  writeLines(dot_string, tmp)
  ret <- system(paste0("dot -Tpng \"", tmp, "\" -o \"", out_path, "\""))
  if (ret != 0) stop("dot rendering failed for: ", out_path)
  cat("Saved:", out_path, "\n")
}

COL_TREATMENT  <- "#1565C0"
COL_OUTCOME    <- "#1B5E20"
COL_CONFOUNDER <- "#FFA500"
COL_UNOBS      <- "#6A1B9A"
COL_PROXY      <- "#F785B1"
COL_ERROR      <- "#B71C1C"
COL_HIGHLIGHT  <- "#FF8F00"

dag1 <- paste0('
digraph dag1_baseline {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 1 — Baseline Causal Structure (Day 1)",
         labelloc=t, fontsize=15]

  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]

  D [label="D\n(Party)",   fillcolor="', COL_TREATMENT, '"]
  Y [label="Y\n(Outcome)", fillcolor="', COL_OUTCOME,   '"]
  X [label="X\n(Year)",    fillcolor="', COL_CONFOUNDER,'"]
  U [label="U\n(Unobs.)",  fillcolor="', COL_UNOBS,     '", style="filled,dashed", peripheries=2]

  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  D -> Y [color="', COL_TREATMENT, '", penwidth=2.5, xlabel="causal effect"]
  X -> D
  X -> Y
  U -> D [style=dashed, color="', COL_UNOBS, '"]
  U -> Y [style=dashed, color="', COL_UNOBS, '"]
}')

render_dot(dag1, "day1_nz/data/dags/dag1_baseline.png")

dag2 <- paste0('
digraph dag2_adjustment {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 2 — Backdoor Adjustment: Conditioning on X (Day 1)",
         labelloc=t, fontsize=15]

  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]

  D [label="D\n(Party)",             fillcolor="', COL_TREATMENT, '"]
  Y [label="Y\n(Outcome)",           fillcolor="', COL_OUTCOME,   '"]
  X [label="X\n(Year)\n[conditioned]", fillcolor="', COL_CONFOUNDER, '",
     color="', COL_HIGHLIGHT, '", penwidth=3.5, fontsize=11]
  U [label="U\n(Unobs.)",            fillcolor="', COL_UNOBS, '",
     style="filled,dashed", peripheries=2]

  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  D -> Y [color="', COL_TREATMENT, '", penwidth=2.5, xlabel="identified effect"]
  X -> D [style=dashed, color="#AAAAAA"]
  X -> Y [style=dashed, color="#AAAAAA"]
  U -> D [style=dashed, color="', COL_UNOBS, '"]
  U -> Y [style=dashed, color="', COL_UNOBS, '"]
}')

render_dot(dag2, "day1_nz/data/dags/dag2_adjustment.png")

dag3 <- paste0('
digraph dag3_measurement {
  graph [rankdir=LR, bgcolor=white, fontname=Helvetica, splines=spline,
         pad=0.4, nodesep=0.7, ranksep=2.4,
         label="DAG 3 — LLM Measurement Error Structure (Day 1)",
         labelloc=t, fontsize=15]

  node [shape=circle, style=filled, fontname=Helvetica, fontsize=13,
        fontcolor=white, width=1.1, fixedsize=true]

  D      [label="D\n(Party)",      fillcolor="', COL_TREATMENT, '"]
  Y      [label="Y\n(True)",       fillcolor="', COL_OUTCOME,   '"]
  Ytilde [label="Y~\n(LLM proxy)", fillcolor="', COL_PROXY,     '"]
  eps    [label="e\n(Error)",      fillcolor="', COL_ERROR,     '"]
  X      [label="X\n(Year)",       fillcolor="', COL_CONFOUNDER,'"]

  edge [arrowsize=0.9, penwidth=1.8, color="#444444"]

  D      -> Y      [color="', COL_TREATMENT, '", penwidth=2.5]
  Y      -> Ytilde [color="', COL_PROXY, '"]
  eps    -> Ytilde [color="', COL_ERROR, '"]
  D      -> eps    [style=dashed, color="', COL_ERROR, '",
                    xlabel="MNAR", fontcolor="', COL_ERROR, '", fontsize=11]
  X      -> D
  X      -> Y
}')

render_dot(dag3, "day1_nz/data/dags/dag3_measurement.png")

cat("\nAll Day 1 DAGs saved to day1_nz/data/dags/\n")