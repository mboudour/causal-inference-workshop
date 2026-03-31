# Day 2 - Part C: Measurement Error and Bias
# Demonstrates MCAR vs MNAR measurement error

library(readr)
library(dplyr)

cat("--- Day 2 Part C: Measurement Error Diagnostics ---\n")

data_path <- "day2/data/speeches_with_sentiment_R.csv"
if (!file.exists(data_path)) {
  stop("Error: data not found. Run partA_sentiment.R first.")
}

df             <- read_csv(data_path, show_col_types = FALSE)
df$year_factor <- as.factor(df$year)

estimate_ate <- function(data, y_col) {
  formula_str <- paste(y_col, "~ D + year_factor")
  model <- lm(as.formula(formula_str), data = data)
  est   <- coef(model)["D"]
  se    <- summary(model)$coefficients["D", "Std. Error"]
  return(c(est, se))
}

# ── 1. Baseline ───────────────────────────────────────────────────────────────
res_true <- estimate_ate(df, "Y_tilde")
cat("\n1. Baseline (No Measurement Error)\n")
cat(sprintf("ATE using true Y: %.4f (SE: %.4f)\n", res_true[1], res_true[2]))

# ── 2. MCAR ───────────────────────────────────────────────────────────────────
set.seed(42)
epsilon_mcar <- rnorm(nrow(df), mean = 0, sd = 1.0)
df$Y_mcar    <- df$Y_tilde + epsilon_mcar

res_mcar <- estimate_ate(df, "Y_mcar")
cat("\n2. Classical Measurement Error (MCAR)\n")
cat("Error epsilon ~ N(0, 1) independent of D\n")
cat(sprintf("ATE using Y_MCAR: %.4f (SE: %.4f)\n", res_mcar[1], res_mcar[2]))
cat(sprintf("Bias: %.4f\n", res_mcar[1] - res_true[1]))
cat("Note: Estimate is unbiased, but standard error increases.\n")

# ── 3. MNAR ───────────────────────────────────────────────────────────────────
bias_term    <- 0.5
epsilon_mnar <- rnorm(nrow(df), mean = 0, sd = 0.5) + (bias_term * df$D)
df$Y_mnar    <- df$Y_tilde + epsilon_mnar

res_mnar <- estimate_ate(df, "Y_mnar")
cat("\n3. Non-classical Measurement Error (MNAR)\n")
cat(sprintf("Error epsilon ~ N(0, 0.5) + %.1f*D\n", bias_term))
cat(sprintf("ATE using Y_MNAR: %.4f (SE: %.4f)\n", res_mnar[1], res_mnar[2]))
cat(sprintf("Bias: %.4f\n", res_mnar[1] - res_true[1]))
cat("Note: Estimate is heavily biased because E[epsilon | D=1] != E[epsilon | D=0].\n")

cat("\nConclusion:\n")
cat("If LLM measurement error is correlated with treatment (MNAR),\n")
cat("standard causal estimators will yield biased results.\n")
