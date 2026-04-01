library(readr)
library(dplyr)
library(broom)

# --- LOAD DATA ---
df <- read_csv("day1/data/speeches_sample.csv", show_col_types = FALSE)

# --- OUTCOME AND TREATMENT ---
df <- df %>%
  mutate(
    Y = nchar(text),
    D = ifelse(party == "Republican", 1, 0),
    year_factor = as.factor(year)
  )

cat("--- PART C: CAUSAL ADJUSTMENT & DIAGNOSTICS ---\n\n")

# 1. NAIVE ESTIMATOR DECOMPOSITION
mu1 <- mean(df$Y[df$D == 1], na.rm = TRUE)
mu0 <- mean(df$Y[df$D == 0], na.rm = TRUE)
naive_ate <- mu1 - mu0
cat(sprintf("Naive difference (Y|D=1 - Y|D=0): %.2f\n", naive_ate))

# 2. PROPENSITY SCORES & OVERLAP CHECK
ps_model <- glm(D ~ year_factor, data = df, family = binomial(link = "logit"))
df$ps <- predict(ps_model, type = "response")

cat(sprintf("\nPropensity score range: [%.4f, %.4f]\n", min(df$ps), max(df$ps)))
if (min(df$ps) > 0.01 && max(df$ps) < 0.99) {
  cat("Overlap assumption appears satisfied.\n")
} else {
  cat("Warning: Possible overlap violation (ps close to 0 or 1).\n")
}

# 3. ADJUSTMENT FORMULA (G-FORMULA)
reg_model <- lm(Y ~ D + year_factor, data = df)

df_1 <- df %>% mutate(D = 1)
df_0 <- df %>% mutate(D = 0)

mu1_hat <- predict(reg_model, newdata = df_1)
mu0_hat <- predict(reg_model, newdata = df_0)

ate_gformula <- mean(mu1_hat - mu0_hat)
cat(sprintf("\nATE via Adjustment Formula (G-formula): %.2f\n", ate_gformula))

# 4. INVERSE PROBABILITY WEIGHTING (IPW)
eps <- 1e-6
df <- df %>%
  mutate(ps_clipped = pmin(pmax(ps, eps), 1 - eps))

ate_ipw <- (
  sum(df$D * df$Y / df$ps_clipped) / sum(df$D / df$ps_clipped) -
  sum((1 - df$D) * df$Y / (1 - df$ps_clipped)) / sum((1 - df$D) / (1 - df$ps_clipped))
)
cat(sprintf("ATE via IPW (normalized): %.2f\n", ate_ipw))

# 5. MEASUREMENT ERROR DIAGNOSTIC (requires Part B output)
stance_path <- "day1/data/speeches_with_stance_R.csv"
if (file.exists(stance_path)) {
  df_llm <- read_csv(stance_path, show_col_types = FALSE)

  if ("Y_tilde" %in% names(df_llm)) {
    corr <- cor(df_llm$Y_tilde, df_llm$D, use = "complete.obs")

    cat(sprintf("\nMeasurement Error Diagnostic:\n"))
    cat(sprintf("Correlation between LLM stance proxy (Y_tilde) and Treatment (D): %.4f\n", corr))

    if (abs(corr) > 0.05) {
      cat("-> Evidence of non-classical measurement error (proxy is correlated with treatment).\n")
    }

    ate_llm_naive <- mean(df_llm$Y_tilde[df_llm$D == 1], na.rm = TRUE) -
                     mean(df_llm$Y_tilde[df_llm$D == 0], na.rm = TRUE)

    llm_reg <- lm(Y_tilde ~ D + as.factor(year), data = df_llm)
    ate_llm_adj <- coef(llm_reg)["D"]

    cat(sprintf("Naive LLM ATE: %.4f\n", ate_llm_naive))
    cat(sprintf("Adjusted LLM ATE: %.4f\n", ate_llm_adj))
  } else {
    cat("\n(speeches_with_stance_R.csv found but Y_tilde column missing — re-run partB first.)\n")
  }
} else {
  cat("\n(Run partB_llm_measurement.R first to generate speeches_with_stance_R.csv)\n")
}
