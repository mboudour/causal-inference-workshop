# Day 2 - Part B: Causal Estimators
# Implements Diff-in-Means, Regression, G-formula, Matching, IPW, AIPW

if (!requireNamespace("MatchIt", quietly = TRUE)) install.packages("MatchIt", repos = "https://cloud.r-project.org")
library(readr)
library(dplyr)
library(MatchIt)

cat("--- Day 2 Part B: Causal Estimators ---\n")

data_path <- "day2/data/speeches_with_sentiment_R.csv"
if (!file.exists(data_path)) {
  stop(paste("Error:", data_path, "not found. Run partA_sentiment.R first."))
}

df            <- read_csv(data_path, show_col_types = FALSE)
df$year_factor <- as.factor(df$year)

Y <- df$Y_tilde
D <- df$D

# ── 1. Difference-in-Means ────────────────────────────────────────────────────
cat("\n1. Difference-in-Means Estimator\n")
treated  <- Y[D == 1]
control  <- Y[D == 0]
n1       <- length(treated)
n0       <- length(control)

est_diff <- mean(treated) - mean(control)
se_diff  <- sqrt(var(treated)/n1 + var(control)/n0)
ci_diff  <- c(est_diff - 1.96 * se_diff, est_diff + 1.96 * se_diff)

cat(sprintf("ATE Estimate: %.4f\n",            est_diff))
cat(sprintf("Standard Error: %.4f\n",           se_diff))
cat(sprintf("95%% CI: [%.4f, %.4f]\n",          ci_diff[1], ci_diff[2]))

# ── 2. Regression Adjustment ──────────────────────────────────────────────────
cat("\n2. Regression Adjustment\n")
model_reg <- lm(Y_tilde ~ D + year_factor, data = df)
est_reg   <- coef(model_reg)["D"]
se_reg    <- summary(model_reg)$coefficients["D", "Std. Error"]
ci_reg    <- confint(model_reg)["D", ]

cat(sprintf("ATE Estimate: %.4f\n",   est_reg))
cat(sprintf("Standard Error: %.4f\n", se_reg))
cat(sprintf("95%% CI: [%.4f, %.4f]\n", ci_reg[1], ci_reg[2]))

# ── 3. G-formula ──────────────────────────────────────────────────────────────
cat("\n3. G-formula (Plug-in Estimator)\n")
df1 <- df; df1$D <- 1
df0 <- df; df0$D <- 0

m1_hat   <- predict(model_reg, newdata = df1)
m0_hat   <- predict(model_reg, newdata = df0)
est_gform <- mean(m1_hat - m0_hat)

cat(sprintf("ATE Estimate: %.4f\n", est_gform))

# ── 4. Nearest-Neighbour Matching (ATT) ───────────────────────────────────────
cat("\n4. Nearest-Neighbour Matching (ATT)\n")
m_out   <- matchit(D ~ year_factor, data = df, method = "nearest", ratio = 1)
m_data  <- match.data(m_out)

treated_m <- m_data$Y_tilde[m_data$D == 1]
control_m <- m_data$Y_tilde[m_data$D == 0]
est_match <- mean(treated_m) - mean(control_m)

cat(sprintf("ATT Estimate: %.4f\n", est_match))
cat("Note: Matching is performed on year only. With a single weak covariate,\n")
cat("matched pairs may be poorly balanced. In practice, richer covariates are needed.\n")

# ── 5. IPW ────────────────────────────────────────────────────────────────────
cat("\n5. Inverse Probability Weighting (IPW)\n")
model_ps <- glm(D ~ year_factor, data = df, family = binomial)
e_hat    <- predict(model_ps, type = "response")
e_hat    <- pmax(pmin(e_hat, 0.95), 0.05)

w1      <- D / e_hat
w0      <- (1 - D) / (1 - e_hat)
est_ipw <- sum(Y * w1) / sum(w1) - sum(Y * w0) / sum(w0)

cat(sprintf("ATE Estimate: %.4f\n", est_ipw))

# ── 6. AIPW (Doubly Robust) ───────────────────────────────────────────────────
cat("\n6. Augmented IPW (Doubly Robust)\n")
term1       <- m1_hat - m0_hat
term2       <- (D * (Y - m1_hat)) / e_hat
term3       <- ((1 - D) * (Y - m0_hat)) / (1 - e_hat)

aipw_scores <- term1 + term2 - term3
est_aipw    <- mean(aipw_scores)
se_aipw     <- sd(aipw_scores) / sqrt(length(Y))
ci_aipw     <- c(est_aipw - 1.96 * se_aipw, est_aipw + 1.96 * se_aipw)

cat(sprintf("ATE Estimate: %.4f\n",   est_aipw))
cat(sprintf("Standard Error: %.4f\n", se_aipw))
cat(sprintf("95%% CI: [%.4f, %.4f]\n", ci_aipw[1], ci_aipw[2]))

# ── Summary ───────────────────────────────────────────────────────────────────
cat("\nSummary of ATE Estimates:\n")
cat(strrep("-", 30), "\n")
cat(sprintf("Diff-in-Means : %8.4f\n", est_diff))
cat(sprintf("Regression    : %8.4f\n", est_reg))
cat(sprintf("G-formula     : %8.4f\n", est_gform))
cat(sprintf("Matching (ATT): %8.4f\n", est_match))
cat(sprintf("IPW           : %8.4f\n", est_ipw))
cat(sprintf("AIPW          : %8.4f\n", est_aipw))
cat(strrep("-", 30), "\n")
