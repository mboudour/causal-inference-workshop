# Day 3 - Part A: Double Machine Learning (DML)
# Fully self-contained: no Python dependency.
# Uses text2vec for TF-IDF features and glmnet for Ridge/Logistic nuisance models.
# Implements 5-fold cross-fitting DML for the partially linear model.

# Auto-install required packages
for (pkg in c("readr", "dplyr", "text2vec", "glmnet", "Matrix")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(text2vec)
  library(glmnet)
  library(Matrix)
})

cat("--- Day 3 Part A: Double Machine Learning (DML) ---\n")

data_path <- "day1/data/speeches_sample.csv"
out_path  <- "day3/data/dml_results_R.csv"

dir.create("day3/data", showWarnings = FALSE, recursive = TRUE)

if (file.exists(data_path)) {
  df <- read_csv(data_path, show_col_types = FALSE)
  cat(sprintf("Loaded %d speeches from %s\n", nrow(df), data_path))
} else {
  cat(sprintf("Note: %s not found. Generating synthetic data.\n", data_path))
  cat("  (Run 'git pull origin main' from seminar_computations/ to get the real data.)\n")
  set.seed(123)
  n_synth  <- 2000
  rep_pool <- c("government", "budget", "deficit", "spending", "tax", "reform",
                "healthcare", "military", "security", "border", "freedom",
                "liberty", "constitution", "economy", "jobs", "growth",
                "business", "regulation", "energy", "bill", "amendment")
  dem_pool <- c("healthcare", "education", "climate", "environment", "equality",
                "rights", "workers", "union", "wage", "social", "medicare",
                "medicaid", "infrastructure", "investment", "community",
                "diversity", "inclusion", "committee", "states", "amendment")
  common   <- c("the", "that", "this", "with", "have", "will", "from", "they",
                "their", "about", "would", "more", "people", "which", "care")
  make_speech <- function(pool, n) {
    words <- c(pool, common, common)
    sapply(seq_len(n), function(i)
      paste(sample(words, sample(60:180, 1), replace = TRUE), collapse = " "))
  }
  n_rep <- round(n_synth * 0.40)
  n_dem <- n_synth - n_rep
  df <- data.frame(
    speech_id = seq_len(n_synth),
    text      = c(make_speech(rep_pool, n_rep), make_speech(dem_pool, n_dem)),
    speaker   = paste0("Speaker_", seq_len(n_synth)),
    party     = c(rep("Republican", n_rep), rep("Democrat", n_dem)),
    date      = sample(c("2009-01-01", "2010-01-01"), n_synth, replace = TRUE),
    year      = sample(c(2009L, 2010L), n_synth, replace = TRUE),
    stringsAsFactors = FALSE
  )
  df <- df[sample(nrow(df)), ]
  cat(sprintf("Generated %d synthetic speeches (%d R, %d D).\n",
              n_synth, n_rep, n_dem))
}

# Treatment
D <- as.integer(df$party == "Republican")
n <- nrow(df)

# -----------------------------------------------------------------------
# Build TF-IDF features via text2vec
# -----------------------------------------------------------------------
cat("Building TF-IDF features...\n")
tokens <- itoken(df$text, preprocessor = tolower,
                 tokenizer = word_tokenizer, progressbar = FALSE)
vocab  <- create_vocabulary(tokens, ngram = c(1L, 2L))
vocab  <- prune_vocabulary(vocab, term_count_min = 5L, vocab_term_max = 3000L)
vect   <- vocab_vectorizer(vocab)
dtm    <- create_dtm(tokens, vect)

# Simulate deterministic outcome Y (same seed as Python scripts)
set.seed(2026)
proj <- rnorm(ncol(dtm))
raw  <- as.vector(dtm %*% proj)
Y    <- (raw - mean(raw)) / sd(raw)

cat(sprintf("Feature matrix: %d x %d\n", nrow(dtm), ncol(dtm)))

# -----------------------------------------------------------------------
# Naive difference-in-means
# -----------------------------------------------------------------------
naive_ate <- mean(Y[D == 1]) - mean(Y[D == 0])
cat(sprintf("\nNaive Difference-in-Means: %.4f\n", naive_ate))

# -----------------------------------------------------------------------
# OLS with year dummies (baseline)
# -----------------------------------------------------------------------
year_dummies <- model.matrix(~ factor(year) - 1, data = df)
X_ols <- cbind(D = D, year_dummies)
ols   <- lm(Y ~ X_ols)
ols_ate <- coef(ols)["X_olsD"]
ols_se  <- summary(ols)$coefficients["X_olsD", "Std. Error"]
cat(sprintf("OLS (year dummies only): %.4f (SE: %.4f)\n", ols_ate, ols_se))

# -----------------------------------------------------------------------
# DML with 5-fold cross-fitting
# -----------------------------------------------------------------------
cat("\nRunning DML with 5-fold cross-fitting...\n")

K <- 5
set.seed(42)
fold_ids <- sample(rep(1:K, length.out = n))

Y_res <- numeric(n)
D_res <- numeric(n)

for (k in 1:K) {
  train_idx <- which(fold_ids != k)
  test_idx  <- which(fold_ids == k)

  X_train <- dtm[train_idx, ]
  X_test  <- dtm[test_idx,  ]
  Y_train <- Y[train_idx]
  D_train <- D[train_idx]

  # Ridge regression for g(X) = E[Y | X]
  cv_ridge <- cv.glmnet(X_train, Y_train, alpha = 0, nfolds = 3, parallel = FALSE)
  g_hat    <- predict(cv_ridge, newx = X_test, s = "lambda.min")[, 1]

  # Logistic regression for e(X) = P(D=1 | X)
  cv_logit <- cv.glmnet(X_train, D_train, family = "binomial",
                         alpha = 0, nfolds = 3, parallel = FALSE)
  e_hat    <- predict(cv_logit, newx = X_test, s = "lambda.min",
                      type = "response")[, 1]
  e_hat    <- pmin(pmax(e_hat, 0.05), 0.95)

  Y_res[test_idx] <- Y[test_idx] - g_hat
  D_res[test_idx] <- D[test_idx] - e_hat
}

# DML point estimate
theta_dml <- sum(D_res * Y_res) / sum(D_res^2)

# Influence-function standard error
psi    <- D_res * (Y_res - theta_dml * D_res)
se_dml <- sqrt(mean(psi^2) / (mean(D_res^2)^2) / n)

ci_lower <- theta_dml - 1.96 * se_dml
ci_upper <- theta_dml + 1.96 * se_dml

cat(sprintf("DML Estimate (theta): %.4f\n", theta_dml))
cat(sprintf("Standard Error:       %.4f\n", se_dml))
cat(sprintf("95%% CI:               [%.4f, %.4f]\n", ci_lower, ci_upper))

# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
results <- data.frame(
  estimator = c("Naive DiM", "OLS (year dummies)", "DML (text features, K=5)"),
  estimate  = c(naive_ate, ols_ate, theta_dml),
  se        = c(NA, ols_se, se_dml),
  ci_lower  = c(NA, ols_ate - 1.96 * ols_se, ci_lower),
  ci_upper  = c(NA, ols_ate + 1.96 * ols_se, ci_upper)
)

write_csv(results, out_path)
cat(sprintf("\nResults saved to %s\n", out_path))

cat("\nSummary:\n")
print(results)
