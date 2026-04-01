# Day 3 - Part B: Design-Based Supervised Learning (DSL)
# Fully self-contained: no Python dependency.
# Simulates true Y and measured Y_tilde with non-classical error,
# then applies DSL correction using a labeled subsample.

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

cat("--- Day 3 Part B: Design-Based Supervised Learning (DSL) ---\n")

data_path <- "day1/data/speeches_sample.csv"
out_path  <- "day3/data/dsl_results_R.csv"

dir.create("day3/data", showWarnings = FALSE, recursive = TRUE)

if (!file.exists(data_path)) {
  stop(paste("Error:", data_path, "not found."))
}

df <- read_csv(data_path, show_col_types = FALSE)
cat(sprintf("Loaded %d speeches.\n", nrow(df)))

D <- as.integer(df$party == "Republican")
n <- nrow(df)

# -----------------------------------------------------------------------
# Build TF-IDF features
# -----------------------------------------------------------------------
cat("Building TF-IDF features...\n")
tokens <- itoken(df$text, preprocessor = tolower,
                 tokenizer = word_tokenizer, progressbar = FALSE)
vocab  <- create_vocabulary(tokens, ngram = c(1L, 2L))
vocab  <- prune_vocabulary(vocab, term_count_min = 5L, vocab_term_max = 2000L)
vect   <- vocab_vectorizer(vocab)
X_feat <- create_dtm(tokens, vect)

# -----------------------------------------------------------------------
# Simulate true Y and measured Y_tilde
# -----------------------------------------------------------------------
set.seed(2026)
proj <- rnorm(ncol(X_feat))
f_X  <- as.vector(X_feat %*% proj)
f_X  <- (f_X - mean(f_X)) / sd(f_X)

Y_true  <- 0.3 * D + f_X + rnorm(n, 0, 0.5)
epsilon <- rnorm(n, 0, 0.5) + 0.5 * D   # non-classical: depends on D
Y_tilde <- Y_true + epsilon

# -----------------------------------------------------------------------
# Split into labeled (L) and unlabeled (U)
# -----------------------------------------------------------------------
label_frac  <- 0.1
labeled_idx <- sample(seq_len(n), size = floor(n * label_frac), replace = FALSE)
unlabeled_idx <- setdiff(seq_len(n), labeled_idx)

cat(sprintf("\nLabeled sample size:   %d\n", length(labeled_idx)))
cat(sprintf("Unlabeled sample size: %d\n", length(unlabeled_idx)))

# -----------------------------------------------------------------------
# Naive ATE on U (biased Y_tilde)
# -----------------------------------------------------------------------
Y_u <- Y_tilde[unlabeled_idx]
D_u <- D[unlabeled_idx]
naive_ate <- mean(Y_u[D_u == 1]) - mean(Y_u[D_u == 0])
cat(sprintf("\nNaive ATE on U (biased Y_tilde): %.4f\n", naive_ate))
cat(sprintf("True ATE (known in simulation):  0.3000\n"))

# -----------------------------------------------------------------------
# DSL: learn correction m(X, D, Y_tilde) on L, apply to U
# -----------------------------------------------------------------------
Y_L       <- Y_true[labeled_idx]
Y_tilde_L <- Y_tilde[labeled_idx]
D_L       <- D[labeled_idx]
X_L       <- X_feat[labeled_idx, ]

# Correction features: [X_text, D, Y_tilde]
X_corr_L <- cbind(X_L, D = D_L, Y_tilde = Y_tilde_L)

cv_corr <- cv.glmnet(X_corr_L, Y_L, alpha = 0, nfolds = 5, parallel = FALSE)

# Apply to U
Y_tilde_U <- Y_tilde[unlabeled_idx]
D_U       <- D[unlabeled_idx]
X_U       <- X_feat[unlabeled_idx, ]
X_corr_U  <- cbind(X_U, D = D_U, Y_tilde = Y_tilde_U)

Y_corr_U <- predict(cv_corr, newx = X_corr_U, s = "lambda.min")[, 1]

dsl_ate <- mean(Y_corr_U[D_u == 1]) - mean(Y_corr_U[D_u == 0])
cat(sprintf("DSL-corrected ATE on U:          %.4f\n", dsl_ate))

# Oracle ATE
Y_true_U <- Y_true[unlabeled_idx]
oracle_ate <- mean(Y_true_U[D_u == 1]) - mean(Y_true_U[D_u == 0])
cat(sprintf("Oracle ATE on U (true Y):        %.4f\n", oracle_ate))

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
cat("\nSummary:\n")
results <- data.frame(
  estimator = c("Naive (biased Y_tilde)", "DSL-corrected", "Oracle (true Y)"),
  ate       = c(naive_ate, dsl_ate, oracle_ate),
  bias      = c(naive_ate - oracle_ate, dsl_ate - oracle_ate, 0.0)
)
print(results)

write_csv(results, out_path)
cat(sprintf("\nResults saved to %s\n", out_path))
