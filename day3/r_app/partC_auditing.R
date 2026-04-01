# Day 3 - Part C: Auditing LLMs as Causal Systems
# Fully self-contained: no Python dependency.
# Implements the causal auditing framework:
#   - Simulate a biased LLM model
#   - Compute individual-level causal effects Delta_i via counterfactual inputs
#   - Compute Average Causal Bias (ACB) and subgroup ACB
#   - Simulate prompt sensitivity

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

cat("--- Day 3 Part C: Auditing LLMs as Causal Systems ---\n")

data_path <- "day1_package/data/speeches_sample.csv"
out_path  <- "day3/data/audit_results_R.csv"

dir.create("day3/data", showWarnings = FALSE, recursive = TRUE)

if (!file.exists(data_path)) {
  stop(paste("Error:", data_path, "not found."))
}

df <- read_csv(data_path, show_col_types = FALSE) %>% mutate(row_id = row_number())
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
X_text <- create_dtm(tokens, vect)

# -----------------------------------------------------------------------
# Simulate true Y and biased LLM model
# -----------------------------------------------------------------------
set.seed(2026)
proj <- rnorm(ncol(X_text))
f_X  <- as.vector(X_text %*% proj)
f_X  <- (f_X - mean(f_X)) / sd(f_X)

Y_true <- 0.3 * D + f_X + rnorm(n, 0, 0.3)

# Party signal (simulates LLM picking up party cues)
party_signal         <- D * 0.8 + rnorm(n, 0, 0.1)
party_signal_flipped <- (1 - D) * 0.8 + rnorm(n, 0, 0.1)

X_full         <- cbind(X_text, party_signal = party_signal)
X_full_flipped <- cbind(X_text, party_signal = party_signal_flipped)

# Train biased model on full features
cv_model <- cv.glmnet(X_full, Y_true, alpha = 0, nfolds = 5, parallel = FALSE)

Y_hat       <- predict(cv_model, newx = X_full,         s = "lambda.min")[, 1]
Y_hat_prime <- predict(cv_model, newx = X_full_flipped, s = "lambda.min")[, 1]

cat(sprintf("\nModel trained. Mean prediction: %.4f\n", mean(Y_hat)))

# -----------------------------------------------------------------------
# Individual-level causal effects
# -----------------------------------------------------------------------
Delta <- Y_hat - Y_hat_prime

ACB    <- mean(Delta)
ACB_var <- var(Delta)
ACB_se  <- sd(Delta) / sqrt(n)
CI_lower <- ACB - 1.96 * ACB_se
CI_upper <- ACB + 1.96 * ACB_se

cat(sprintf("\nAverage Causal Bias (ACB): %.4f\n", ACB))
cat(sprintf("Variance of Delta_i:       %.4f\n", ACB_var))
cat(sprintf("95%% CI for ACB:            [%.4f, %.4f]\n", CI_lower, CI_upper))

# Subgroup ACB
ACB_R <- mean(Delta[D == 1])
ACB_D <- mean(Delta[D == 0])
cat(sprintf("\nSubgroup ACB (Republicans): %.4f\n", ACB_R))
cat(sprintf("Subgroup ACB (Democrats):   %.4f\n", ACB_D))

# -----------------------------------------------------------------------
# Prompt sensitivity simulation
# -----------------------------------------------------------------------
cat("\n--- Prompt Sensitivity Simulation ---\n")
prompt_configs <- list(
  list(name = "Zero-shot (high noise)", noise = 0.5, bias_mult = 1.0),
  list(name = "Few-shot balanced",      noise = 0.1, bias_mult = 0.5),
  list(name = "Chain-of-Thought",       noise = 0.05, bias_mult = 1.5)
)

for (cfg in prompt_configs) {
  ps      <- D * 0.8 * cfg$bias_mult + rnorm(n, 0, cfg$noise)
  ps_flip <- (1 - D) * 0.8 * cfg$bias_mult + rnorm(n, 0, cfg$noise)

  X_p      <- cbind(X_text, party_signal = ps)
  X_p_flip <- cbind(X_text, party_signal = ps_flip)

  Y_p      <- predict(cv_model, newx = X_p,      s = "lambda.min")[, 1]
  Y_p_flip <- predict(cv_model, newx = X_p_flip, s = "lambda.min")[, 1]

  acb_p <- mean(Y_p - Y_p_flip)
  cat(sprintf("  %-35s: ACB = %.4f\n", cfg$name, acb_p))
}

# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
df_out <- df %>%
  select(speaker, party, year) %>%
  mutate(
    Y_hat       = Y_hat,
    Y_hat_prime = Y_hat_prime,
    Delta_i     = Delta
  )

write_csv(df_out, out_path)
cat(sprintf("\nIndividual-level results saved to %s\n", out_path))

summary_df <- data.frame(
  metric = c("ACB", "Var_Delta", "ACB_SE", "CI_lower", "CI_upper",
             "ACB_Republicans", "ACB_Democrats"),
  value  = c(ACB, ACB_var, ACB_se, CI_lower, CI_upper, ACB_R, ACB_D)
)
summary_path <- "day3/data/audit_summary_R.csv"
write_csv(summary_df, summary_path)
cat(sprintf("Summary statistics saved to %s\n", summary_path))
