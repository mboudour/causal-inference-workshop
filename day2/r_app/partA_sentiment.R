# Day 2 - Part A: Measurement with LLMs (Sentiment Score)
# Fully self-contained R implementation.
# Uses text2vec for TF-IDF embeddings and a fixed random projection
# to simulate an LLM sentiment score. No Python dependency.

for (pkg in c("readr", "dplyr", "text2vec")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
library(readr)
library(dplyr)
library(text2vec)

cat("--- Day 2 Part A: LLM Measurement (Sentiment) ---\n")

data_path <- "day1_package/data/speeches_sample.csv"
out_path  <- "day2/data/speeches_with_sentiment_R.csv"

# Ensure output directory exists
dir.create("day2/data", showWarnings = FALSE, recursive = TRUE)

if (!file.exists(data_path)) {
  stop(paste("Error:", data_path, "not found."))
}

df <- read_csv(data_path, show_col_types = FALSE)
cat(sprintf("Loaded %d speeches.\n", nrow(df)))

# ── Build TF-IDF document-term matrix ────────────────────────────────────────
cat("Building TF-IDF embeddings...\n")
it    <- itoken(df$text, preprocessor = tolower,
                tokenizer = word_tokenizer, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
vect  <- vocab_vectorizer(vocab)
dtm   <- create_dtm(it, vect)

tfidf      <- TfIdf$new()
dtm_tfidf  <- fit_transform(dtm, tfidf)

# Convert to dense matrix for projection
embeddings <- as.matrix(dtm_tfidf)

# ── Simulate LLM sentiment via fixed random projection ───────────────────────
# Seed 2026 matches Python script; different from Day 1 (seed 42) so
# sentiment != stance.
set.seed(2026)
projection_vector <- rnorm(ncol(embeddings))
raw_sentiment     <- as.vector(embeddings %*% projection_vector)

# Standardize to mean 0, sd 1
sentiment_score <- (raw_sentiment - mean(raw_sentiment)) / sd(raw_sentiment)

df <- df %>%
  mutate(
    sentiment_score = sentiment_score,
    Y_tilde         = sentiment_score,
    D               = ifelse(party == "Republican", 1L, 0L)
  )

cat("\nSample of computed sentiment scores:\n")
print(head(df %>% select(speaker, party, Y_tilde)))

# Naive difference in means
mean_R     <- mean(df$Y_tilde[df$D == 1])
mean_D     <- mean(df$Y_tilde[df$D == 0])
naive_diff <- mean_R - mean_D

cat(sprintf("\nNaive Difference-in-Means (R - D) on Sentiment: %.4f\n", naive_diff))

write_csv(df, out_path)
cat(sprintf("\nSaved dataset with sentiment scores to %s\n", out_path))
