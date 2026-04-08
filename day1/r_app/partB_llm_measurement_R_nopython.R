# Day 1 - Part B: LLM Measurement (Python-free R version)
# Matches the local folder structure where Day 1 data live under day1/data/llm/.
# Uses text2vec TF-IDF features and a fixed random projection to construct
# a deterministic stance proxy without any Python dependency.

for (pkg in c("readr", "dplyr", "text2vec")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(readr)
library(dplyr)
library(text2vec)

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, winslash = "/", mustWork = FALSE))
  }
  return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
}

find_root_with_day1_llm <- function() {
  script_path <- get_script_path()
  script_dir <- if (dir.exists(script_path)) script_path else dirname(script_path)

  candidates <- c(
    normalizePath(file.path(script_dir, "..", ".."), winslash = "/", mustWork = FALSE),
    normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  )

  for (root in unique(candidates)) {
    if (file.exists(file.path(root, "day1", "data", "llm", "speeches_sample.csv"))) {
      return(root)
    }
  }

  stop(
    paste(
      "Could not find the local Day 1 data file at day1/data/llm/speeches_sample.csv.",
      "Run this script from the parent folder that contains day1/, or keep the script inside day1/r_app/."
    )
  )
}

root_dir <- find_root_with_day1_llm()
resolve_path <- function(...) file.path(root_dir, ...)

cat("--- Day 1 Part B: LLM Measurement (Python-free R version) ---\n")
cat(sprintf("Working from root: %s\n", root_dir))

data_path  <- resolve_path("day1", "data", "llm", "speeches_sample.csv")
out_path   <- resolve_path("day1", "data", "llm", "speeches_with_stance_R.csv")
cache_path <- resolve_path("day1", "data", "llm", "embeddings_cache_R.rds")

df <- read_csv(data_path, show_col_types = FALSE)
cat(sprintf("Loaded %d speeches.\n", nrow(df)))

if (!"text" %in% names(df)) stop("Dataset must contain a 'text' column.")
if (!"party" %in% names(df)) stop("Dataset must contain a 'party' column.")

df <- df %>%
  filter(party %in% c("Republican", "Democrat")) %>%
  mutate(D = ifelse(party == "Republican", 1L, 0L))

cat(sprintf("Retained %d speeches after party filtering.\n", nrow(df)))

if (file.exists(cache_path)) {
  cat("Loading cached TF-IDF representation...\n")
  embeddings <- readRDS(cache_path)
} else {
  cat("Cache not found. Building TF-IDF representation...\n")

  it <- itoken(
    df$text,
    preprocessor = tolower,
    tokenizer = word_tokenizer,
    progressbar = FALSE
  )

  vocab <- create_vocabulary(it)
  vocab <- prune_vocabulary(vocab, term_count_min = 5L)
  vectorizer <- vocab_vectorizer(vocab)
  dtm <- create_dtm(it, vectorizer)

  tfidf <- TfIdf$new()
  embeddings <- fit_transform(dtm, tfidf)

  saveRDS(embeddings, cache_path)
  cat(sprintf("Saved TF-IDF cache to %s\n", cache_path))
}

set.seed(42)
projection_vector <- rnorm(ncol(embeddings))
raw_stance <- as.vector(embeddings %*% projection_vector)
stance <- as.numeric(scale(raw_stance))

df <- df %>%
  mutate(
    stance = stance,
    Y_tilde = stance
  )

cat("\nSample of computed stance scores:\n")
print(head(df %>% select(any_of(c("speaker", "party")), Y_tilde)))

ate_llm <- mean(df$Y_tilde[df$D == 1], na.rm = TRUE) -
  mean(df$Y_tilde[df$D == 0], na.rm = TRUE)
cat(sprintf("\nLLM-based difference (Republican - Democrat): %.4f\n", ate_llm))

write_csv(df, out_path)
cat(sprintf("Saved dataset with stance scores to %s\n", out_path))
