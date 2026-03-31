# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

library(readr)
library(dplyr)
library(reticulate)

# --- PYTHON ENVIRONMENT ---
# Point reticulate to the same conda environment used for the Python scripts.
# Change the environment name below if yours differs.
use_condaenv("seminar_computations", required = TRUE)

# --- LOAD DATA ---
df <- read_csv("day1/data/llm/speeches_sample.csv", show_col_types = FALSE)

# --- EMBEDDING CACHE ---
# On first run, embeddings are computed via SentenceTransformer and saved to disk.
# On subsequent runs, the cached file is loaded directly,
# skipping the model download and encoding entirely.
cache_path <- "day1/data/llm/embeddings_cache.npy"

if (file.exists(cache_path)) {
  message("Loading embeddings from cache...")
  np <- import("numpy")
  embeddings <- np$load(cache_path)
} else {
  message("Cache not found. Computing embeddings (first run only)...")

  # Set environment variables for Intel Mac stability
  Sys.setenv(KMP_DUPLICATE_LIB_OK   = "TRUE")
  Sys.setenv(MKL_THREADING_LAYER    = "GNU")
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")
  Sys.setenv(OMP_NUM_THREADS        = "1")
  Sys.setenv(MKL_NUM_THREADS        = "1")

  st  <- import("sentence_transformers")
  np  <- import("numpy")

  model      <- st$SentenceTransformer("all-MiniLM-L6-v2", device = "cpu")
  embeddings <- model$encode(
    as.list(df$text),
    show_progress_bar = TRUE,
    batch_size        = 4L
  )

  np$save(cache_path, embeddings)
  message(paste("Embeddings saved to cache:", cache_path))
}

# --- CONSTRUCT STANCE ---
set.seed(42)
direction <- rnorm(ncol(embeddings))

stance <- as.numeric(embeddings %*% direction)
stance <- (stance - mean(stance)) / sd(stance)

df$stance  <- stance
df$Y_tilde <- stance
df$D       <- ifelse(df$party == "Republican", 1L, 0L)

# --- NAIVE ESTIMATE ---
ate_llm <- mean(df$Y_tilde[df$D == 1]) - mean(df$Y_tilde[df$D == 0])
cat("LLM-based difference (Republican - Democrat):", ate_llm, "\n")

# --- SAVE RESULTS ---
write_csv(df, "day1/data/llm/speeches_with_stance_R.csv")
cat("LLM labels saved.\n")
