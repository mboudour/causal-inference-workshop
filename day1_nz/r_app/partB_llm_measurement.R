library(readr)
library(dplyr)

# Fast seminar version:
# reuse the already-working Python output instead of calling reticulate / transformers.

infile  <- "day1_nz/data/llm/speeches_with_stance.csv"
outfile <- "day1_nz/data/llm/speeches_with_stance_R.csv"

if (!file.exists(infile)) {
  stop("Missing file: day1_nz/data/llm/speeches_with_stance.csv. Run the Python Part B first.")
}

df <- read_csv(infile, show_col_types = FALSE)
write_csv(df, outfile)

cat("Copied Python Part B output to R output file.\n")
cat("Saved:", outfile, "\n")

if ("Y_tilde" %in% names(df) && "party" %in% names(df)) {
  df <- df %>% mutate(D = ifelse(party == "Republican", 1L, 0L))
  ate_llm <- mean(df$Y_tilde[df$D == 1], na.rm = TRUE) -
             mean(df$Y_tilde[df$D == 0], na.rm = TRUE)
  cat("LLM-based difference (Republican - Democrat):", ate_llm, "\n")
}
