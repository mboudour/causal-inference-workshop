library(readr)
library(dplyr)

# --- LOAD DATA ---
df <- read_csv("day1_nz/data/llm/speeches_sample.csv", show_col_types = FALSE)

# --- OUTCOME ---
df <- df %>%
  mutate(
    Y = nchar(text),
    D = ifelse(party == "Republican", 1, 0),
    year_factor = as.factor(year)
  )

# --- NAIVE ESTIMATE ---
ate_naive <- mean(df$Y[df$D == 1]) - mean(df$Y[df$D == 0])
cat("Naive difference (Republican - Democrat):", ate_naive, "\n")

# --- REGRESSION ADJUSTMENT ---
model <- lm(Y ~ D + year_factor, data = df)
print(summary(model))

# --- SAVE SHORT OUTPUT ---
out_lines <- c(
  paste("Naive difference (Republican - Democrat):", round(ate_naive, 4)),
  paste("Regression coefficient on D:", round(coef(model)["D"], 4))
)
writeLines(out_lines, "day1_nz/data/standard/results_partA.txt")
cat("Saved: day1_nz/data/standard/results_partA.txt\n")