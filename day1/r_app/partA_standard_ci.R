# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

library(readr)
library(dplyr)

# --- LOAD DATA ---
df <- read_csv("day1/data/llm/speeches_sample.csv", show_col_types = FALSE)

# --- OUTCOME ---
df <- df %>%
  mutate(
    Y = nchar(text),
    D = ifelse(party == "Republican", 1, 0),
    year_factor = as.factor(year)   # year as categorical, consistent with Part C
  )

# --- NAIVE ESTIMATE ---
ate_naive <- mean(df$Y[df$D == 1]) - mean(df$Y[df$D == 0])
cat("Naive difference (Republican - Democrat):", ate_naive, "\n")

# --- REGRESSION (CONTROL FOR YEAR AS CATEGORICAL) ---
# Year is treated as a factor for consistency with the propensity score
# model and G-formula in Part C.
model <- lm(Y ~ D + year_factor, data = df)
summary(model)
