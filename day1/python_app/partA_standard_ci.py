import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- LOAD DATA ---
df = pd.read_csv("day1/data/llm/speeches_sample.csv")

# --- OUTCOME ---
df["Y"] = df["text"].str.len()

# --- TREATMENT ---
df["D"] = (df["party"] == "Republican").astype(int)

# --- NAIVE ESTIMATE ---
ate_naive = df[df["D"] == 1]["Y"].mean() - df[df["D"] == 0]["Y"].mean()
print("Naive difference (Republican - Democrat):", ate_naive)

# --- REGRESSION (CONTROL FOR YEAR AS CATEGORICAL) ---
# Year is treated as a categorical variable (factor) for consistency with
# the propensity score model and G-formula in Part C.
year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
X = pd.concat([df[["D"]], year_dummies], axis=1).astype(float)
X = sm.add_constant(X)

model = sm.OLS(df["Y"], X).fit()
print(model.summary())

# --- SAVE RESULTS ---
with open("day1/data/standard/results_partA.txt", "w") as f:
    f.write(f"Naive difference: {ate_naive}\n\n")
    f.write(model.summary().as_text())
