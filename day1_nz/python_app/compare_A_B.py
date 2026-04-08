import pandas as pd

# --- LOAD ---

# dfA = pd.read_csv("day1/data/llm/speeches_sample.csv")
# dfB = pd.read_csv("day1/data/llm/speeches_with_stance.csv")
dfA = pd.read_csv("day1_nz/data/llm/speeches_sample.csv")
dfB = pd.read_csv("day1_nz/data/llm/speeches_with_stance.csv")

# --- PART A outcome ---

dfA["Y"] = dfA["text"].str.len()
dfA["D"] = (dfA["party"] == "Republican").astype(int)

ate_A = dfA[dfA["D"] == 1]["Y"].mean() - dfA[dfA["D"] == 0]["Y"].mean()

# --- PART B outcome ---

dfB["D"] = (dfB["party"] == "Republican").astype(int)

ate_B = dfB[dfB["D"] == 1]["Y_tilde"].mean() - dfB[dfB["D"] == 0]["Y_tilde"].mean()

print("\nPART A (Observed proxy - text length):", ate_A)
print("PART B (LLM-measured stance):", ate_B)