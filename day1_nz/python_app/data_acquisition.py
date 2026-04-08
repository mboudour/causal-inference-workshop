import pandas as pd
from datasets import load_dataset

# --- STEP 1: Load REAL political dataset (AG News - political subset) ---

dataset = load_dataset("ag_news")

df = pd.DataFrame(dataset["train"])

# --- STEP 2: Keep only politics category (label = 0 = World / politics-heavy) ---

df = df[df["label"] == 0].copy()

# --- STEP 3: adapt structure ---

df = df.rename(columns={"text": "text"})
df["speech_id"] = range(len(df))

# create pseudo-party from title keywords (weak but real text)
df["party"] = df["text"].apply(lambda x: "Democrat" if "government" in x.lower() else "Republican")

df["speaker"] = "unknown"
df["year"] = 2020

df = df[["speech_id", "text", "party", "speaker", "year"]]

# --- STEP 4: filter + sample ---

df = df[df["text"].str.len() > 50].head(3000)

# --- STEP 5: save ---

df.to_csv("day1/data/llm/speeches_sample.csv", index=False)

print("Real political dataset saved:", len(df), "rows")