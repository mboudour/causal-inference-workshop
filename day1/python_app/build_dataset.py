import pandas as pd

# --- STEP 1: Load speeches ---

df = pd.read_csv("day1/data/raw/hein_bound_111_raw/raw_111_speeches.csv")

# --- STEP 2: Keep relevant columns ---

df = df[["speech_id", "text", "speaker", "party", "date"]]

# --- STEP 3: Filter valid parties ---

df = df[df["party"].isin(["D", "R"])]

df["party"] = df["party"].map({
    "D": "Democrat",
    "R": "Republican"
})

# --- STEP 4: Filter meaningful text ---

df = df[df["text"].str.len() > 100]

# --- STEP 5: Sample manageable size ---

df = df.sample(n=4000, random_state=42)

# --- STEP 6: Extract year ---

df["year"] = df["date"].astype(str).str[:4]

# --- STEP 7: Save clean dataset ---

df.to_csv("day1/data/llm/speeches_sample.csv", index=False)

print("Final dataset ready:", len(df))