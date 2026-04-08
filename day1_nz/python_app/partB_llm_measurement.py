import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# --- LOAD DATA ---
df = pd.read_csv("day1_nz/data/llm/speeches_sample.csv")

# --- SIMPLE TEXT REPRESENTATION (NO TORCH, NO TRANSFORMERS) ---
texts = df["text"].fillna("").astype(str).tolist()

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    min_df=5
)
X = vectorizer.fit_transform(texts)

svd = TruncatedSVD(n_components=1, random_state=42)
stance = svd.fit_transform(X).ravel()
stance = (stance - stance.mean()) / stance.std()

# --- DEFINE VARIABLES ---
df["stance"] = stance
df["Y_tilde"] = df["stance"]
df["D"] = (df["party"] == "Republican").astype(int)

# --- NAIVE ESTIMATE ---
ate_llm = df[df["D"] == 1]["Y_tilde"].mean() - df[df["D"] == 0]["Y_tilde"].mean()
print("LLM-style text-derived difference (Republican - Democrat):", ate_llm)

# --- SAVE RESULTS ---
df.to_csv("day1_nz/data/llm/speeches_with_stance.csv", index=False)
print("Saved: day1_nz/data/llm/speeches_with_stance.csv")