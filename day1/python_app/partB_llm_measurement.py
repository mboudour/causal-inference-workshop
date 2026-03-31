# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys

# --- INTEL MAC STABILITY BLOCK ---
# These MUST be at the very top to prevent Segmentation Faults
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch

# --- LOAD DATA ---
df = pd.read_csv("day1/data/llm/speeches_sample.csv")

# --- EMBEDDING CACHE ---
# On first run, embeddings are computed and saved to disk.
# On subsequent runs, the cached file is loaded directly,
# skipping the model download and encoding entirely.
CACHE_PATH = "day1/data/llm/embeddings_cache.npy"

if os.path.exists(CACHE_PATH):
    print("Loading embeddings from cache...")
    embeddings = np.load(CACHE_PATH)
else:
    print("Cache not found. Computing embeddings (first run only)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=4
    )
    np.save(CACHE_PATH, embeddings)
    print(f"Embeddings saved to cache: {CACHE_PATH}")

# --- CONSTRUCT STANCE ---
np.random.seed(42)
direction = np.random.normal(size=embeddings.shape[1])

stance = embeddings @ direction
stance = (stance - stance.mean()) / stance.std()
df["stance"] = stance

# --- DEFINE VARIABLES ---
df["Y_tilde"] = df["stance"]
df["D"] = (df["party"] == "Republican").astype(int)

# --- NAIVE ESTIMATE ---
ate_llm = df[df["D"] == 1]["Y_tilde"].mean() - df[df["D"] == 0]["Y_tilde"].mean()
print(f"\nSystem Check: Torch {torch.__version__} | NumPy {np.__version__}")
print("LLM-based difference (Republican - Democrat):", ate_llm)

# --- SAVE RESULTS ---
df.to_csv("day1/data/llm/speeches_with_stance.csv", index=False)
print("LLM labels saved.")
