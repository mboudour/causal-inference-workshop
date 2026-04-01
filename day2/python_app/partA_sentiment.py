"""
Day 2 - Part A: Measurement with LLMs (Sentiment Score)
This script simulates an LLM assigning a sentiment score to each speech.
To ensure reproducibility without calling an actual LLM API during the seminar,
we use a local SentenceTransformer model and a random projection to generate
a continuous score, which we standardize.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Stability settings for Intel Macs
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    print("--- Day 2 Part A: LLM Measurement (Sentiment) ---")

    data_path  = 'day1/data/speeches_sample.csv'
    out_path   = 'day2/data/speeches_with_sentiment.csv'
    cache_path = 'day2/data/embeddings_cache.npy'

    # Ensure output directory exists
    os.makedirs('day2/data', exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} speeches.")

    # We use a fixed seed for the random projection so the "sentiment" score
    # is deterministic and reproducible across runs. We use a different seed
    # than Day 1 (which used 42) so sentiment != stance.
    rng = np.random.default_rng(2026)

    # Load or compute embeddings
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        embeddings = np.load(cache_path)
    else:
        print("Computing embeddings (this may take a minute)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
        np.save(cache_path, embeddings)
        print("Embeddings cached.")

    # Simulate an LLM sentiment score via random projection
    # In reality, this would be an LLM API call: `sentiment = get_llm_sentiment(text)`
    print("Simulating LLM sentiment extraction...")
    projection_vector = rng.standard_normal(embeddings.shape[1])
    raw_sentiment = embeddings @ projection_vector

    # Standardize the score to have mean 0, std 1
    sentiment_score = (raw_sentiment - raw_sentiment.mean()) / raw_sentiment.std()

    df['sentiment_score'] = sentiment_score

    # For Day 2, sentiment_score is our target variable Y_tilde
    # Treatment D is party (Republican = 1, Democrat = 0)
    df['Y_tilde'] = df['sentiment_score']
    df['D'] = (df['party'] == 'Republican').astype(int)

    print("\nSample of computed sentiment scores:")
    print(df[['speaker', 'party', 'Y_tilde']].head())

    # Naive difference in means on the LLM measurement
    mean_R = df[df['D'] == 1]['Y_tilde'].mean()
    mean_D = df[df['D'] == 0]['Y_tilde'].mean()
    naive_diff = mean_R - mean_D

    print(f"\nNaive Difference-in-Means (R - D) on Sentiment: {naive_diff:.4f}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset with sentiment scores to {out_path}")

if __name__ == "__main__":
    main()
