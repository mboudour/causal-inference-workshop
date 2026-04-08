import pandas as pd
import pyreadr

RAW_PATH = "day1_nz/data/raw/Corp_NZHoR_V2.rds"
OUT_PATH = "day1_nz/data/llm/speeches_sample.csv"

PARTY_TREATED = "National"
PARTY_CONTROL = "Labour"
MIN_CHARS = 100
SAMPLE_N = 8000
RANDOM_STATE = 42

result = pyreadr.read_r(RAW_PATH)
df = next(iter(result.values())).copy()

df = df[df["chair"] != True].copy()
df = df[df["party"].notna()].copy()
df = df[df["text"].notna()].copy()
df = df[df["speaker"].notna()].copy()
df["text"] = df["text"].astype(str)
df = df[df["text"].str.len() > MIN_CHARS].copy()

df = df[df["party"].isin([PARTY_TREATED, PARTY_CONTROL])].copy()

df["party"] = df["party"].map({
    PARTY_TREATED: "Republican",
    PARTY_CONTROL: "Democrat"
})

df["date"] = df["date"].astype(str)
df["year"] = df["date"].str[:4]
df["speech_id"] = "NZ_" + df["date"] + "_" + df["speechnumber"].astype(str)

out = df[["speech_id", "text", "speaker", "party", "date", "year"]].copy()

if len(out) > SAMPLE_N:
    out = out.sample(n=SAMPLE_N, random_state=RANDOM_STATE)

out.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH} with {len(out)} rows")
print(out["party"].value_counts(dropna=False))