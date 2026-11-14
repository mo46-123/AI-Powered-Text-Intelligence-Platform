import html
import re
import pandas as pd

RAW_PATH = "data/group_sample_seed46.csv"
OUT_PATH = "data/clean_reviews.csv"

def clean_text(s):
    s = html.unescape(str(s))             # turn &quot; etc. into real characters
    s = re.sub(r"<[^>]+>", " ", s)        # remove HTML tags like <br />
    s = re.sub(r"\s+", " ", s).strip()    # collapse extra whitespace
    return s

def to_sentiment(score):
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"

df = pd.read_csv(RAW_PATH)
print("Raw rows:", len(df))

df["Summary"] = df["Summary"].fillna("")
df["Text"] = df["Text"].apply(clean_text)
df["Summary"] = df["Summary"].apply(clean_text)

df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)
print("After dedup:", len(df))

df["Sentiment"] = df["Score"].apply(to_sentiment)
df["input_text"] = (df["Summary"] + ". " + df["Text"]).str.strip()

df[["Id", "Score", "Sentiment", "input_text"]].to_csv(OUT_PATH, index=False)
print(df["Sentiment"].value_counts(normalize=True).round(4) * 100)