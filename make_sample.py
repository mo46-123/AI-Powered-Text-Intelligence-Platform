import pandas as pd

seed = 46
df = pd.read_csv("data/Reviews.csv")   # raw Kaggle file, downloaded locally
sample = df.sample(n=50000, random_state=seed)
sample.to_csv("data/group_sample_seed46.csv", index=False)
print(f"Saved {len(sample)} rows")