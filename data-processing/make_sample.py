from pathlib import Path
import pandas as pd
import kagglehub

repo_root = Path(__file__).resolve().parents[1]

# Download raw dataset
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews", output_dir=str(repo_root / 'data'))
print("Path to dataset files:", path)

# seed raw dataset 
seed = 46
df = pd.read_csv("data/Reviews.csv")   # raw Kaggle file, downloaded locally
sample = df.sample(n=50000, random_state=seed)
sample.to_csv("data/group_sample_seed46.csv", index=False)
print(f"Saved {len(sample)} rows")