import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/clean_reviews.csv")

# step 1: lock away 20% as the test set
trainval, test = train_test_split(
    df, test_size=0.2, stratify=df["Sentiment"], random_state=46
)

# step 2: 12.5% of the remaining 80% = 10% of the total
train, val = train_test_split(
    trainval, test_size=0.125, stratify=trainval["Sentiment"], random_state=46
)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

for name, part in [("Train", train), ("Val", val), ("Test", test)]:
    print(name, len(part))
    print((part["Sentiment"].value_counts(normalize=True) * 100).round(2))