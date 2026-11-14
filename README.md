# AI-Powered Text Intelligence Platform

## Team Members

Project Lead: Michael Xu \
Data Engineer: Mohamed Yattara \
ML Engineer: Shao Zhu \
App Engineer: Jiajie Chen \
Analyst / Writer: Jackson


## Description

An app that uses AI to predict whether a review is positive, negative, or neutral.
## Getting the data (not committed to Git)

Group sample uses random seed 46.

1. Download `Reviews.csv` from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews into `data/`.
2. From the repo root, run:

```
pip install pandas scikit-learn matplotlib
python make_sample.py     # 50,000-review sample (seed 46)
python preprocess.py      # clean text, 3-class labels, remove duplicate texts
python split_data.py      # stratified 70/10/20 train/val/test split
python features.py        # BoW + TF-IDF features and saved vectorizers
```

You should get train 32,418 / val 4,632 / test 9,263. If your sizes match, you have the same split as the team.
