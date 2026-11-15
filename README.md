# AI-Powered Text Intelligence Platform

## Team Members

Project Lead: Michael Xu \
Data Engineer: Mohamed Yattara \
ML Engineer: Shao Zhu \
App Engineer: Jiajie Chen \
Analyst / Writer: Jackson


## Description

An app that uses AI to predict whether a review is positive, negative, or neutral.

## Quick start 
### Setup
```
git clone https://github.com/mo46-123/AI-Powered-Text-Intelligence-Platform.git
cd AI-Powered-Text_Intelligence-Platform
```
### Install dependencies
```
# (optional) get up virtual environment
conda create -n text-intelligence python=3.14
conda activate text-intelligence
# install deps
pip install -r requirements.txt
```
### Download pre-trained classifier 
1. downlaod `naive_bayes_vectorizer.joblib` and `naive_bayes.joblib` from: https://drive.google.com/drive/folders/1360TgaawgRQdyROQfkwo9Z-N39VzVEpX?usp=sharing
2. create and place them under:
AI-Powered-Text-Intelligence-Plateform/
    |
    |-models/
        |-naive_bayes_vectorizer.joblib
        |-naive_bayes.joblib

### demo 
```
python model/naive-bayes/demo.py -c "Love the Pill Pockets. We have three dogs. Two of them will let us put the pill in their throat ..."
python model/naive-bayes/demo.py -cf data/clean_reviews.csv
```

### Getting data

```
python make_sample.py     # 50,000-review sample (seed 46)
python preprocess.py      # clean text, 3-class labels, remove duplicate texts
python split_data.py      # stratified 70/10/20 train/val/test split
python features.py        # BoW + TF-IDF features and saved vectorizers
```

You should get train 32,418 / val 4,632 / test 9,263. If your sizes match, you have the same split as the team.

### training 
Make sure you have prepared the data like the following
AI-Powered-Text-Intelligence-Plateform/
    |
    |-data/
        |
        | - test.csv
        | - train.csv
        | - val.csv
then run
```
python model//naive-bayes/train.py
```
