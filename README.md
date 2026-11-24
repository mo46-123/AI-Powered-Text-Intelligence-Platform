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
cd AI-Powered-Text-Intelligence-Platform
```
### Install dependencies
```
# (optional) set up virtual environment
conda create -n text-intelligence python=3.14
conda activate text-intelligence
# install deps
pip install -r requirements.txt
```

## Use Classic ML models 
This repo currently support 2 classic baseline models, naive_bayes, and logistic regression 

### Download pre-trained classifier
1. download `naive_bayes_vectorizer.joblib` and `naive_bayes.joblib`, or `logistic_regression_vectorizer.joblib` and `logistic_regressions.joblib` from: https://drive.google.com/drive/folders/1360TgaawgRQdyROQfkwo9Z-N39VzVEpX?usp=sharing
2. create and place them under:

```text
AI-Powered-Text-Intelligence-Platform/
    |-models/
        |-naive_bayes_vectorizer.joblib
        |-naive_bayes.joblib
        |-logistic_regression_vectorizer.joblib
        |-logistic_regression.joblib
```

### Demo 
```
python model/classic/demo.py -m naive-bayes -c "Love the Pill Pockets. We have three dogs. Two of them will let us put the pill in their throat ..." # use naive bayes classifier 
python model/classic/demo.py -m logistic regression -cf data/clean_reviews.csv # use logistic regression classifier 
```

## Use modern models 
This repo currently support 2 modern models, custom transfomer, and fine tuned BERT

### Use Custom Transformer Classifier

#### Download pre-trained classifier
1. download `transformer.pth`, `custom_tokenizer.json`, and `transformer_config.json` from: https://drive.google.com/drive/folders/1lCEpnpn85n-OKTbzCN9p-uAHti6tTAqt?usp=sharing

2. create and place them under:
```text
AI-Powered-Text-Intelligence-Platform/
    |-models/
        |-transformer.pth
        |-custom_tokenizer.json
        |-transformer_config.json
```
#### Demo 
```
python model/modern/custom-transformer/demo.py -c "Love the Pill Pockets. We have three dogs. Two of them will let us put the pill in their throat ..." 
python model/modern/custom-transformer/demo.py -m logistic regression -cf data/clean_reviews.csv # use logistic regression classifier 
```
### Use fine tuned BERT
#### Download tuned model
1. download `fine_tuned/` from: https://drive.google.com/drive/folders/1lCEpnpn85n-OKTbzCN9p-uAHti6tTAqt?usp=sharing google should zip it for you
2. extract the directory under `./models`
```text
AI-Powered-Text-Intelligence-Platform/
    |-models/
        |-fine_tuned/
```
#### Demo
```
python model/modern/fine-tuning/demo.py -c "Love the Pill Pockets. We have three dogs. Two of them will let us put the pill in their throat ..." 
python model/modern/fine-tuning/demo.py -m logistic regression -cf data/clean_reviews.csv # use logistic regression classifier 
```

## Train your own model 

### Getting data
1. pre-prepare data
```
python make_sample.py     # 50,000-review sample (seed 46)
python preprocess.py      # clean text, 3-class labels, remove duplicate texts
python split_data.py      # stratified 70/10/20 train/val/test split
python features.py        # BoW + TF-IDF features and saved vectorizers
```
You should get train 32,418 / val 4,632 / test 9,263. If your sizes match, you have the same split as the team.

2. Make sure you have prepared the data like the following:

```text
AI-Powered-Text-Intelligence-Platform/
    |-data/
        |-test.csv
        |-train.csv
        |-val.csv
```

3. If you want to train modern models 
    - create `CSE 3521/data/` under MyDrive in google drive
    - upload `test.csv`, `train.csv`, and `val.csv` to that directory
```text
MyDrive/
    |-CSE 3521/
        |-data/
            |-test.csv
            |-train.csv
            |-val.csv
```

### Training Classic ML models 
```
python model/classic/naive-bayes/train.py # train naive bayes classifier
python model/classic/logistic-regression/train.py # train logistic regression classifier 
```
### Training model models 
Make sure you have folder `MyDrive/CSE 3521/modern/` on you google drive
```text
MyDrive/
    |-CSE 3521/
        |-modern/
```
### Training custom transformer classifier 
1. Make sure you have the colab plugin on your vscode install
2. train tokenizer 
    - create a folder called `CSE 3521/modern/` under `MyDrive`
    - open `./model/custom-transformer/tokenizer.ipynb`
    - click run all
    - create colab GPU server when prompted   
2. train model  
    - right click on `./model/custom-transformer/custom_dataset.py`, `./model/custom-transformer/transformer.py`, and `./model/utils.py`
    - select upload to colab from menu
    - open `./model/custom-transformer/train.ipynb` 
    - click run all

### Training fine tuned BERT
1. Make sure you have the colab plugin on your vscode install
2. fine tune model  
    - right click on `./model/fine-tuning/fine_tune_dataset.py`, and `./model/utils.py`
    - select upload to colab from menu
    - open `./model/fine-tuning/train.ipynb`
    - click run all

## Run the Streamlit app
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Download the Naive Bayes and Logistic Regression model files (see "Use Classic ML models" above)
3. Start the app:
```
streamlit run app.py
```
4. Open http://localhost:8501, paste a review, and click **Predict** to see both models' predictions side by side.
Live app: not deployed yet.