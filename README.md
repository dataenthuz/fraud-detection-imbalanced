# fraud-detection-imbalanced

Working through the classic problem of detecting fraud when 99.8% of your data is legitimate transactions.

## The problem

Standard accuracy is useless here. A model that predicts "not fraud" every single time gets 99.8% accuracy while catching zero fraud cases. This project focuses on actually useful metrics: precision-recall, F1 on the minority class, and ROC-AUC.

## Dataset

Kaggle's Credit Card Fraud Detection dataset: 284,807 transactions, 492 fraud cases (0.17%).

## Get the data

```bash
# one-time setup: put kaggle.json in ~/.kaggle/ (kaggle.com > Account > Settings > API > Create New Token)
pip install kaggle
python data/download.py
```

This downloads `creditcard.csv` into the `data/` folder. File is ~150MB so it's excluded from git.

## What I'm comparing

- **Baseline** - XGBoost with no resampling, just to see how bad it is
- **SMOTE** - oversample the minority class
- **class_weight** - let the model penalize misclassifying fraud more heavily
- **Threshold tuning** - train normally, then shift the decision boundary

Early result: SMOTE helped recall but killed precision. Threshold tuning on a cost-sensitive model is looking more promising.

## Run it

```bash
pip install -r requirements.txt
python data/download.py       # first time only
python fraud_detection.py     # train all models and compare
```

## Notebook

For the full EDA and step-by-step analysis: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

```bash
jupyter notebook notebooks/eda.ipynb
```

## Files

```
data/
  download.py         # fetches creditcard.csv from Kaggle
  creditcard.csv      # ~150MB, gitignored
notebooks/
  eda.ipynb           # EDA and modeling walkthrough
fraud_detection.py    # main script: trains all models and compares
requirements.txt
```

## Stack

Python - pandas - scikit-learn - XGBoost - imbalanced-learn - matplotlib
