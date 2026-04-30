# fraud-detection-imbalanced

Working through the classic problem of detecting fraud when 99.8% of your data is legitimate transactions.

## The problem

Standard accuracy is useless here. A model that predicts "not fraud" every single time gets 99.8% accuracy while catching zero fraud cases. This project focuses on actually useful metrics: precision-recall, F1 on the minority class, and ROC-AUC.

## Dataset

Kaggle's Credit Card Fraud Detection dataset: 284,807 transactions, 492 fraud cases (0.17%).

## What I'm comparing

- **Baseline** - XGBoost with no resampling, just to see how bad it is
- **SMOTE** - oversample the minority class
- **class_weight='balanced'** - let the model penalize misclassifying fraud more
- **Threshold tuning** - train normally, then shift the decision boundary

Early result: SMOTE helped recall but killed precision. Threshold tuning on a cost-sensitive model is looking more promising.

## Files

```
notebooks/
  01_eda.ipynb          # data exploration, class imbalance viz
  02_baseline.ipynb     # plain XGBoost
  03_smote.ipynb        # SMOTE + XGBoost
  04_threshold.ipynb    # threshold optimization
src/
  preprocess.py
  evaluate.py           # precision-recall curves, confusion matrix
requirements.txt
```

## Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · matplotlib
