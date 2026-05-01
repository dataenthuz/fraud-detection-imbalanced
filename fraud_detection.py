import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 284,807 transactions, 492 fraud (0.17%)
df = pd.read_csv('data/creditcard.csv')

print(f"Shape: {df.shape}")
print(f"Fraud: {df['Class'].sum()} cases ({df['Class'].mean()*100:.3f}%)")
print(f"Legit: {(df['Class']==0).sum()} cases\n")

X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

def evaluate(name, y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, target_names=['legit', 'fraud'], output_dict=True)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    results[name] = {
        'precision_fraud': report['fraud']['precision'],
        'recall_fraud': report['fraud']['recall'],
        'f1_fraud': report['fraud']['f1-score'],
        'roc_auc': auc,
        'avg_precision': ap
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=['legit', 'fraud']))
    print(f"ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")

# 1. Baseline - no resampling
print("Training baseline...")
base = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0)
base.fit(X_train, y_train)
evaluate("Baseline XGBoost", y_test, base.predict(X_test), base.predict_proba(X_test)[:,1])

# 2. SMOTE oversampling
print("\nTraining with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - fraud: {y_res.sum()}, legit: {(y_res==0).sum()}")
smote_model = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0)
smote_model.fit(X_res, y_res)
evaluate("SMOTE + XGBoost", y_test, smote_model.predict(X_test), smote_model.predict_proba(X_test)[:,1])

# 3. scale_pos_weight (XGBoost's class weight equivalent)
print("\nTraining with scale_pos_weight...")
ratio = (y_train == 0).sum() / (y_train == 1).sum()
weighted = xgb.XGBClassifier(
    n_estimators=200, scale_pos_weight=ratio,
    random_state=42, eval_metric='logloss', verbosity=0
)
weighted.fit(X_train, y_train)
evaluate("scale_pos_weight XGBoost", y_test, weighted.predict(X_test), weighted.predict_proba(X_test)[:,1])

# 4. Threshold tuning on baseline model
print("\nThreshold tuning...")
probs = base.predict_proba(X_test)[:,1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probs)
f1_vals = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-9)
best_idx = np.argmax(f1_vals[:-1])
best_thresh = thresholds[best_idx]
print(f"Best threshold: {best_thresh:.4f} (default is 0.5)")
preds_tuned = (probs >= best_thresh).astype(int)
evaluate(f"Threshold Tuned (t={best_thresh:.3f})", y_test, preds_tuned, probs)

print("\n\n=== SUMMARY ===")
summary = pd.DataFrame(results).T.round(4)
print(summary.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
for name, model in [("Baseline", base), ("SMOTE", smote_model), ("Weighted", weighted)]:
    p, r, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    ap = average_precision_score(y_test, model.predict_proba(X_test)[:,1])
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves - Fraud Detection")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("precision_recall_curves.png", dpi=150)
print("\nSaved precision_recall_curves.png")
