import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as smf

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
df["WinnerRed"] = (df["Winner"] == "Red").astype(int)
df["TitleBout"] = df["TitleBout"].astype(int)

predictors = [
    "RedHeightCms", "BlueHeightCms", "RedReachCms", "BlueReachCms",
    "RedWeightLbs", "RedAge", "BlueAge",
    "RedAvgSigStrLanded", "BlueAvgSigStrLanded",
    "RedAvgSigStrPct", "BlueAvgSigStrPct",
    "RedAvgSubAtt", "BlueAvgSubAtt",
    "RedAvgTDLanded", "BlueAvgTDLanded",
    "RedAvgTDPct", "BlueAvgTDPct",
    "RedWins", "BlueWins", "RedLosses", "BlueLosses",
    "RedCurrentWinStreak", "BlueCurrentWinStreak",
    "RedTotalRoundsFought", "BlueTotalRoundsFought",
    "RedTotalTitleBouts", "BlueTotalTitleBouts",
    "NumberOfRounds", "TitleBout", "WeightClass",
    "RedWinsByKO", "BlueWinsByKO",
]

model_df = df[["WinnerRed"] + predictors].dropna()
y = model_df["WinnerRed"]
X = model_df[predictors].copy()

# Encode WeightClass for sklearn (statsmodels formula handles it via C())
if "WeightClass" in X.columns:
    X["WeightClass"] = LabelEncoder().fit_transform(X["WeightClass"])

print(f"Rows: {len(model_df)} / {len(df)}")
print(f"Predictors: {len(predictors)}")
print(f"Class balance: Red={y.mean():.3f}, Blue={1-y.mean():.3f}\n")

# ============================================================
# 1. Statsmodels logit for coefficients and summary
# ============================================================
formula = "WinnerRed ~ " + " + ".join(predictors)
logit = smf.logit(formula, data=model_df).fit(disp=0)
print("=" * 65)
print(f"Logistic Regression — Stats no odds ({len(predictors)} predictors)")
print("=" * 65)
print(logit.summary())

# ============================================================
# 2. Sklearn logistic with 5-fold CV
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000, random_state=42))

y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba)

print(f"\n5-fold CV: Accuracy={acc:.4f}  AUC={auc:.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

# ============================================================
# 3. Confusion matrix
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Blue", "Red"],
                                         cmap="Blues", ax=ax)
ax.set_title(f"Logistic no odds -- Confusion Matrix (Acc={acc:.3f})")
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_confusion.png", dpi=151)
plt.close()
print("Saved plots/models/logit_no_odds_confusion.png")

# ============================================================
# 4. ROC curve
# ============================================================
fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"Logistic no odds (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC -- Logistic Regression (no odds, {len(predictors)} predictors)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_roc.png", dpi=150)
plt.close()
print("Saved plots/models/logit_no_odds_roc.png")

# ============================================================
# 5. Save CV predictions for compare_roc.py
# ============================================================
np.savez("results/logit_no_odds.npz",
         y=y.values, y_pred=y_pred, y_proba=y_proba)
print("Saved results/logit_no_odds.npz")
