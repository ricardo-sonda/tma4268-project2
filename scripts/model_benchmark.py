"""Benchmark model which predicts "winner = red" if RedImpliedProb >= 0.5."""

import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

# ── Load data ────────────────────────────────────────────────
conn = sqlite3.connect("sql/database.db")
df = pd.read_sql("SELECT * FROM ultimate_ufc__ufc_clean", conn)
conn.close()

df = df[df["Winner"].isin(["Red", "Blue"])].copy()
df["WinnerRed"] = (df["Winner"] == "Red").astype(int)

# select only betting odds
y = df["WinnerRed"]
X = pd.DataFrame(df["RedImpliedProb"])

# model
lr = LogisticRegression(max_iter=2000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(lr, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]

print("=" * 60)
print("Logistic Regression 5-Fold CV")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y, y_proba):.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

# Fit on full data for coefficients
lr.fit(X, y)

# ── 2. ROC curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"Benchmark Logistic (AUC={roc_auc_score(y, y_proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Benchmark Logistic")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/logit_benchmark_roc.png", dpi=150)
plt.close()
print("Saved plots/models/logit_benchmark_roc.png")

# ── Save CV predictions ──────────────────────────────────────
np.savez("results/logit_benchmark.npz", y=y.values, y_pred=y_pred, y_proba=y_proba)
print("Saved results/logit_benchmark.npz")
