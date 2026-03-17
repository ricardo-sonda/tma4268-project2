"""Logistic regression on UFC fight outcomes excluding all odds/probability features."""

import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
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

# ── Drop odds/probability, identifiers, and target columns ──
drop_keywords = ["odds", "prob"]
drop_cols = [
    c for c in df.columns
    if any(kw in c.lower() for kw in drop_keywords)
]
non_feature_cols = [
    "RedFighter", "BlueFighter", "Date", "Location", "Country",
    "Winner", "WinnerRed",
]
drop_cols = list(set(drop_cols + non_feature_cols))

feature_df = df.drop(columns=drop_cols)

# ── Encode categoricals ─────────────────────────────────────
cat_cols = feature_df.select_dtypes(include="object").columns.tolist()
feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

# ── Drop rows with missing values ───────────────────────────
model_df = pd.concat([df["WinnerRed"], feature_df], axis=1).dropna()
y = model_df["WinnerRed"]
X = model_df.drop(columns=["WinnerRed"])

print(f"Rows: {len(model_df)} / {len(df)}  |  Features: {X.shape[1]}")

# ── MinMax scaling ───────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ── 5-Fold CV ───────────────────────────────────────────────
lr = LogisticRegression(max_iter=2000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict")
y_proba = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

print("=" * 60)
print("Logistic Regression 5-Fold CV (no odds, MinMax scaled)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y, y_proba):.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

# Fit on full data for coefficients
lr.fit(X_scaled, y)

# ── 1. Confusion matrix ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred, display_labels=["Blue", "Red"], cmap="Blues", ax=ax,
)
ax.set_title("Logistic (no odds) — Confusion Matrix (5-fold CV)")
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_confusion.png", dpi=150)
plt.close()
print("Saved plots/models/logit_no_odds_confusion.png")

# ── 2. ROC curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"Logistic no odds (AUC={roc_auc_score(y, y_proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Logistic Regression (no odds)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_roc.png", dpi=150)
plt.close()
print("Saved plots/models/logit_no_odds_roc.png")

# ── 3. Standardised coefficients (top 20) ───────────────────
coef_series = pd.Series(lr.coef_[0], index=X.columns)
top = coef_series.abs().nlargest(20).index
top_coefs = coef_series[top].sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
top_coefs.plot.barh(ax=ax, color=["#d9534f" if v < 0 else "#5cb85c" for v in top_coefs])
ax.set_xlabel("Coefficient (MinMax-scaled features)")
ax.set_title("Top 20 Logistic Regression Coefficients (no odds)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_coefs.png", dpi=150)
plt.close()
print("Saved plots/models/logit_no_odds_coefs.png")

# ── 4. Binned residual plot ──────────────────────────────────
order = np.argsort(y_proba)
y_sorted = y.values[order]
p_sorted = y_proba[order]

n_bins = 40
bin_size = len(y_sorted) // n_bins
bin_mid, bin_resid, bin_se = [], [], []
for i in range(n_bins):
    s, e = i * bin_size, (i + 1) * bin_size
    avg_p = p_sorted[s:e].mean()
    avg_r = (y_sorted[s:e] - p_sorted[s:e]).mean()
    se = np.sqrt(avg_p * (1 - avg_p) / (e - s))
    bin_mid.append(avg_p)
    bin_resid.append(avg_r)
    bin_se.append(se)

bin_mid = np.array(bin_mid)
bin_resid = np.array(bin_resid)
bin_se = np.array(bin_se)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(bin_mid, bin_resid, s=30, zorder=3)
ax.plot(bin_mid, 2 * bin_se, "r--", lw=1, label="±2 SE")
ax.plot(bin_mid, -2 * bin_se, "r--", lw=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Average residual")
ax.set_title("Binned Residual Plot — Logistic (no odds)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_binned_resid.png", dpi=150)
plt.close()
print("Saved plots/models/logit_no_odds_binned_resid.png")

# ── Save CV predictions ──────────────────────────────────────
np.savez("results/logit_no_odds.npz", y=y.values, y_pred=y_pred, y_proba=y_proba)
print("Saved results/logit_no_odds.npz")
