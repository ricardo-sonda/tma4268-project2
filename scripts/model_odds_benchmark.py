import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
df["WinnerRed"] = (df["Winner"] == "Red").astype(int)

odds_cols = ["RedOdds", "BlueOdds"]
model_df = df[["WinnerRed"] + odds_cols].dropna()
print(f"Rows: {len(model_df)} / {len(df)}")

y = model_df["WinnerRed"]
X = model_df[odds_cols]

# ============================================================
# 1. OLS Linear Probability Model
# ============================================================
ols = smf.ols("WinnerRed ~ RedOdds + BlueOdds", data=model_df).fit()
print("=" * 60)
print("OLS Linear Probability Model (odds only)")
print("=" * 60)
print(ols.summary())

print("\n" + "=" * 60)
print("ANOVA (Type II)")
print("=" * 60)
anova_table = anova_lm(ols, typ=2)
anova_table["pct_SS"] = (anova_table["sum_sq"] / anova_table["sum_sq"].sum() * 100).round(2)
print(anova_table.sort_values("sum_sq", ascending=False).to_string())

# ============================================================
# 2. Logistic Regression (proper classifier) with 5-fold CV
# ============================================================
lr = LogisticRegression(max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(lr, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]

print("\n" + "=" * 60)
print("Logistic Regression 5-Fold CV (odds only)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y, y_proba):.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

# Fit on full data for coefficients
lr.fit(X, y)
print(f"Logistic coefs: RedOdds={lr.coef_[0][0]:.6f}, BlueOdds={lr.coef_[0][1]:.6f}")
print(f"Intercept: {lr.intercept_[0]:.6f}")

# ============================================================
# 3. Confusion matrix
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Blue", "Red"],
                                         cmap="Blues", ax=ax)
ax.set_title("Odds Benchmark — Confusion Matrix (5-fold CV)")
plt.tight_layout()
plt.savefig("plots/models/benchmark_confusion.png", dpi=150)
plt.close()
print("Saved plots/models/benchmark_confusion.png")

# ============================================================
# 4. ROC curve
# ============================================================
fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"Odds benchmark (AUC={roc_auc_score(y, y_proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Odds Benchmark")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/benchmark_roc.png", dpi=150)
plt.close()
print("Saved plots/models/benchmark_roc.png")

# ============================================================
# 5. Model comparison table
# ============================================================
print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)
print(f"{'Model':<35} {'Accuracy':>10} {'AUC':>8} {'Predictors':>12}")
print("-" * 67)
print(f"{'Naive (always Red)':<35} {'0.5804':>10} {'0.500':>8} {'0':>12}")
print(f"{'OLS Dif features (no odds)':<35} {'—':>10} {'—':>8} {'18':>12}")
print(f"  R²=0.036")
print(f"{'XGBoost (no odds)':<35} {'0.6017':>10} {'0.617':>8} {'67':>12}")
print(f"{'Logistic Odds benchmark':<35} {accuracy_score(y, y_pred):>10.4f} {roc_auc_score(y, y_proba):>8.3f} {'2':>12}")
