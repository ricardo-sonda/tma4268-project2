import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")

# Target
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
y = (df["Winner"] == "Red").astype(int)

# --- Drop columns: odds, target leakage, identifiers ---
drop_patterns = ["Odds", "ExpectedValue",          # odds
                 "Winner", "Finish", "FinishRound", # target leakage
                 "FinishRoundTime", "TotalFightTimeSecs",
                 "Fighter", "Date", "Location", "Country"]

cols_to_drop = [c for c in df.columns
                if any(p.lower() in c.lower() for p in drop_patterns)]

# Also drop rank columns (>70% missing)
rank_cols = [c for c in df.columns if "Rank" in c]
cols_to_drop += rank_cols
cols_to_drop += ["EmptyArena"]  # >22% missing, not a fighter stat

X = df.drop(columns=list(set(cols_to_drop)), errors="ignore")

# Encode categoricals
cat_cols = X.select_dtypes(include="object").columns.tolist()
print(f"Categorical columns to encode: {cat_cols}")
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Class balance: Red={y.mean():.3f}, Blue={1-y.mean():.3f}\n")

# --- XGBoost with 5-fold stratified CV ---
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    enable_categorical=False,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated predictions
y_pred = cross_val_predict(xgb, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba")[:, 1]

# ============================================================
# 1. Metrics
# ============================================================
print("=" * 60)
print("XGBoost 5-Fold CV Results (no odds)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y, y_proba):.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

# ============================================================
# 2. Confusion matrix
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Blue", "Red"],
                                         cmap="Blues", ax=ax)
ax.set_title("XGBoost Confusion Matrix (5-fold CV)")
plt.tight_layout()
plt.savefig("plots/models/xgb_confusion.png", dpi=150)
plt.close()
print("Saved plots/models/xgb_confusion.png")

# ============================================================
# 3. ROC curve
# ============================================================
fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC={roc_auc_score(y, y_proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost (no odds)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/xgb_roc.png", dpi=150)
plt.close()
print("Saved plots/models/xgb_roc.png")

# ============================================================
# 4. Feature importance (top 25)
# ============================================================
# Fit on full data for importance plot
xgb.fit(X, y)
importances = pd.Series(xgb.feature_importances_, index=X.columns)
top25 = importances.nlargest(25)

fig, ax = plt.subplots(figsize=(10, 8))
top25.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("XGBoost Top 25 Features (no odds)")
plt.tight_layout()
plt.savefig("plots/models/xgb_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/xgb_importance.png")

print("\nTop 25 features:")
print(top25.sort_values(ascending=False).to_string())

# Save CV predictions
np.savez("results/xgboost.npz",
         y=y.values, y_pred=y_pred, y_proba=y_proba)
print("Saved results/xgboost.npz")
