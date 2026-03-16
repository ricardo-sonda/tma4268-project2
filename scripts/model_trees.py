import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             classification_report, ConfusionMatrixDisplay)

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
y = (df["Winner"] == "Red").astype(int)

# --- Drop: odds, target leakage, identifiers, ranks (>70% missing) ---
drop_patterns = ["Odds", "ExpectedValue",
                 "Winner", "Finish", "FinishRound",
                 "FinishRoundTime", "TotalFightTimeSecs",
                 "Fighter", "Date", "Location", "Country"]
rank_cols = [c for c in df.columns if "Rank" in c]
cols_to_drop = [c for c in df.columns
                if any(p.lower() in c.lower() for p in drop_patterns)]
cols_to_drop += rank_cols + ["EmptyArena"]

X = df.drop(columns=list(set(cols_to_drop)), errors="ignore")

# Encode categoricals
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Class balance: Red={y.mean():.3f}, Blue={1-y.mean():.3f}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 1. Single Decision Tree (pruned)
# ============================================================
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=50,
    min_samples_split=100,
    random_state=42,
)

dt_pred = cross_val_predict(dt, X, y, cv=cv, method="predict")
dt_proba = cross_val_predict(dt, X, y, cv=cv, method="predict_proba")[:, 1]
dt_acc = accuracy_score(y, dt_pred)
dt_auc = roc_auc_score(y, dt_proba)

print("=" * 60)
print("Decision Tree (max_depth=6, min_leaf=50)")
print("=" * 60)
print(f"Accuracy:  {dt_acc:.4f}")
print(f"ROC AUC:   {dt_auc:.4f}\n")
print(classification_report(y, dt_pred, target_names=["Blue", "Red"]))

# Fit on full data for visualization
dt.fit(X, y)
fig, ax = plt.subplots(figsize=(28, 12))
plot_tree(dt, ax=ax, feature_names=X.columns, class_names=["Blue", "Red"],
          filled=True, rounded=True, fontsize=6, max_depth=4)
ax.set_title("Decision Tree (showing top 4 levels)")
plt.tight_layout()
plt.savefig("plots/models/tree_structure.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/tree_structure.png")

# Feature importance
dt_imp = pd.Series(dt.feature_importances_, index=X.columns).nlargest(20)
fig, ax = plt.subplots(figsize=(10, 7))
dt_imp.sort_values().plot.barh(ax=ax, color="forestgreen")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Decision Tree — Top 20 Features")
plt.tight_layout()
plt.savefig("plots/models/tree_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/tree_importance.png")

# ============================================================
# 2. Random Forest
# ============================================================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

rf_pred = cross_val_predict(rf, X, y, cv=cv, method="predict")
rf_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
rf_acc = accuracy_score(y, rf_pred)
rf_auc = roc_auc_score(y, rf_proba)

print("\n" + "=" * 60)
print("Random Forest (500 trees, max_depth=10)")
print("=" * 60)
print(f"Accuracy:  {rf_acc:.4f}")
print(f"ROC AUC:   {rf_auc:.4f}\n")
print(classification_report(y, rf_pred, target_names=["Blue", "Red"]))

# Fit on full data for importance
rf.fit(X, y)
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
fig, ax = plt.subplots(figsize=(10, 7))
rf_imp.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest — Top 20 Features")
plt.tight_layout()
plt.savefig("plots/models/rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/rf_importance.png")

# ============================================================
# 3. Confusion matrices side by side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y, dt_pred, display_labels=["Blue", "Red"],
                                         cmap="Greens", ax=axes[0])
axes[0].set_title(f"Decision Tree (Acc={dt_acc:.3f})")
ConfusionMatrixDisplay.from_predictions(y, rf_pred, display_labels=["Blue", "Red"],
                                         cmap="Blues", ax=axes[1])
axes[1].set_title(f"Random Forest (Acc={rf_acc:.3f})")
plt.tight_layout()
plt.savefig("plots/models/trees_confusion.png", dpi=150)
plt.close()
print("Saved plots/models/trees_confusion.png")

# ============================================================
# 4. ROC comparison (all models so far)
# ============================================================
dt_fpr, dt_tpr, _ = roc_curve(y, dt_proba)
rf_fpr, rf_tpr, _ = roc_curve(y, rf_proba)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(dt_fpr, dt_tpr, lw=2, color="forestgreen",
        label=f"Decision Tree (AUC={dt_auc:.3f})")
ax.plot(rf_fpr, rf_tpr, lw=2, color="steelblue",
        label=f"Random Forest (AUC={rf_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
# Reference lines from previous models
ax.axhline(y=0, color="white")  # dummy for legend spacing
ax.plot([], [], ' ', label="--- Benchmarks ---")
ax.plot([], [], 's', color="#e41a1c", label="Logistic Odds only: AUC=0.711")
ax.plot([], [], 's', color="#984ea3", label="XGBoost no odds: AUC=0.617")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Tree Models vs Benchmarks (no odds, 5-fold CV)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/models/trees_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/trees_roc.png")

# ============================================================
# 5. Full comparison table
# ============================================================
print("\n" + "=" * 60)
print("Full Model Comparison (5-fold CV)")
print("=" * 60)
print(f"{'Model':<38} {'Accuracy':>10} {'AUC':>8}")
print("-" * 58)
print(f"{'Naive (always Red)':<38} {'0.5804':>10} {'0.500':>8}")
print(f"{'Logistic: Dif features (18)':<38} {'0.6000':>10} {'0.596':>8}")
print(f"{'Decision Tree no odds (67)':<38} {dt_acc:>10.4f} {dt_auc:>8.4f}")
print(f"{'Logistic: Stats no odds (25)':<38} {'0.6167':>10} {'0.637':>8}")
print(f"{'XGBoost no odds (67)':<38} {'0.6017':>10} {'0.617':>8}")
print(f"{'Random Forest no odds (67)':<38} {rf_acc:>10.4f} {rf_auc:>8.4f}")
print(f"{'Logistic: Odds only (2)':<38} {'0.6599':>10} {'0.711':>8}")
