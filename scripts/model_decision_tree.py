import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Class balance: Red={y.mean():.3f}, Blue={1-y.mean():.3f}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=50,
    min_samples_split=100,
    random_state=42,
)

y_pred = cross_val_predict(dt, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(dt, X, y, cv=cv, method="predict_proba")[:, 1]
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba)

print("=" * 60)
print("Decision Tree (max_depth=6, min_leaf=50)")
print("=" * 60)
print(f"Accuracy:  {acc:.4f}")
print(f"ROC AUC:   {auc:.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

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
ax.set_title("Decision Tree -- Top 20 Features")
plt.tight_layout()
plt.savefig("plots/models/tree_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/tree_importance.png")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Blue", "Red"],
                                         cmap="Greens", ax=ax)
ax.set_title(f"Decision Tree (Acc={acc:.3f})")
plt.tight_layout()
plt.savefig("plots/models/tree_confusion.png", dpi=150)
plt.close()
print("Saved plots/models/tree_confusion.png")

# Save CV predictions
np.savez("results/decision_tree.npz",
         y=y.values, y_pred=y_pred, y_proba=y_proba)
print("Saved results/decision_tree.npz")
