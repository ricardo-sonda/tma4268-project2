import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Class balance: Red={y.mean():.3f}, Blue={1-y.mean():.3f}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

y_pred = cross_val_predict(rf, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba)

print("=" * 60)
print("Random Forest (500 trees, max_depth=10)")
print("=" * 60)
print(f"Accuracy:  {acc:.4f}")
print(f"ROC AUC:   {auc:.4f}\n")
print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

# Fit on full data for importance
rf.fit(X, y)
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
fig, ax = plt.subplots(figsize=(10, 7))
rf_imp.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest -- Top 20 Features")
plt.tight_layout()
plt.savefig("plots/models/rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/rf_importance.png")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Blue", "Red"],
                                         cmap="Blues", ax=ax)
ax.set_title(f"Random Forest (Acc={acc:.3f})")
plt.tight_layout()
plt.savefig("plots/models/rf_confusion.png", dpi=150)
plt.close()
print("Saved plots/models/rf_confusion.png")

# Save CV predictions
np.savez("results/random_forest.npz",
         y=y.values, y_pred=y_pred, y_proba=y_proba)
print("Saved results/random_forest.npz")
