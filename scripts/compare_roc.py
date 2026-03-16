"""
Compare ROC curves across all models.
Run the individual model scripts first to generate results/*.npz files.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from pathlib import Path

# ============================================================
# Load saved CV predictions from each model
# ============================================================
models = {
    "Logistic: Odds only (2)":       "results/logit_odds.npz",
    "Logistic: Stats no odds (32)":  "results/logit_no_odds.npz",
    "Logistic: Dif features (18)":   "results/logit_dif.npz",
    "Decision Tree (67)":            "results/decision_tree.npz",
    "Random Forest (67)":            "results/random_forest.npz",
    "XGBoost (67)":                  "results/xgboost.npz",
}

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

fig, ax = plt.subplots(figsize=(9, 7))

print(f"{'Model':<38} {'N':>6} {'Accuracy':>10} {'AUC':>8}")
print("-" * 66)

for (name, path), color in zip(models.items(), colors):
    if not Path(path).exists():
        print(f"{name:<38} {'--':>6} {'MISSING':>10} {'--':>8}")
        continue

    data = np.load(path)
    y, y_pred, y_proba = data["y"], data["y_pred"], data["y_proba"]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)

    print(f"{name:<38} {len(y):>6} {acc:>10.4f} {auc:>8.4f}")
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Comparison -- All Models (5-fold CV)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/models/roc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/models/roc_comparison.png")
