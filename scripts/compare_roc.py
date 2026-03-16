import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
df["WinnerRed"] = (df["Winner"] == "Red").astype(int)
df["TitleBout"] = df["TitleBout"].astype(int)
df["IsMale"] = (df["Gender"] == "MALE").astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# Define models
# ============================================================
models = {}

# 1. Odds benchmark (2 predictors)
odds_preds = ["RedOdds", "BlueOdds"]
m1 = df[["WinnerRed"] + odds_preds].dropna()
models["Logistic: Odds only (2)"] = (m1[odds_preds], m1["WinnerRed"])

# 2. Full linear model (27 predictors, with odds)
full_preds = [
    "RedOdds", "BlueOdds",
    "RedHeightCms", "BlueHeightCms", "RedReachCms", "BlueReachCms",
    "RedWeightLbs", "RedAge", "BlueAge",
    "RedAvgSigStrLanded", "BlueAvgSigStrLanded",
    "RedAvgSigStrPct", "BlueAvgSigStrPct",
    "RedAvgSubAtt", "BlueAvgSubAtt",
    "RedAvgTDLanded", "BlueAvgTDLanded",
    "RedAvgTDPct", "BlueAvgTDPct",
    "RedWins", "BlueWins", "RedLosses", "BlueLosses",
    "RedCurrentWinStreak", "BlueCurrentWinStreak",
    "NumberOfRounds", "TitleBout",
]
m2 = df[["WinnerRed"] + full_preds].dropna()
models["Logistic: Full + odds (27)"] = (m2[full_preds], m2["WinnerRed"])

# 3. Dif model (no odds, 18 predictors)
dif_preds = [
    "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif",
    "WinDif", "LossDif", "TotalRoundDif", "TotalTitleBoutDif",
    "KODif", "SubDif", "HeightDif", "ReachDif", "AgeDif",
    "SigStrDif", "AvgSubAttDif", "AvgTDDif",
    "TitleBout", "NumberOfRounds", "IsMale",
]
m3 = df[["WinnerRed"] + dif_preds].dropna()
models["Logistic: Dif features (18)"] = (m3[dif_preds], m3["WinnerRed"])

# 4. Full Red/Blue stats, no odds (25 predictors)
stats_preds = [p for p in full_preds if p not in ("RedOdds", "BlueOdds")]
m4 = df[["WinnerRed"] + stats_preds].dropna()
models["Logistic: Stats no odds (25)"] = (m4[stats_preds], m4["WinnerRed"])

# ============================================================
# Fit and evaluate
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

print(f"{'Model':<38} {'N':>6} {'Accuracy':>10} {'AUC':>8}")
print("-" * 66)

results = {}
for (name, (X, y)), color in zip(models.items(), colors):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    y_pred = cross_val_predict(lr, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)

    print(f"{name:<38} {len(y):>6} {acc:>10.4f} {auc:>8.4f}")
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
    results[name] = {"acc": acc, "auc": auc}

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Comparison — All Linear Models (5-fold CV)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/models/roc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/models/roc_comparison.png")
