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

# ============================================================
# 6. Feature importance analysis
# ============================================================
from sklearn.inspection import permutation_importance

print("\n" + "=" * 65)
print("Feature Importance Analysis")
print("=" * 65)

# --- 6a. Statsmodels: Wald z-statistics and p-values ---
sm_results = pd.DataFrame({
    "coef": logit.params,
    "z": logit.tvalues,
    "p-value": logit.pvalues,
    "abs_z": logit.tvalues.abs(),
}).drop("Intercept", errors="ignore").sort_values("abs_z", ascending=False)

print("\nWald z-statistics (statsmodels logit):")
print(sm_results.to_string())

fig, ax = plt.subplots(figsize=(10, 9))
plot_data = sm_results.sort_values("abs_z")
colors = ["firebrick" if p < 0.05 else "grey" for p in plot_data["p-value"]]
ax.barh(plot_data.index, plot_data["abs_z"], color=colors)
ax.axvline(1.96, color="black", linestyle="--", linewidth=0.8, label="z=1.96 (p=0.05)")
ax.set_xlabel("|Wald z-statistic|")
ax.set_title("Feature Importance -- Wald z (red = p < 0.05)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_wald.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/logit_no_odds_wald.png")

# --- 6b. Standardized coefficients (sklearn) ---
pipe.fit(X, y)
lr_model = pipe.named_steps["logisticregression"]
scaler = pipe.named_steps["standardscaler"]

std_coefs = pd.Series(lr_model.coef_[0], index=X.columns)
std_coefs_abs = std_coefs.abs().sort_values(ascending=False)

print("\nStandardized coefficients (sklearn, scaled features):")
print(pd.DataFrame({
    "std_coef": std_coefs[std_coefs_abs.index],
    "abs_std_coef": std_coefs_abs
}).to_string())

fig, ax = plt.subplots(figsize=(10, 9))
plot_std = std_coefs[std_coefs_abs.index].sort_values()
colors = np.where(plot_std > 0, "firebrick", "steelblue")
ax.barh(plot_std.index, plot_std.values, color=colors)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Standardized Coefficient")
ax.set_title("Feature Importance -- Standardized Logistic Coefs (red=+, blue=-)")
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_std_coefs.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/logit_no_odds_std_coefs.png")

# --- 6c. Permutation importance (on full fitted model) ---
perm = permutation_importance(pipe, X, y, n_repeats=20, random_state=42,
                              scoring="roc_auc", n_jobs=-1)
perm_df = pd.DataFrame({
    "mean": perm.importances_mean,
    "std": perm.importances_std,
}, index=X.columns).sort_values("mean", ascending=False)

print("\nPermutation importance (AUC drop, 20 repeats):")
print(perm_df.to_string())

fig, ax = plt.subplots(figsize=(10, 9))
plot_perm = perm_df.sort_values("mean")
colors = ["firebrick" if m > 0.001 else "grey" for m in plot_perm["mean"]]
ax.barh(plot_perm.index, plot_perm["mean"], xerr=plot_perm["std"], color=colors)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Mean AUC decrease")
ax.set_title("Permutation Importance -- Logistic no odds")
plt.tight_layout()
plt.savefig("plots/models/logit_no_odds_perm_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/logit_no_odds_perm_importance.png")
