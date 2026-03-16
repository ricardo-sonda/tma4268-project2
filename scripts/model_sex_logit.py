import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             classification_report)

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
    "NumberOfRounds", "TitleBout",
]

formula = "WinnerRed ~ " + " + ".join(predictors)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def fit_and_report(subset_df, label):
    """Fit statsmodels logit + sklearn CV, print results, return ROC data."""
    mdf = subset_df[["WinnerRed"] + predictors].dropna()
    y = mdf["WinnerRed"]
    X = mdf[predictors]
    print(f"\n{'='*65}")
    print(f"{label} (n={len(mdf)}, Red%={y.mean():.3f})")
    print(f"{'='*65}")

    # Statsmodels logit for coefficients
    logit = smf.logit(formula, data=mdf).fit(disp=0)
    print(logit.summary())

    # Deviance residuals
    y_hat = logit.predict()
    y_arr = y.values
    sign = np.where(y_arr == 1, 1, -1)
    dev_resid = sign * np.sqrt(
        -2 * (y_arr * np.log(np.clip(y_hat, 1e-10, 1)) +
              (1 - y_arr) * np.log(np.clip(1 - y_hat, 1e-10, 1))))

    # Sklearn CV for accuracy/AUC
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=42))
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    print(f"\n5-fold CV: Accuracy={acc:.4f}  AUC={auc:.4f}")
    print(classification_report(y, y_pred, target_names=["Blue", "Red"]))

    fpr, tpr, _ = roc_curve(y, y_proba)
    return fpr, tpr, auc, y_hat, dev_resid, y_arr


# ============================================================
# 1 & 2. Separate models by sex
# ============================================================
male_df = df[df["Gender"] == "MALE"]
female_df = df[df["Gender"] == "FEMALE"]

m_fpr, m_tpr, m_auc, m_yhat, m_resid, m_y = fit_and_report(male_df, "MALE")
f_fpr, f_tpr, f_auc, f_yhat, f_resid, f_y = fit_and_report(female_df, "FEMALE")

# ============================================================
# 3. Mixed-effects logistic regression (random intercept for Gender)
# ============================================================
print(f"\n{'='*65}")
print("Mixed-Effects Logistic Regression (random intercept: Gender)")
print(f"{'='*65}")

all_df = df[["WinnerRed", "Gender"] + predictors].dropna()
print(f"n={len(all_df)}")

mixed = smf.mixedlm(
    formula,
    data=all_df,
    groups=all_df["Gender"],
).fit(reml=False, method="lbfgs")
print(mixed.summary())

# For mixed model CV: use Gender as a fixed effect instead
# (sklearn can't do GLMM, so we approximate with Gender as fixed effect)
all_df["IsMale"] = (all_df["Gender"] == "MALE").astype(int)
mixed_preds = predictors + ["IsMale"]
X_mixed = all_df[mixed_preds]
y_mixed = all_df["WinnerRed"]

pipe_mixed = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, random_state=42))
mx_pred = cross_val_predict(pipe_mixed, X_mixed, y_mixed, cv=cv, method="predict")
mx_proba = cross_val_predict(pipe_mixed, X_mixed, y_mixed, cv=cv, method="predict_proba")[:, 1]
mx_acc = accuracy_score(y_mixed, mx_pred)
mx_auc = roc_auc_score(y_mixed, mx_proba)
print(f"\n5-fold CV (fixed Gender proxy): Accuracy={mx_acc:.4f}  AUC={mx_auc:.4f}")
print(classification_report(y_mixed, mx_pred, target_names=["Blue", "Red"]))

mx_fpr, mx_tpr, _ = roc_curve(y_mixed, mx_proba)

# ============================================================
# 4. ROC comparison plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(m_fpr, m_tpr, lw=2, color="#377eb8", label=f"Male only (AUC={m_auc:.3f}, n={len(m_y)})")
ax.plot(f_fpr, f_tpr, lw=2, color="#e41a1c", label=f"Female only (AUC={f_auc:.3f}, n={len(f_y)})")
ax.plot(mx_fpr, mx_tpr, lw=2, color="#4daf4a", label=f"Mixed/Gender fixed (AUC={mx_auc:.3f})")
ax.plot([], [], "s", color="grey", label="Pooled no odds: AUC=0.637")
ax.plot([], [], "s", color="black", label="Odds benchmark: AUC=0.711")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC -- Sex-Stratified & Mixed Logistic Models (no odds)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/models/roc_sex_models.png", dpi=150)
plt.close()
print("\nSaved plots/models/roc_sex_models.png")

# ============================================================
# 5. Residual plots by sex (side by side)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Male deviance residuals vs fitted
ax = axes[0, 0]
ax.scatter(m_yhat, m_resid, alpha=0.12, s=6, color="#377eb8")
ax.axhline(0, color="black", linewidth=0.5)
smooth = lowess(m_resid, m_yhat, frac=0.3)
ax.plot(smooth[:, 0], smooth[:, 1], color="firebrick", lw=2)
ax.set_xlabel("Fitted P(Red)")
ax.set_ylabel("Deviance Residuals")
ax.set_title(f"Male (n={len(m_y)})")

# Female deviance residuals vs fitted
ax = axes[0, 1]
ax.scatter(f_yhat, f_resid, alpha=0.2, s=8, color="#e41a1c")
ax.axhline(0, color="black", linewidth=0.5)
smooth = lowess(f_resid, f_yhat, frac=0.3)
ax.plot(smooth[:, 0], smooth[:, 1], color="firebrick", lw=2)
ax.set_xlabel("Fitted P(Red)")
ax.set_ylabel("Deviance Residuals")
ax.set_title(f"Female (n={len(f_y)})")

# Male histogram
ax = axes[1, 0]
ax.hist(m_resid, bins=50, density=True, color="#377eb8", alpha=0.7)
ax.set_xlabel("Deviance Residuals")
ax.set_title("Male Residual Distribution")

# Female histogram
ax = axes[1, 1]
ax.hist(f_resid, bins=50, density=True, color="#e41a1c", alpha=0.7)
ax.set_xlabel("Deviance Residuals")
ax.set_title("Female Residual Distribution")

plt.suptitle("Residual Diagnostics by Sex -- Logistic (no odds)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/models/residuals_by_sex.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/residuals_by_sex.png")

# ============================================================
# 6. Summary table
# ============================================================
print(f"\n{'='*65}")
print("Model Comparison")
print(f"{'='*65}")
print(f"{'Model':<40} {'N':>6} {'Acc':>8} {'AUC':>8}")
print("-" * 64)
print(f"{'Pooled stats no odds (25)':<40} {'5487':>6} {'0.6167':>8} {'0.637':>8}")
print(f"{'Male only (25)':<40} {len(m_y):>6} {accuracy_score(m_y, (m_yhat>0.5).astype(int)):>8.4f} {m_auc:>8.4f}")
print(f"{'Female only (25)':<40} {len(f_y):>6} {accuracy_score(f_y, (f_yhat>0.5).astype(int)):>8.4f} {f_auc:>8.4f}")
print(f"{'Mixed / Gender fixed (26)':<40} {len(y_mixed):>6} {mx_acc:>8.4f} {mx_auc:>8.4f}")
print(f"{'Odds benchmark (2)':<40} {'6290':>6} {'0.6599':>8} {'0.711':>8}")
