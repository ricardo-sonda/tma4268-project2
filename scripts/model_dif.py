import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")

# Target: Red=1, Blue=0, drop draws/no contests
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
df["WinnerRed"] = (df["Winner"] == "Red").astype(int)

# Encode categorical bout context
df["TitleBout"] = df["TitleBout"].astype(int)
df["IsMale"] = (df["Gender"] == "MALE").astype(int)

# --- Predictors: difference features + bout context ---
dif_cols = [
    "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif",
    "WinDif", "LossDif", "TotalRoundDif", "TotalTitleBoutDif",
    "KODif", "SubDif",
    "HeightDif", "ReachDif", "AgeDif",
    "SigStrDif", "AvgSubAttDif", "AvgTDDif",
]

context_cols = ["TitleBout", "NumberOfRounds", "IsMale"]

predictors = dif_cols + context_cols
model_df = df[["WinnerRed"] + predictors].dropna()
print(f"Rows after dropping NAs: {len(model_df)} / {len(df)}")
print(f"Predictors: {len(predictors)}")
print(f"  Dif features: {len(dif_cols)}")
print(f"  Context features: {len(context_cols)}\n")

# ============================================================
# 1. OLS Linear Probability Model
# ============================================================
formula = "WinnerRed ~ " + " + ".join(predictors)
ols = smf.ols(formula, data=model_df).fit()
print("=" * 70)
print("OLS Linear Probability Model (Dif features, no odds)")
print("=" * 70)
print(ols.summary())

# ============================================================
# 2. ANOVA Type II
# ============================================================
print("\n" + "=" * 70)
print("ANOVA (Type II)")
print("=" * 70)
anova_table = anova_lm(ols, typ=2)
anova_table["pct_SS"] = (anova_table["sum_sq"] / anova_table["sum_sq"].sum() * 100).round(2)
anova_table = anova_table.sort_values("sum_sq", ascending=False)
print(anova_table.to_string())

fig, ax = plt.subplots(figsize=(10, 7))
anova_plot = anova_table.drop("Residual", errors="ignore").sort_values("sum_sq")
colors = ["firebrick" if p < 0.05 else "grey" for p in anova_plot["PR(>F)"]]
anova_plot["sum_sq"].plot.barh(ax=ax, color=colors)
ax.set_xlabel("Sum of Squares (Type II)")
ax.set_title("ANOVA — Dif model (red = p < 0.05)")
plt.tight_layout()
plt.savefig("plots/anova/anova_dif.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/anova/anova_dif.png")

# ============================================================
# 3. VIF
# ============================================================
print("\n" + "=" * 70)
print("Variance Inflation Factors")
print("=" * 70)
X = model_df[predictors]
vif_data = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
vif_sorted = vif_data.sort_values("VIF")
colors = np.where(vif_sorted["VIF"] > 10, "firebrick",
         np.where(vif_sorted["VIF"] > 5, "orange", "steelblue"))
ax.barh(vif_sorted["Variable"], vif_sorted["VIF"], color=colors)
ax.axvline(5, color="orange", linestyle="--", label="VIF=5")
ax.axvline(10, color="firebrick", linestyle="--", label="VIF=10")
ax.set_xlabel("VIF")
ax.set_title("VIF — Dif model")
ax.legend()
plt.tight_layout()
plt.savefig("plots/anova/vif_dif.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/anova/vif_dif.png")

# ============================================================
# 4. Condition indices
# ============================================================
print("\n" + "=" * 70)
print("Condition Index Analysis")
print("=" * 70)
X_std = StandardScaler().fit_transform(X)
corr_matrix = np.corrcoef(X_std.T)
eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
cond_idx = np.sqrt(eigenvalues[0] / eigenvalues)
eig_df = pd.DataFrame({
    "Eigenvalue": eigenvalues.round(4),
    "Cond Index": cond_idx.round(2),
    "Flag": np.where(cond_idx > 30, "SEVERE",
            np.where(cond_idx > 10, "MODERATE", ""))
})
print(eig_df.to_string(index=False))
print(f"\nMax condition index: {cond_idx.max():.1f}")

# ============================================================
# 5. Correlation heatmap of predictors
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 7}, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation — Dif Model Predictors")
plt.tight_layout()
plt.savefig("plots/correlation/corr_dif_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/correlation/corr_dif_model.png")

# ============================================================
# 6. Significant predictors summary
# ============================================================
print("\n" + "=" * 70)
print("Significant predictors (p < 0.05)")
print("=" * 70)
sig = ols.pvalues[ols.pvalues < 0.05].drop("Intercept", errors="ignore")
coefs = ols.params[sig.index]
vif_lookup = vif_data.set_index("Variable")["VIF"]
summary_sig = pd.DataFrame({
    "coef": coefs.round(4),
    "p-value": sig.map("{:.2e}".format),
    "VIF": [round(vif_lookup.get(v, np.nan), 2) for v in sig.index]
}).sort_values("coef", key=abs, ascending=False)
print(summary_sig.to_string())

# ============================================================
# 7. Compare with previous model R²
# ============================================================
print("\n" + "=" * 70)
print("Model comparison")
print("=" * 70)
print(f"Dif model (no odds):  R²={ols.rsquared:.4f}  Adj R²={ols.rsquared_adj:.4f}  AIC={ols.aic:.0f}")
print(f"Previous model (with odds, 27 predictors): R²=0.1280  Adj R²=0.1240  AIC=6828")
print(f"\nPredictors: {len(predictors)} vs 27")
