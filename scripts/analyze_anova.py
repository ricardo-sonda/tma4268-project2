import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")

# Encode target: Winner = 1 if Red, 0 if Blue (drop draws/no contests)
df = df[df["Winner"].isin(["Red", "Blue"])].copy()
df["WinnerRed"] = (df["Winner"] == "Red").astype(int)

# --- Select predictors: meaningful, <15% missing ---
predictors = [
    # Betting odds
    "RedOdds", "BlueOdds",
    # Physical attributes
    "RedHeightCms", "BlueHeightCms",
    "RedReachCms", "BlueReachCms",
    "RedWeightLbs",  # drop BlueWeightLbs (r=0.97 with Red)
    "RedAge", "BlueAge",
    # Fight stats
    "RedAvgSigStrLanded", "BlueAvgSigStrLanded",
    "RedAvgSigStrPct", "BlueAvgSigStrPct",
    "RedAvgSubAtt", "BlueAvgSubAtt",
    "RedAvgTDLanded", "BlueAvgTDLanded",
    "RedAvgTDPct", "BlueAvgTDPct",
    # Record
    "RedWins", "BlueWins",
    "RedLosses", "BlueLosses",
    "RedCurrentWinStreak", "BlueCurrentWinStreak",
    # Bout context
    "NumberOfRounds", "TitleBout",
]

# Encode TitleBout
df["TitleBout"] = df["TitleBout"].astype(int)

# Drop rows with missing values in selected predictors
model_df = df[["WinnerRed"] + predictors].dropna()
print(f"Rows after dropping NAs: {len(model_df)} / {len(df)}")
print(f"Predictors: {len(predictors)}\n")

# ============================================================
# 1. OLS linear probability model + ANOVA (Type II)
# ============================================================
# Use formula API so anova_lm can do Type II SS
formula = "WinnerRed ~ " + " + ".join(predictors)
ols = smf.ols(formula, data=model_df).fit()
print("=" * 70)
print("OLS Linear Probability Model Summary")
print("=" * 70)
print(ols.summary())

# Also keep matrix version for VIF
X = model_df[predictors]
X_const = sm.add_constant(X)

# Type II ANOVA
print("\n" + "=" * 70)
print("ANOVA (Type II) — each predictor adjusted for all others")
print("=" * 70)
anova_table = anova_lm(ols, typ=2)
anova_table["pct_SS"] = (anova_table["sum_sq"] / anova_table["sum_sq"].sum() * 100).round(2)
anova_table = anova_table.sort_values("sum_sq", ascending=False)
print(anova_table.to_string())

# Plot ANOVA sum of squares
fig, ax = plt.subplots(figsize=(10, 8))
anova_plot = anova_table.drop("Residual", errors="ignore").sort_values("sum_sq")
colors = ["firebrick" if p < 0.05 else "grey" for p in anova_plot["PR(>F)"]]
anova_plot["sum_sq"].plot.barh(ax=ax, color=colors)
ax.set_xlabel("Sum of Squares (Type II)")
ax.set_title("ANOVA — predictor importance (red = p < 0.05)")
plt.tight_layout()
plt.savefig("plots/anova/anova_ss.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/anova/anova_ss.png")

# ============================================================
# 2. Collinearity analysis — VIF
# ============================================================
print("\n" + "=" * 70)
print("Variance Inflation Factors")
print("=" * 70)
vif_data = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
vif_sorted = vif_data.sort_values("VIF")
colors = np.where(vif_sorted["VIF"] > 10, "firebrick",
         np.where(vif_sorted["VIF"] > 5, "orange", "steelblue"))
ax.barh(vif_sorted["Variable"], vif_sorted["VIF"], color=colors)
ax.axvline(5, color="orange", linestyle="--", label="VIF=5 (moderate)")
ax.axvline(10, color="firebrick", linestyle="--", label="VIF=10 (high)")
ax.set_xlabel("VIF")
ax.set_title("Variance Inflation Factors")
ax.legend()
plt.tight_layout()
plt.savefig("plots/anova/vif_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/anova/vif_analysis.png")

# ============================================================
# 3. Condition number & eigenvalue analysis
# ============================================================
print("\n" + "=" * 70)
print("Condition Number & Eigenvalue Analysis")
print("=" * 70)

# Standardize for eigenvalue analysis
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
corr_matrix = np.corrcoef(X_std.T)
eigenvalues = np.linalg.eigvalsh(corr_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]
condition_indices = np.sqrt(eigenvalues[0] / eigenvalues)

print(f"\nCondition number of X'X: {np.linalg.cond(X_const.T @ X_const):.1f}")
print(f"Max condition index: {condition_indices.max():.1f}")
print(f"\nEigenvalues and condition indices:")
eig_df = pd.DataFrame({
    "Eigenvalue": eigenvalues.round(4),
    "Condition Index": condition_indices.round(2),
    "Flag": np.where(condition_indices > 30, "SEVERE",
            np.where(condition_indices > 10, "MODERATE", ""))
})
print(eig_df.to_string(index=False))

# ============================================================
# 4. Correlation heatmap of selected predictors
# ============================================================
fig, ax = plt.subplots(figsize=(14, 12))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 6}, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Matrix of Selected Predictors")
plt.tight_layout()
plt.savefig("plots/correlation/corr_predictors.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/correlation/corr_predictors.png")

# ============================================================
# 5. Significant predictors summary
# ============================================================
print("\n" + "=" * 70)
print("Significant predictors (p < 0.05)")
print("=" * 70)
sig = ols.pvalues[ols.pvalues < 0.05].drop("Intercept", errors="ignore")
coefs = ols.params[sig.index]
vif_lookup = vif_data.set_index("Variable")["VIF"]
summary_sig = pd.DataFrame({
    "coef": coefs.round(4),
    "p-value": sig.round(4),
    "VIF": [vif_lookup.get(v, np.nan) for v in sig.index]
}).sort_values("p-value")
summary_sig["VIF"] = summary_sig["VIF"].round(2)
print(summary_sig.to_string())
