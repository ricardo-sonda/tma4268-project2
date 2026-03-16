import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import jarque_bera
from scipy import stats

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

model_df = df[["WinnerRed"] + predictors].dropna()
print(f"Rows: {len(model_df)}")

# ============================================================
# Fit logistic regression via statsmodels (for deviance residuals)
# ============================================================
formula = "WinnerRed ~ " + " + ".join(predictors)
logit = smf.logit(formula, data=model_df).fit(disp=0)
print(logit.summary())

y = model_df["WinnerRed"].values
y_hat = logit.predict()

# ============================================================
# Residual types
# ============================================================
# Pearson residuals
pearson_resid = (y - y_hat) / np.sqrt(y_hat * (1 - y_hat))

# Deviance residuals
sign = np.where(y == 1, 1, -1)
deviance_resid = sign * np.sqrt(
    -2 * (y * np.log(np.clip(y_hat, 1e-10, 1)) +
           (1 - y) * np.log(np.clip(1 - y_hat, 1e-10, 1)))
)

# Leverage (hat values)
X = logit.model.exog
W = np.diag(y_hat * (1 - y_hat))
try:
    H = X @ np.linalg.solve(X.T @ W @ X, X.T @ W)
    leverage = np.diag(H)
except np.linalg.LinAlgError:
    leverage = np.full(len(y), np.nan)

# Standardized Pearson residuals
std_pearson = pearson_resid / np.sqrt(1 - leverage)

# Cook's distance (approximate for GLM)
k = X.shape[1]
cooks_d = (std_pearson ** 2 * leverage) / (k * (1 - leverage))

print(f"\nDeviance residuals — mean: {deviance_resid.mean():.4f}, std: {deviance_resid.std():.4f}")
print(f"Pearson residuals  — mean: {pearson_resid.mean():.4f}, std: {pearson_resid.std():.4f}")

# ============================================================
# 1. Deviance residuals vs fitted values
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
ax.scatter(y_hat, deviance_resid, alpha=0.15, s=8, color="steelblue")
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(2, color="red", linewidth=0.5, linestyle="--")
ax.axhline(-2, color="red", linewidth=0.5, linestyle="--")
# Lowess smoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
smooth = lowess(deviance_resid, y_hat, frac=0.3)
ax.plot(smooth[:, 0], smooth[:, 1], color="firebrick", linewidth=2)
ax.set_xlabel("Fitted probability P(Red wins)")
ax.set_ylabel("Deviance Residuals")
ax.set_title("Deviance Residuals vs Fitted Values")

# ============================================================
# 2. Pearson residuals vs fitted values
# ============================================================
ax = axes[0, 1]
ax.scatter(y_hat, pearson_resid, alpha=0.15, s=8, color="steelblue")
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(2, color="red", linewidth=0.5, linestyle="--")
ax.axhline(-2, color="red", linewidth=0.5, linestyle="--")
smooth_p = lowess(pearson_resid, y_hat, frac=0.3)
ax.plot(smooth_p[:, 0], smooth_p[:, 1], color="firebrick", linewidth=2)
ax.set_xlabel("Fitted probability P(Red wins)")
ax.set_ylabel("Pearson Residuals")
ax.set_title("Pearson Residuals vs Fitted Values")

# ============================================================
# 3. Scale-location (sqrt standardized residuals)
# ============================================================
ax = axes[1, 0]
sqrt_std = np.sqrt(np.abs(std_pearson))
ax.scatter(y_hat, sqrt_std, alpha=0.15, s=8, color="steelblue")
smooth_sl = lowess(sqrt_std, y_hat, frac=0.3)
ax.plot(smooth_sl[:, 0], smooth_sl[:, 1], color="firebrick", linewidth=2)
ax.set_xlabel("Fitted probability P(Red wins)")
ax.set_ylabel("√|Standardized Pearson Residual|")
ax.set_title("Scale-Location Plot")

# ============================================================
# 4. Leverage vs standardized residuals (Cook's distance)
# ============================================================
ax = axes[1, 1]
ax.scatter(leverage, std_pearson, alpha=0.15, s=8, color="steelblue")
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(2, color="red", linewidth=0.5, linestyle="--")
ax.axhline(-2, color="red", linewidth=0.5, linestyle="--")
# Cook's distance contours
lev_range = np.linspace(0.001, leverage.max(), 100)
for D in [0.5, 1.0]:
    cook_bound = np.sqrt(D * k * (1 - lev_range) / lev_range)
    ax.plot(lev_range, cook_bound, "g--", linewidth=1,
            label=f"Cook's D={D}" if D == 0.5 else f"Cook's D={D}")
    ax.plot(lev_range, -cook_bound, "g--", linewidth=1)
ax.set_xlabel("Leverage")
ax.set_ylabel("Standardized Pearson Residuals")
ax.set_title("Residuals vs Leverage")
ax.legend(fontsize=8)

plt.suptitle("Residual Diagnostics — Logistic Regression (no odds, 25 predictors)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/models/residuals_logistic.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plots/models/residuals_logistic.png")

# ============================================================
# 5. Binned residual plot (Gelman approach)
# ============================================================
order = np.argsort(y_hat)
n_bins = 40
bin_size = len(y) // n_bins

bin_means = []
bin_resid_means = []
bin_resid_se = []

for i in range(n_bins):
    start = i * bin_size
    end = start + bin_size if i < n_bins - 1 else len(y)
    idx = order[start:end]
    bin_means.append(y_hat[idx].mean())
    r = y[idx] - y_hat[idx]
    bin_resid_means.append(r.mean())
    bin_resid_se.append(r.std() / np.sqrt(len(idx)))

bin_means = np.array(bin_means)
bin_resid_means = np.array(bin_resid_means)
bin_resid_se = np.array(bin_resid_se)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(bin_means, bin_resid_means, color="steelblue", s=40, zorder=3)
ax.fill_between(bin_means,
                -2 * bin_resid_se, 2 * bin_resid_se,
                alpha=0.2, color="grey", label="±2 SE")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Binned Fitted Probability")
ax.set_ylabel("Average Residual (observed - predicted)")
ax.set_title("Binned Residual Plot (Gelman) — Logistic, no odds")
ax.legend()
plt.tight_layout()
plt.savefig("plots/models/residuals_binned.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/residuals_binned.png")

# ============================================================
# 6. Residual distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(deviance_resid, bins=50, density=True, color="steelblue", alpha=0.7)
x_range = np.linspace(-4, 4, 200)
axes[0].plot(x_range, stats.norm.pdf(x_range), "r-", lw=2, label="N(0,1)")
axes[0].set_xlabel("Deviance Residuals")
axes[0].set_title("Distribution of Deviance Residuals")
axes[0].legend()

sm.qqplot(deviance_resid, line="45", ax=axes[1], alpha=0.15, markersize=3)
axes[1].set_title("Q-Q Plot — Deviance Residuals")

plt.tight_layout()
plt.savefig("plots/models/residuals_qq.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/models/residuals_qq.png")

# ============================================================
# 7. Summary statistics
# ============================================================
print("\n" + "=" * 60)
print("Goodness of Fit")
print("=" * 60)
null_dev = -2 * logit.llnull
resid_dev = -2 * logit.llf
print(f"Null deviance:     {null_dev:.1f} on {logit.nobs - 1:.0f} df")
print(f"Residual deviance: {resid_dev:.1f} on {logit.df_resid:.0f} df")
print(f"AIC: {logit.aic:.1f}")
print(f"BIC: {logit.bic:.1f}")
print(f"Pseudo R² (McFadden): {logit.prsquared:.4f}")

# Hosmer-Lemeshow test
n_groups = 10
order = np.argsort(y_hat)
hl_stat = 0
print(f"\nHosmer-Lemeshow Test ({n_groups} groups):")
print(f"{'Group':>5} {'N':>6} {'Obs':>6} {'Exp':>8} {'(O-E)²/E':>10}")
group_size = len(y) // n_groups
for g in range(n_groups):
    start = g * group_size
    end = start + group_size if g < n_groups - 1 else len(y)
    idx = order[start:end]
    obs = y[idx].sum()
    exp = y_hat[idx].sum()
    n = len(idx)
    hl_stat += (obs - exp) ** 2 / exp + ((n - obs) - (n - exp)) ** 2 / (n - exp)
    print(f"{g+1:>5} {n:>6} {obs:>6} {exp:>8.1f} {(obs-exp)**2/exp:>10.3f}")

hl_pval = 1 - stats.chi2.cdf(hl_stat, n_groups - 2)
print(f"\nHL statistic: {hl_stat:.2f}, df={n_groups-2}, p-value={hl_pval:.4f}")
if hl_pval < 0.05:
    print("-> Poor fit (reject H0 of adequate fit)")
else:
    print("-> Adequate fit (fail to reject H0)")

# High influence points
n_high_cook = (cooks_d > 4 / len(y)).sum()
n_high_leverage = (leverage > 2 * k / len(y)).sum()
n_large_resid = (np.abs(std_pearson) > 2).sum()
print(f"\nInfluential observations:")
print(f"  Cook's D > 4/n:           {n_high_cook} ({100*n_high_cook/len(y):.1f}%)")
print(f"  Leverage > 2p/n:          {n_high_leverage} ({100*n_high_leverage/len(y):.1f}%)")
print(f"  |Std Pearson resid| > 2:  {n_large_resid} ({100*n_large_resid/len(y):.1f}%)")
