import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")

# Select only numeric columns and drop those with >50% missing
numeric = df.select_dtypes(include="number")
numeric = numeric.loc[:, numeric.isnull().mean() < 0.5]

print(f"Numeric columns with <50% missing: {numeric.shape[1]}")

# --- 1. Full correlation heatmap (clustered) ---
corr = numeric.corr()
g = sns.clustermap(corr, cmap="coolwarm", center=0, figsize=(22, 20),
                   linewidths=0, xticklabels=True, yticklabels=True,
                   dendrogram_ratio=0.08, vmin=-1, vmax=1)
g.ax_heatmap.tick_params(labelsize=6)
g.fig.suptitle("Clustered Correlation Heatmap (numeric, <50% missing)", y=1.01, fontsize=14)
g.savefig("plots/correlation/corr_clustered.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/correlation/corr_clustered.png")

# --- 2. Top correlated pairs ---
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (upper.stack()
         .reset_index()
         .rename(columns={"level_0": "var1", "level_1": "var2", 0: "r"}))
pairs["abs_r"] = pairs["r"].abs()
pairs = pairs.sort_values("abs_r", ascending=False)

print("\nTop 20 correlated pairs:")
print(pairs.head(20).to_string(index=False))

print("\nTop 20 negatively correlated pairs:")
print(pairs[pairs["r"] < 0].head(20).to_string(index=False))

# --- 3. Fighter stats comparison (Red vs Blue mirror) ---
fighter_stats = [
    "AvgSigStrLanded", "AvgSigStrPct", "AvgSubAtt",
    "AvgTDLanded", "AvgTDPct", "CurrentLoseStreak",
    "CurrentWinStreak", "Draws", "Losses", "Wins",
    "TotalRoundsFought", "TotalTitleBouts", "LongestWinStreak",
    "WinsByKO", "WinsBySubmission", "WinsByDecisionUnanimous",
    "HeightCms", "ReachCms", "WeightLbs", "Age",
]

red_cols = [f"Red{s}" for s in fighter_stats if f"Red{s}" in numeric.columns]
blue_cols = [f"Blue{s}" for s in fighter_stats if f"Blue{s}" in numeric.columns]
common = [s for s in fighter_stats if f"Red{s}" in numeric.columns and f"Blue{s}" in numeric.columns]

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Red fighter stats
red_corr = numeric[[f"Red{s}" for s in common]].corr()
red_corr.index = common
red_corr.columns = common
sns.heatmap(red_corr, ax=axes[0], cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 6}, vmin=-1, vmax=1)
axes[0].set_title("Red Fighter Stats Correlation")
axes[0].tick_params(labelsize=7)

# Blue fighter stats
blue_corr = numeric[[f"Blue{s}" for s in common]].corr()
blue_corr.index = common
blue_corr.columns = common
sns.heatmap(blue_corr, ax=axes[1], cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 6}, vmin=-1, vmax=1)
axes[1].set_title("Blue Fighter Stats Correlation")
axes[1].tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("plots/correlation/corr_fighter_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/correlation/corr_fighter_stats.png")

# --- 4. Difference features correlation ---
dif_cols = [c for c in numeric.columns if c.endswith("Dif")]
if len(dif_cols) > 1:
    dif_corr = numeric[dif_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(dif_corr, ax=ax, cmap="coolwarm", center=0, annot=True,
                fmt=".2f", vmin=-1, vmax=1)
    ax.set_title("Difference Features Correlation")
    plt.tight_layout()
    plt.savefig("plots/correlation/corr_dif_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/correlation/corr_dif_features.png")

# --- 5. Correlation with Winner (encode as 1=Red, 0=Blue) ---
df_enc = numeric.copy()
df_enc["WinnerRed"] = (df["Winner"] == "Red").astype(int)
target_corr = df_enc.corr()["WinnerRed"].drop("WinnerRed").sort_values()

fig, ax = plt.subplots(figsize=(10, 14))
target_corr.plot.barh(ax=ax, color=np.where(target_corr > 0, "firebrick", "steelblue"))
ax.set_title("Correlation with Winner (Red=1, Blue=0)")
ax.set_xlabel("Pearson r")
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/correlation/corr_with_winner.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/correlation/corr_with_winner.png")

print("\nTop predictors of Red winning:")
print(target_corr.abs().sort_values(ascending=False).head(15))
