import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/ultimate-ufc/ufc-clean.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42)

# --- Logistic regression baseline ---
numeric_features = [
    # "RedAge",
    # "BlueAge",
    "RedHeightCms",
    "BlueHeightCms",
    "RedReachCms",
    "BlueReachCms",
    "RedCurrentWinStreak",
    "BlueCurrentWinStreak",
    "RedCurrentLoseStreak",
    "BlueCurrentLoseStreak",
    # "RedWins",
    # "BlueWins",
    # "RedLosses",
    # "BlueLosses",
    "RedAvgSigStrLanded",
    "BlueAvgSigStrLanded",
    "RedAvgSigStrPct",
    "BlueAvgSigStrPct",
    "RedAvgTDLanded",
    "BlueAvgTDLanded",
    "RedAvgTDPct",
    "BlueAvgTDPct",
]
target = "Winner"

# Calculate correlation matrix (encode Winner so it is numeric for corr)
corr_df = train[numeric_features + [target]].copy()
corr_df[target] = pd.Categorical(corr_df[target]).codes
corr_matrix = corr_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"shrink": 0.72, "label": "Pearson r"},
)
plt.title("Correlation matrix (lower triangle)", pad=12)
plt.tight_layout()
plt.show(block=True)
