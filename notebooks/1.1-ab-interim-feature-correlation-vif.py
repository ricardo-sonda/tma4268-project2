import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import load_ground_zero_frame

REPORTS_DIR = Path("reports/figures/correlation")
CORRELATION_PLOT_PATH = REPORTS_DIR / "interim-clean-feature-correlation.png"
VIF_PLOT_PATH = REPORTS_DIR / "interim-clean-feature-vif.png"

NON_FEATURE_COLUMNS = {
    "RedFighter",
    "BlueFighter",
    "Date",
    "EventDate",
    "Location",
    "Country",
    "Winner",
    "WinnerRed",
    "RedDecimalOdds",
    "BlueDecimalOdds",
    "RedImpliedProb",
    "BlueImpliedProb",
}


def build_numeric_feature_frame() -> pd.DataFrame:
    frame = load_ground_zero_frame()
    feature_frame = frame.drop(columns=sorted(NON_FEATURE_COLUMNS & set(frame.columns)))
    return feature_frame.select_dtypes(include="number")


def strongest_correlation_pairs(correlation_matrix: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=0)
    pairs = correlation_matrix.abs().mask(mask).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_correlation"]
    return pairs.sort_values("abs_correlation", ascending=False).head(top_n).reset_index(drop=True)


def calculate_vif(feature_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in feature_frame.columns:
        target = feature_frame[column]
        predictors = feature_frame.drop(columns=[column])

        model = LinearRegression()
        model.fit(predictors, target)
        r_squared = model.score(predictors, target)
        vif = np.inf if r_squared >= 0.999999 else 1.0 / (1.0 - r_squared)
        rows.append({"feature": column, "vif": vif, "r_squared": r_squared})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def save_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    plt.title("Interim Clean Dataset Feature Correlation", pad=12)
    plt.tight_layout()
    plt.savefig(CORRELATION_PLOT_PATH, dpi=200)
    plt.close()


def save_vif_plot(vif_frame: pd.DataFrame, top_n: int = 15) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top_vif = vif_frame.head(top_n).sort_values("vif", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_vif["feature"], top_vif["vif"], color="#34699a")
    plt.axvline(5, color="#f39c12", linestyle="--", linewidth=1.5, label="VIF = 5")
    plt.axvline(10, color="#c0392b", linestyle="--", linewidth=1.5, label="VIF = 10")
    plt.xlabel("Variance Inflation Factor")
    plt.ylabel("Feature")
    plt.title("Top VIF Scores In Interim Clean Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIF_PLOT_PATH, dpi=200)
    plt.close()


numeric_features = build_numeric_feature_frame()
complete_case_features = numeric_features.dropna().reset_index(drop=True)
correlation_matrix = complete_case_features.corr()
top_correlations = strongest_correlation_pairs(correlation_matrix, top_n=20)
vif_frame = calculate_vif(complete_case_features)

save_correlation_heatmap(correlation_matrix)
save_vif_plot(vif_frame)

print(f"Numeric feature columns: {numeric_features.shape[1]}")
print(f"Rows before complete-case filter: {numeric_features.shape[0]}")
print(f"Rows after complete-case filter: {complete_case_features.shape[0]}")
print(f"Saved correlation heatmap to: {CORRELATION_PLOT_PATH}")
print(f"Saved VIF plot to: {VIF_PLOT_PATH}")

print("\nStrongest absolute correlations")
print(top_correlations.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

print("\nHighest VIF scores")
print(vif_frame.head(20).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
