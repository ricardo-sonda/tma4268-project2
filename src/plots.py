"""Generate EDA plots for the UFC dataset."""

import argparse

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, UFC_MASTER_CSV


def plot_missingness() -> None:
    output_dir = FIGURES_DIR / "missing"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(UFC_MASTER_CSV)

    msno.matrix(df, figsize=(16, 8), fontsize=8, sparkline=True)
    plt.savefig(output_dir / "missing_matrix.png", dpi=150)
    plt.close()

    msno.bar(df, figsize=(16, 8), fontsize=8)
    plt.savefig(output_dir / "missing_bar.png", dpi=150)
    plt.close()

    msno.heatmap(df, figsize=(14, 10), fontsize=8)
    plt.savefig(output_dir / "missing_heatmap.png", dpi=150)
    plt.close()

    msno.dendrogram(df, figsize=(14, 8), fontsize=8)
    plt.savefig(output_dir / "missing_dendrogram.png", dpi=150)
    plt.close()


def plot_correlations() -> None:
    output_dir = FIGURES_DIR / "correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(UFC_MASTER_CSV)
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.isnull().mean() < 0.5]
    corr = numeric.corr()

    grid = sns.clustermap(
        corr,
        cmap="coolwarm",
        center=0,
        figsize=(22, 20),
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=0.08,
        vmin=-1,
        vmax=1,
    )
    grid.ax_heatmap.tick_params(labelsize=6)
    grid.fig.suptitle("Clustered Correlation Heatmap (numeric, <50% missing)", y=1.01, fontsize=14)
    grid.savefig(output_dir / "corr_clustered.png", dpi=150, bbox_inches="tight")
    plt.close()

    fighter_stats = [
        "AvgSigStrLanded", "AvgSigStrPct", "AvgSubAtt",
        "AvgTDLanded", "AvgTDPct", "CurrentLoseStreak",
        "CurrentWinStreak", "Draws", "Losses", "Wins",
        "TotalRoundsFought", "TotalTitleBouts", "LongestWinStreak",
        "WinsByKO", "WinsBySubmission", "WinsByDecisionUnanimous",
        "HeightCms", "ReachCms", "WeightLbs", "Age",
    ]
    common = [stat for stat in fighter_stats if f"Red{stat}" in numeric.columns and f"Blue{stat}" in numeric.columns]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    red_corr = numeric[[f"Red{stat}" for stat in common]].corr()
    red_corr.index = common
    red_corr.columns = common
    sns.heatmap(red_corr, ax=axes[0], cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, vmin=-1, vmax=1)
    axes[0].set_title("Red Fighter Stats Correlation")
    axes[0].tick_params(labelsize=7)

    blue_corr = numeric[[f"Blue{stat}" for stat in common]].corr()
    blue_corr.index = common
    blue_corr.columns = common
    sns.heatmap(blue_corr, ax=axes[1], cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, vmin=-1, vmax=1)
    axes[1].set_title("Blue Fighter Stats Correlation")
    axes[1].tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "corr_fighter_stats.png", dpi=150, bbox_inches="tight")
    plt.close()

    difference_columns = [column for column in numeric.columns if column.endswith("Dif")]
    if len(difference_columns) > 1:
        difference_corr = numeric[difference_columns].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(difference_corr, ax=ax, cmap="coolwarm", center=0, annot=True, fmt=".2f", vmin=-1, vmax=1)
        ax.set_title("Difference Features Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "corr_dif_features.png", dpi=150, bbox_inches="tight")
        plt.close()

    encoded = numeric.copy()
    encoded["WinnerRed"] = (df["Winner"] == "Red").astype(int)
    target_corr = encoded.corr()["WinnerRed"].drop("WinnerRed").sort_values()
    fig, ax = plt.subplots(figsize=(10, 14))
    target_corr.plot.barh(ax=ax, color=np.where(target_corr > 0, "firebrick", "steelblue"))
    ax.set_title("Correlation with Winner (Red=1, Blue=0)")
    ax.set_xlabel("Pearson r")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "corr_with_winner.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_all() -> None:
    if not UFC_MASTER_CSV.exists():
        raise FileNotFoundError(f"{UFC_MASTER_CSV} not found. Run the data pipeline first.")
    plot_missingness()
    plot_correlations()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UFC plots.")
    parser.add_argument("command", choices=["all", "missing", "correlation"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "missing":
        plot_missingness()
    elif args.command == "correlation":
        plot_correlations()
    else:
        plot_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
