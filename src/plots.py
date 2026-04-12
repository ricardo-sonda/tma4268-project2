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


def plot_feature_importances(model_name: str, importances: pd.Series) -> None:
    """Save a horizontal bar chart of feature importances."""
    output_dir = FIGURES_DIR / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)

    top = importances.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Blues_r", len(top))
    top.plot.barh(ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importances — {model_name}", fontsize=12)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_importance.png", dpi=150)
    plt.close()


def plot_class_balance_and_timeline() -> None:
    """Bar chart of Red/Blue win counts and bout count per year."""
    from .features import load_ground_zero_frame

    output_dir = FIGURES_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_ground_zero_frame()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Class balance
    counts = frame["Winner"].value_counts().reindex(["Red", "Blue"])
    axes[0].bar(counts.index, counts.values, color=["firebrick", "steelblue"], width=0.5)
    axes[0].set_title("Win counts by corner", fontsize=12)
    axes[0].set_ylabel("Number of bouts")
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(i, val + 20, f"{val}\n({val/len(frame)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=10)

    # Bouts per year
    frame["Year"] = frame["EventDate"].dt.year
    yearly = frame.groupby("Year").size()
    axes[1].bar(yearly.index, yearly.values, color="steelblue")
    axes[1].set_title("Bouts per year", fontsize=12)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Number of bouts")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "class_balance_timeline.png", dpi=150)
    plt.close()


def plot_outcome_feature_violin() -> None:
    """Violin plots of key diff features split by fight outcome."""
    from .features import diff_feature_set

    output_dir = FIGURES_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = diff_feature_set()
    plot_frame = ds.X.copy()
    plot_frame["Outcome"] = ds.y.map({1: "Red wins", 0: "Blue wins"})

    features = [
        "CurrentWinStreakDiff",
        "AgeDiff",
        "ReachCmsDiff",
        "AvgSigStrPctDiff",
        "AvgTDLandedDiff",
        "LossesDiff",
    ]
    features = [f for f in features if f in plot_frame.columns]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    palette = {"Red wins": "firebrick", "Blue wins": "steelblue"}
    for ax, feat in zip(axes, features):
        data = plot_frame[[feat, "Outcome"]].dropna()
        sns.violinplot(
            data=data, x="Outcome", y=feat,
            hue="Outcome", palette=palette, legend=False, ax=ax,
            inner="quartile", linewidth=0.8,
        )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    plt.suptitle("Key difference features by fight outcome", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_violin_by_outcome.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_all() -> None:
    if not UFC_MASTER_CSV.exists():
        raise FileNotFoundError(f"{UFC_MASTER_CSV} not found. Run the data pipeline first.")
    plot_missingness()
    plot_correlations()
    plot_class_balance_and_timeline()
    plot_outcome_feature_violin()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UFC plots.")
    parser.add_argument(
        "command",
        choices=["all", "missing", "correlation", "eda", "feature-importance"],
    )
    parser.add_argument("--model", default=None, help="Model name for feature-importance command")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "missing":
        plot_missingness()
    elif args.command == "correlation":
        plot_correlations()
    elif args.command == "eda":
        plot_class_balance_and_timeline()
        plot_outcome_feature_violin()
    elif args.command == "feature-importance":
        if args.model is None:
            raise SystemExit("--model is required for feature-importance command")
        # Dynamically import to avoid hard dependency at module level
        from .modeling.registry import MODEL_CLASSES
        from .evaluation import chronological_split

        model_cls = MODEL_CLASSES[args.model]
        model = model_cls()
        ds = model.feature_builder()
        X_train, _, y_train, _, _, _ = chronological_split(ds.X, ds.y, ds.metadata)
        model.fit(X_train, y_train)

        rf_step = model.pipeline.named_steps.get("model")
        if rf_step is None or not hasattr(rf_step, "feature_importances_"):
            raise SystemExit(f"Model {args.model!r} does not expose feature_importances_")

        prep = model.pipeline.named_steps["prep"]
        names = prep.get_feature_names_out()
        importances = pd.Series(rf_step.feature_importances_, index=names)
        plot_feature_importances(args.model, importances)
    else:
        plot_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
