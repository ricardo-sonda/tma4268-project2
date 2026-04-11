"""Analyze and visualize missingness in the cleaned UFC dataset."""

import argparse

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd

from .config import FIGURES_DIR, UFC_CLEAN_CSV

OUTPUT_DIR = FIGURES_DIR / "missing"


def _load_missing_cols(max_cols: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df, subset_df) where subset contains only columns with missing values.

    Capped at *max_cols* columns, sorted descending by missingness rate.
    """
    df = pd.read_csv(UFC_CLEAN_CSV)
    missing_rate = df.isnull().mean()
    has_missing = missing_rate[missing_rate > 0].sort_values(ascending=False)
    top_cols = has_missing.head(max_cols).index.tolist()
    return df, df[top_cols]


def missing_bar(max_cols: int = 40) -> None:
    """Horizontal bar chart showing missingness rate for the top missing columns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(UFC_CLEAN_CSV)

    missing_rate = df.isnull().mean()
    has_missing = missing_rate[missing_rate > 0].sort_values(ascending=False).head(max_cols)

    if has_missing.empty:
        print("No missing values found in the cleaned dataset.")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(has_missing) * 0.35)))
    colors = ["#c0392b" if r > 0.5 else "#2980b9" for r in has_missing.values]
    ax.barh(has_missing.index[::-1], has_missing.values[::-1], color=colors[::-1])
    ax.set_xlabel("Missing fraction", fontsize=11)
    ax.set_title(
        f"Columns with missing values (top {len(has_missing)}, cleaned dataset)",
        fontsize=12,
    )
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", label="50%")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_bar.png", dpi=150)
    plt.close(fig)


def missing_matrix(max_cols: int = 40) -> None:
    """missingno matrix restricted to columns that actually have missing values."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, subset = _load_missing_cols(max_cols)
    if subset.empty:
        return
    msno.matrix(subset, figsize=(14, 7), fontsize=8, sparkline=True)
    plt.savefig(OUTPUT_DIR / "missing_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def missing_heatmap(max_cols: int = 30) -> None:
    """missingno co-occurrence heatmap for the most-missing columns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, subset = _load_missing_cols(max_cols)
    if subset.shape[1] < 2:
        return
    msno.heatmap(subset, figsize=(12, 10), fontsize=8)
    plt.savefig(OUTPUT_DIR / "missing_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def missing_dendrogram(max_cols: int = 30) -> None:
    """missingno dendrogram for the most-missing columns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, subset = _load_missing_cols(max_cols)
    if subset.shape[1] < 2:
        return
    msno.dendrogram(subset, figsize=(12, 7), fontsize=8)
    plt.savefig(OUTPUT_DIR / "missing_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()


def missing_summary() -> pd.DataFrame:
    """Return a DataFrame with missingness stats for every column that has gaps."""
    df = pd.read_csv(UFC_CLEAN_CSV)
    n = len(df)
    stats = (
        df.isnull()
        .sum()
        .rename("n_missing")
        .to_frame()
        .assign(pct_missing=lambda x: x["n_missing"] / n)
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )
    return stats


def plot_all() -> None:
    if not UFC_CLEAN_CSV.exists():
        raise FileNotFoundError(f"{UFC_CLEAN_CSV} not found. Run the data pipeline first.")
    missing_bar()
    missing_matrix()
    missing_heatmap()
    missing_dendrogram()
    summary = missing_summary()
    print(summary.to_string())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missingness analysis for the UFC dataset.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "bar", "matrix", "heatmap", "dendrogram", "summary"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = args.command
    if cmd == "bar":
        missing_bar()
    elif cmd == "matrix":
        missing_matrix()
    elif cmd == "heatmap":
        missing_heatmap()
    elif cmd == "dendrogram":
        missing_dendrogram()
    elif cmd == "summary":
        print(missing_summary().to_string())
    else:
        plot_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
