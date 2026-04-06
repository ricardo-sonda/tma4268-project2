"""Standardized chronological evaluation shared across all models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from .config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR
from .modeling.base import BaseModel

TEST_FRACTION = 0.2


@dataclass(frozen=True)
class EvaluationSummary:
    model_name: str
    feature_set: str
    train_rows: int
    test_rows: int
    accuracy: float
    bookmaker_accuracy: float
    log_loss: float
    bookmaker_log_loss: float
    brier: float
    bookmaker_brier: float
    roc_auc: float
    bookmaker_roc_auc: float


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    split_index = int(len(X) * (1 - test_fraction))
    split_index = min(max(1, split_index), len(X) - 1)
    return (
        X.iloc[:split_index].copy(),
        X.iloc[split_index:].copy(),
        y.iloc[:split_index].copy(),
        y.iloc[split_index:].copy(),
        metadata.iloc[:split_index].copy(),
        metadata.iloc[split_index:].copy(),
    )


def evaluate_model(model: BaseModel) -> EvaluationSummary:
    prepared = model.feature_builder()
    X_train, X_test, y_train, y_test, _, metadata_test = chronological_split(
        prepared.X,
        prepared.y,
        prepared.metadata,
    )

    model.fit(X_train, y_train)

    model_prob = model.predict_proba(X_test)
    model_pred = model.predict(X_test)
    bookmaker_prob = metadata_test["RedImpliedProb"].to_numpy()
    bookmaker_pred = (bookmaker_prob >= 0.5).astype(int)

    summary = EvaluationSummary(
        model_name=model.name,
        feature_set=prepared.name,
        train_rows=len(X_train),
        test_rows=len(X_test),
        accuracy=accuracy_score(y_test, model_pred),
        bookmaker_accuracy=accuracy_score(y_test, bookmaker_pred),
        log_loss=log_loss(y_test, model_prob),
        bookmaker_log_loss=log_loss(y_test, bookmaker_prob),
        brier=brier_score_loss(y_test, model_prob),
        bookmaker_brier=brier_score_loss(y_test, bookmaker_prob),
        roc_auc=roc_auc_score(y_test, model_prob),
        bookmaker_roc_auc=roc_auc_score(y_test, bookmaker_prob),
    )

    save_predictions(model.name, y_test, metadata_test, model_prob, bookmaker_prob)
    save_summary(summary)
    save_roc_plot(model.name, y_test, model_prob, bookmaker_prob)
    print_summary(summary)
    return summary


def save_predictions(
    model_name: str,
    y_test: pd.Series,
    metadata_test: pd.DataFrame,
    model_prob: np.ndarray,
    bookmaker_prob: np.ndarray,
) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_frame = metadata_test.copy()
    prediction_frame["WinnerRed"] = y_test.to_numpy()
    prediction_frame["ModelProbRed"] = model_prob
    prediction_frame["BookmakerProbRed"] = bookmaker_prob
    prediction_frame["ModelPredRed"] = (model_prob >= 0.5).astype(int)
    prediction_frame["BookmakerPredRed"] = (bookmaker_prob >= 0.5).astype(int)
    prediction_frame.to_csv(PREDICTIONS_DIR / f"{model_name}.csv", index=False)


def save_summary(summary: EvaluationSummary) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / f"{summary.model_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)


def save_roc_plot(
    model_name: str,
    y_test: pd.Series,
    model_prob: np.ndarray,
    bookmaker_prob: np.ndarray,
) -> None:
    output_dir = FIGURES_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_fpr, model_tpr, _ = roc_curve(y_test, model_prob)
    bookmaker_fpr, bookmaker_tpr, _ = roc_curve(y_test, bookmaker_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(model_fpr, model_tpr, lw=2, label=f"Model (AUC={roc_auc_score(y_test, model_prob):.3f})")
    ax.plot(
        bookmaker_fpr,
        bookmaker_tpr,
        lw=2,
        label=f"Bookmaker (AUC={roc_auc_score(y_test, bookmaker_prob):.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Comparison - {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_roc.png", dpi=150)
    plt.close()


def print_summary(summary: EvaluationSummary) -> None:
    print("=" * 60)
    print(f"Model: {summary.model_name}")
    print(f"Feature set: {summary.feature_set}")
    print("Split: chronological holdout (last 20% test)")
    print("=" * 60)
    print(f"Train rows:            {summary.train_rows}")
    print(f"Test rows:             {summary.test_rows}")
    print(f"Model accuracy:        {summary.accuracy:.4f}")
    print(f"Bookmaker accuracy:    {summary.bookmaker_accuracy:.4f}")
    print(f"Model log loss:        {summary.log_loss:.4f}")
    print(f"Bookmaker log loss:    {summary.bookmaker_log_loss:.4f}")
    print(f"Model Brier score:     {summary.brier:.4f}")
    print(f"Bookmaker Brier score: {summary.bookmaker_brier:.4f}")
    print(f"Model ROC AUC:         {summary.roc_auc:.4f}")
    print(f"Bookmaker ROC AUC:     {summary.bookmaker_roc_auc:.4f}")


def _delta_cell(model_val: float, book_val: float, lower_is_better: bool) -> Text:
    """Format a metric cell showing model value and delta vs bookmaker."""
    delta = model_val - book_val
    is_better = delta < 0 if lower_is_better else delta > 0

    val_text = f"{model_val:.4f}"
    if abs(delta) < 1e-6:
        return Text(val_text, style="dim")

    arrow = "+" if delta > 0 else ""
    delta_str = f" ({arrow}{delta:.4f})"
    text = Text(val_text, style="bold green" if is_better else "bold red")
    text.append(delta_str, style="green" if is_better else "red")
    return text


def print_leaderboard(summaries: list[EvaluationSummary]) -> None:
    """Print a rich leaderboard table sorted by log loss (primary metric)."""
    if not summaries:
        return

    console = Console()
    ranked = sorted(summaries, key=lambda s: s.log_loss)

    # Bookmaker reference (same across all models since same test split)
    book = ranked[0]

    # Header info
    console.print()
    console.print(
        Panel(
            f"[bold]{len(ranked)}[/bold] models  ·  "
            f"[bold]{ranked[0].train_rows}[/bold] train / "
            f"[bold]{ranked[0].test_rows}[/bold] test rows  ·  "
            f"chronological 80/20 split",
            title="[bold]Model Leaderboard[/bold]",
            subtitle="ranked by log loss (lower is better)",
            border_style="blue",
        )
    )

    # Bookmaker baseline row
    console.print(
        f"  [dim]Bookmaker baseline:[/dim]  "
        f"Acc [bold]{book.bookmaker_accuracy:.4f}[/bold]  "
        f"LogLoss [bold]{book.bookmaker_log_loss:.4f}[/bold]  "
        f"Brier [bold]{book.bookmaker_brier:.4f}[/bold]  "
        f"AUC [bold]{book.bookmaker_roc_auc:.4f}[/bold]"
    )
    console.print()

    table = Table(show_header=True, header_style="bold cyan", show_lines=True, pad_edge=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="bold white", min_width=20)
    table.add_column("Features", style="dim", min_width=12)
    table.add_column("Accuracy", min_width=18, justify="right")
    table.add_column("Log Loss", min_width=18, justify="right")
    table.add_column("Brier", min_width=18, justify="right")
    table.add_column("ROC AUC", min_width=18, justify="right")

    for i, s in enumerate(ranked, 1):
        rank_style = "bold yellow" if i == 1 else ("bold" if i <= 3 else "")
        rank_text = Text(str(i), style=rank_style)

        if i == 1:
            name_text = Text(f"* {s.model_name}", style="bold yellow")
        else:
            name_text = Text(s.model_name)

        table.add_row(
            rank_text,
            name_text,
            s.feature_set,
            _delta_cell(s.accuracy, s.bookmaker_accuracy, lower_is_better=False),
            _delta_cell(s.log_loss, s.bookmaker_log_loss, lower_is_better=True),
            _delta_cell(s.brier, s.bookmaker_brier, lower_is_better=True),
            _delta_cell(s.roc_auc, s.bookmaker_roc_auc, lower_is_better=False),
        )

    console.print(table)
    console.print(
        "  [dim]Delta vs bookmaker: [green]+better[/green] [red]+worse[/red][/dim]"
    )
    console.print()
