"""Per-strategy confusion matrices on the chronological holdout test split.

For every model in the active registry, reads the standardized prediction CSV
written by `src/evaluation.py` and plots the model's confusion matrix
side-by-side with the bookmaker confusion matrix on the same test split.

Outputs are written to `reports/figures/confusion_matrix/{model_name}.png`.

Run from repo root:

    uv run python notebooks/confusion_matrices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR, PREDICTIONS_DIR
from src.modeling.registry import MODEL_CLASSES

OUTPUT_DIR = FIGURES_DIR / "confusion_matrix"

CLASS_LABELS = ("Blue wins", "Red wins")


def _plot_cm(ax: plt.Axes, cm: np.ndarray, title: str, accuracy: float) -> None:
    """Render a 2x2 confusion matrix with counts and row-normalized percentages."""
    row_totals = cm.sum(axis=1, keepdims=True)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    cm_norm = cm / row_totals_safe

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = cm_norm[i, j] * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title}\naccuracy = {accuracy:.4f}")

    return im


def plot_one_model(model_name: str) -> Path | None:
    csv_path = PREDICTIONS_DIR / f"{model_name}.csv"
    if not csv_path.exists():
        print(f"skip {model_name}: no prediction CSV at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    y_true = df["WinnerRed"].to_numpy(dtype=int)
    y_model = df["ModelPredRed"].to_numpy(dtype=int)
    y_book = df["BookmakerPredRed"].to_numpy(dtype=int)

    cm_model = confusion_matrix(y_true, y_model, labels=[0, 1])
    cm_book = confusion_matrix(y_true, y_book, labels=[0, 1])
    acc_model = accuracy_score(y_true, y_model)
    acc_book = accuracy_score(y_true, y_book)

    fig, (ax_model, ax_book) = plt.subplots(
        1, 2, figsize=(10, 4.6), constrained_layout=True
    )

    _plot_cm(ax_model, cm_model, model_name, acc_model)
    im = _plot_cm(ax_book, cm_book, "bookmaker", acc_book)

    fig.colorbar(im, ax=(ax_model, ax_book), shrink=0.85, label="row-normalized")
    fig.suptitle(
        f"Confusion matrix  ·  test n = {len(df)}  ·  chronological 80/20 split",
        fontsize=11,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    written: list[Path] = []
    for name in sorted(MODEL_CLASSES):
        path = plot_one_model(name)
        if path is not None:
            written.append(path)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
