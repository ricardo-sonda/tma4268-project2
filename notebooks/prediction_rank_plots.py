"""Per-model plots: test fights ordered by predicted P(Red wins), decreasing.

The center panel is the model probability curve (monotone after sorting).
The top panel is a histogram of how many predictions were correct in each
rank bin (fights with similar model confidence). The bottom panel is the same
for failed predictions.

Run from repo root:

    uv run python notebooks/prediction_rank_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR
from src.evaluation import chronological_split
from src.modeling.registry import MODEL_CLASSES

OUTPUT_DIR = FIGURES_DIR / "prediction_rank"

HIST_BINS = 28


def _binned_counts(
    correct_s: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate correct / wrong counts along fight rank (0 .. n-1)."""
    n = len(correct_s)
    n_bins = max(1, min(n_bins, n))
    edges = np.linspace(0, n, n_bins + 1)
    correct_counts = np.zeros(n_bins)
    wrong_counts = np.zeros(n_bins)
    for i in range(n_bins):
        lo = int(np.floor(edges[i]))
        hi = int(np.ceil(edges[i + 1]))
        hi = min(hi, n)
        if lo >= hi:
            continue
        seg = correct_s[lo:hi]
        c = int(seg.sum())
        correct_counts[i] = c
        wrong_counts[i] = len(seg) - c
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges) * 0.92
    return centers, widths, correct_counts, wrong_counts


def plot_one_model(model_name: str, model_cls: type) -> Path:
    model = model_cls()
    prepared = model.feature_builder()
    X_train, X_test, y_train, y_test, _, _ = chronological_split(
        prepared.X,
        prepared.y,
        prepared.metadata,
    )

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    y = y_test.to_numpy(dtype=int)
    pred_class = (prob >= 0.5).astype(int)
    correct = pred_class == y

    order = np.argsort(-prob)
    prob_s = prob[order]
    correct_s = correct[order]

    x = np.arange(len(prob_s))
    centers, widths, correct_c, wrong_c = _binned_counts(correct_s, HIST_BINS)

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(10, 6.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (112, 180, 112), "hspace": 0.08},
    )

    ax_top.bar(centers, correct_c, width=widths, color="#2ca02c", edgecolor="none", align="center")
    ax_top.set_ylabel("Correct\n(count)", fontsize=9)
    ax_top.set_title(f"{model_name}  ·  test n={len(prob_s)}")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, axis="y", ls=":", alpha=0.5)

    ax_mid.plot(x, prob_s, color="0.2", lw=1.2, label="P(Red wins)")
    ax_mid.set_ylabel("P(Red wins)")
    ax_mid.set_ylim(-0.08, 1.08)
    ax_mid.grid(True, axis="y", ls=":", alpha=0.5)
    ax_mid.legend(loc="upper right", framealpha=0.9)
    ax_mid.tick_params(axis="x", labelbottom=False)

    ax_bot.bar(centers, wrong_c, width=widths, color="#d62728", edgecolor="none", align="center")
    ax_bot.set_ylabel("Failed\n(count)", fontsize=9)
    ax_bot.set_xlabel("Fights (sorted by predicted probability, high to low)")

    x_max = len(prob_s) - 1
    for ax in (ax_top, ax_mid, ax_bot):
        ax.set_xlim(-0.5, max(0.5, x_max) + 0.5)

    ax_bot.grid(True, axis="y", ls=":", alpha=0.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    paths = []
    for name in sorted(MODEL_CLASSES):
        paths.append(plot_one_model(name, MODEL_CLASSES[name]))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
