"""Parametric inference for logistic-regression UFC models.

Refits the educational logistic model (and the full complete-case diff logistic model)
without regularization, then computes Wald-style standard errors, z-statistics,
p-values, and 95% confidence intervals for each coefficient. The goal is to
show *what each beta is estimated to* and *how strong the evidence is*, in a
form that mirrors a textbook `summary()` table.

Outputs:
  reports/figures/parametric_inference/educational-coefficients.png
  reports/figures/parametric_inference/diff-complete-coefficients.png
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import chronological_split
from src.features import diff_complete_feature_set
from src.modeling.logistic_educational import LogisticEducationalModel

REPORTS_DIR = REPO_ROOT / "reports" / "figures" / "parametric_inference"
EDU_PLOT = REPORTS_DIR / "educational-coefficients.png"
DIFF_PLOT = REPORTS_DIR / "diff-complete-coefficients.png"

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

console = Console()


# ── Inference helpers ────────────────────────────────────────────────


def fit_unregularized_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit MLE logistic regression — no penalty, so betas are interpretable."""
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000)
    model.fit(X, y)
    return model


def wald_inference(
    model: LogisticRegression, X: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Compute beta, SE, z, p-value and 95% CI from the observed Fisher info.

    For logistic regression the asymptotic covariance of the MLE is
        Cov(beta_hat) = (X_aug^T W X_aug)^(-1),  W = diag(p (1 - p))
    where X_aug includes a leading intercept column.
    """
    intercept = float(model.intercept_[0])
    coefs = model.coef_[0]
    betas = np.concatenate([[intercept], coefs])
    names = ["(intercept)", *feature_names]

    X_aug = np.column_stack([np.ones(len(X)), X])
    p = model.predict_proba(X)[:, 1]
    W = p * (1.0 - p)

    fisher = (X_aug * W[:, None]).T @ X_aug
    # pinv tolerates near-collinear designs; the diagonals can still go
    # negative for severely ill-conditioned columns, so clip before sqrt
    cov = np.linalg.pinv(fisher)
    var = np.diag(cov)
    se = np.where(var > 0, np.sqrt(np.clip(var, 0, None)), np.nan)

    z = betas / se
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    ci_low = betas - 1.96 * se
    ci_high = betas + 1.96 * se

    return pd.DataFrame(
        {
            "feature": names,
            "beta": betas,
            "std_error": se,
            "z": z,
            "p_value": p_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "odds_ratio": np.exp(betas),
            "or_ci_low": np.exp(ci_low),
            "or_ci_high": np.exp(ci_high),
        }
    )


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


# ── Pretty printing ──────────────────────────────────────────────────


def print_summary_table(title: str, table: pd.DataFrame) -> None:
    rich_table = Table(title=title, show_header=True, header_style="bold")
    rich_table.add_column("feature", style="cyan", no_wrap=True)
    rich_table.add_column("beta", justify="right")
    rich_table.add_column("SE", justify="right")
    rich_table.add_column("z", justify="right")
    rich_table.add_column("p-value", justify="right")
    rich_table.add_column("95% CI (beta)", justify="right")
    rich_table.add_column("OR", justify="right")
    rich_table.add_column("sig", justify="left")

    for row in table.itertuples():
        sig = significance_stars(row.p_value)
        if row.p_value < 0.05:
            row_style = "green"
        elif row.p_value < 0.1:
            row_style = "yellow"
        else:
            row_style = "white"
        rich_table.add_row(
            row.feature,
            f"{row.beta:+.4f}",
            f"{row.std_error:.4f}",
            f"{row.z:+.2f}",
            f"{row.p_value:.4g}",
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
            f"{row.odds_ratio:.3f}",
            sig,
            style=row_style,
        )

    console.print(rich_table)
    console.print(
        "[dim]signif: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1[/dim]\n"
    )


# ── Plotting ─────────────────────────────────────────────────────────


def coefficient_plot(
    table: pd.DataFrame,
    title: str,
    out_path: Path,
    drop_intercept: bool = True,
    figsize: tuple[float, float] | None = None,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = table.copy()
    if drop_intercept:
        rows = rows[rows["feature"] != "(intercept)"]
    rows = rows.sort_values("beta", ascending=True).reset_index(drop=True)

    n = len(rows)
    if figsize is None:
        figsize = (10, max(3.5, 0.4 * n + 1.5))
    _, ax = plt.subplots(figsize=figsize)

    def color_for(p: float) -> str:
        if p < 0.01:
            return "#1b5e20"
        if p < 0.05:
            return "#43a047"
        if p < 0.1:
            return "#fbc02d"
        return "#9e9e9e"

    colors = [color_for(p) for p in rows["p_value"]]
    yticks = np.arange(n)
    err_low = rows["beta"] - rows["ci_low"]
    err_high = rows["ci_high"] - rows["beta"]

    ax.barh(yticks, rows["beta"], color=colors, alpha=0.85, edgecolor="black", lw=0.4)
    ax.errorbar(
        rows["beta"],
        yticks,
        xerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(rows["feature"])
    ax.set_xlabel("Coefficient (log-odds, with 95% CI)")
    ax.set_title(title)

    # p-value annotations to the right of each bar
    span = float(rows["ci_high"].max() - rows["ci_low"].min())
    pad = 0.02 * span if span > 0 else 0.05
    for y, beta, ci_h, ci_l, p in zip(
        yticks,
        rows["beta"],
        rows["ci_high"],
        rows["ci_low"],
        rows["p_value"],
        strict=True,
    ):
        anchor = ci_h if beta >= 0 else ci_l
        ha = "left" if beta >= 0 else "right"
        offset = pad if beta >= 0 else -pad
        ax.text(
            anchor + offset,
            y,
            f"p={p:.3g}{significance_stars(p)}",
            va="center",
            ha=ha,
            fontsize=8,
        )

    # legend for the color coding
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1b5e20", label="p < 0.01"),
        plt.Rectangle((0, 0), 1, 1, color="#43a047", label="p < 0.05"),
        plt.Rectangle((0, 0), 1, 1, color="#fbc02d", label="p < 0.1"),
        plt.Rectangle((0, 0), 1, 1, color="#9e9e9e", label="n.s."),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)

    # widen x-limits so annotations don't get clipped
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 0.25 * (xmax - xmin), xmax + 0.35 * (xmax - xmin))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── Run: educational model ───────────────────────────────────────────

dataset = diff_complete_feature_set()
X_train_full, X_test_full, y_train, y_test, _, _ = chronological_split(
    dataset.X, dataset.y, dataset.metadata
)

edu_features = LogisticEducationalModel().active_feature_names
console.rule("[bold]LogisticEducationalModel — parametric inference[/bold]")
console.print(f"features: {edu_features}")
console.print(f"train rows: {len(X_train_full)}, test rows: {len(X_test_full)}\n")

X_train_edu = X_train_full[edu_features].to_numpy(dtype=float)
y_train_arr = y_train.to_numpy(dtype=int)

# (a) Inference on the natural (raw) scale — coefficients have unit meaning
edu_model_raw = fit_unregularized_logit(X_train_edu, y_train_arr)
edu_table_raw = wald_inference(edu_model_raw, X_train_edu, edu_features)
edu_table_raw["units"] = [
    "—",
    "log-odds per +1 year (Red − Blue)",
    "log-odds per +1 loss in current streak",
    "log-odds per +1 win in current streak",
    "log-odds per +1 cm height",
    "log-odds per +1 cm reach",
    "log-odds per +1 lb weight",
]
print_summary_table(
    "Educational logistic — raw units (per 1 unit of difference)", edu_table_raw
)

# (b) Inference on the standardized scale — coefficients are comparable
scaler = StandardScaler()
X_train_edu_std = scaler.fit_transform(X_train_edu)
edu_model_std = fit_unregularized_logit(X_train_edu_std, y_train_arr)
edu_table_std = wald_inference(edu_model_std, X_train_edu_std, edu_features)
print_summary_table(
    "Educational logistic — standardized units (per 1 SD difference)", edu_table_std
)

coefficient_plot(
    edu_table_std,
    "Educational logistic — standardized coefficients (95% CI, Wald p-values)",
    EDU_PLOT,
)
console.print(f"[dim]Saved {EDU_PLOT.relative_to(REPO_ROOT)}[/dim]\n")


# ── Run: full complete-case diff logistic for context ────────────────

console.rule("[bold]LogisticDiffCompleteModel — parametric inference[/bold]")
diff_features = X_train_full.columns.tolist()
console.print(f"features: {len(diff_features)}")

X_train_diff = X_train_full.to_numpy(dtype=float)
diff_scaler = StandardScaler()
X_train_diff_std = diff_scaler.fit_transform(X_train_diff)
diff_model = fit_unregularized_logit(X_train_diff_std, y_train_arr)
diff_table = wald_inference(diff_model, X_train_diff_std, diff_features)

# Highlight: top 15 by |z|, plus anything significant at 5%
significant = diff_table[diff_table["p_value"] < 0.05].copy()
significant = significant.sort_values("p_value")

console.print(
    f"[bold]Significant features at p<0.05: {len(significant)}/{len(diff_table) - 1}[/bold]"
)
if len(significant):
    print_summary_table(
        "Complete-case diff logistic — significant coefficients (standardized)",
        significant,
    )
else:
    console.print("[yellow]No coefficients reach p<0.05 in this fit.[/yellow]")

# Plot top 20 by |z| so the figure stays readable
top_by_z = (
    diff_table[diff_table["feature"] != "(intercept)"]
    .assign(abs_z=lambda d: d["z"].abs())
    .nlargest(20, "abs_z")
    .drop(columns="abs_z")
)
coefficient_plot(
    top_by_z,
    "Diff-only logistic — top 20 coefficients by |z| (95% CI, Wald p-values)",
    DIFF_PLOT,
    drop_intercept=False,
)
console.print(f"[dim]Saved {DIFF_PLOT.relative_to(REPO_ROOT)}[/dim]")
