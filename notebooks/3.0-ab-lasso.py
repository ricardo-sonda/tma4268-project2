"""LASSO feature selection on the diff-only feature set.

Uses L1-penalized logistic regression across a range of regularization
strengths to identify which diff features carry signal for predicting
UFC fight winners.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import chronological_split
from src.features import diff_only_feature_set

REPORTS_DIR = REPO_ROOT / "reports" / "figures" / "lasso"
COEF_PATH_PLOT = REPORTS_DIR / "lasso-coefficient-paths.png"
SELECTED_FEATURES_PLOT = REPORTS_DIR / "lasso-selected-features.png"

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def build_data():
    dataset = diff_only_feature_set()
    X_train, X_test, y_train, y_test, _, metadata_test = chronological_split(
        dataset.X, dataset.y, dataset.metadata
    )
    return X_train, X_test, y_train, y_test, metadata_test, dataset.X.columns.tolist()


def make_lasso_model(C):
    return LogisticRegression(
        penalty="l1", C=C, solver="saga", max_iter=10_000, random_state=42
    )


def fit_lasso_path(X_train, y_train, feature_names, C_values):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    coef_matrix = np.zeros((len(C_values), len(feature_names)))
    for i, C in enumerate(C_values):
        model = make_lasso_model(C)
        model.fit(X_scaled, y_train)
        coef_matrix[i] = model.coef_[0]

    return coef_matrix, scaler


def feature_entry_order(C_values, coef_matrix, feature_names):
    """For each feature, find the smallest C at which it first becomes nonzero."""
    entry = {}
    for j, name in enumerate(feature_names):
        nonzero_indices = np.where(coef_matrix[:, j] != 0)[0]
        if len(nonzero_indices) > 0:
            entry[name] = C_values[nonzero_indices[0]]
    return dict(sorted(entry.items(), key=lambda kv: kv[1]))


def evaluate_at_C(scaler, X_train, X_test, y_train, y_test, C):
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)
    model = make_lasso_model(C)
    model.fit(X_tr, y_train)
    return model, model.score(X_tr, y_train), model.score(X_te, y_test)


def save_coefficient_path_plot(C_values, coef_matrix, feature_names):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    max_abs = np.max(np.abs(coef_matrix), axis=0)
    top_indices = np.argsort(max_abs)[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx in top_indices:
        ax.plot(np.log10(C_values), coef_matrix[:, idx], lw=1.5, label=feature_names[idx])

    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("log10(C)  [stronger regularization <-- --> weaker]")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("LASSO Coefficient Paths (top 15 by peak magnitude)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(COEF_PATH_PLOT, dpi=200)
    plt.close()


def save_selected_features_plot(feature_names, coefs):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    nonzero_mask = coefs != 0
    names = np.array(feature_names)[nonzero_mask]
    values = coefs[nonzero_mask]
    order = np.argsort(np.abs(values))

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    colors = ["#c0392b" if v < 0 else "#2980b9" for v in values[order]]
    ax.barh(names[order], values[order], color=colors)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("LASSO-selected features (nonzero coefficients)")
    plt.tight_layout()
    plt.savefig(SELECTED_FEATURES_PLOT, dpi=200)
    plt.close()


# ── Run ──────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test, metadata_test, feature_names = build_data()

print(f"Total diff-only features: {len(feature_names)}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

C_values = np.logspace(-3, 2, 80)
coef_matrix, scaler = fit_lasso_path(X_train, y_train, feature_names, C_values)

save_coefficient_path_plot(C_values, coef_matrix, feature_names)
print(f"\nSaved coefficient path plot to: {COEF_PATH_PLOT}")

# Feature entry order - which features appear first as regularization loosens
entry_order = feature_entry_order(C_values, coef_matrix, feature_names)
print("\n--- Feature entry order (smallest C = most important) ---")
for rank, (name, c_val) in enumerate(entry_order.items(), 1):
    print(f"  {rank:3d}. {name:<40s}  enters at C = {c_val:.4f}")

# Accuracy across regularization strengths
print("\n--- Accuracy across regularization strengths ---")
print(f"{'C':>10s}  {'nonzero':>7s}  {'train_acc':>9s}  {'test_acc':>9s}")
best_test_acc, best_C = 0.0, C_values[0]
for C in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0]:
    model, train_acc, test_acc = evaluate_at_C(scaler, X_train, X_test, y_train, y_test, C)
    n_nonzero = np.sum(model.coef_[0] != 0)
    print(f"{C:10.3f}  {n_nonzero:7d}  {train_acc:9.4f}  {test_acc:9.4f}")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_C = C

print(f"\nBest test accuracy: {best_test_acc:.4f} at C={best_C}")

# Refit at best C and show surviving features
best_model, _, _ = evaluate_at_C(scaler, X_train, X_test, y_train, y_test, best_C)
coefs = best_model.coef_[0]

save_selected_features_plot(feature_names, coefs)
print(f"Saved selected features plot to: {SELECTED_FEATURES_PLOT}")

surviving = pd.DataFrame({
    "feature": np.array(feature_names)[coefs != 0],
    "coefficient": coefs[coefs != 0],
}).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)

print(f"\n--- Features selected by LASSO at C={best_C} ({len(surviving)}/{len(feature_names)}) ---")
print(surviving.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
