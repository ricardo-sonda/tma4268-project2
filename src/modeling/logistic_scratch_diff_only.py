"""Unregularized logistic regression fit from scratch with NumPy."""

from __future__ import annotations

import numpy as np

from ..features import diff_only_feature_set
from .base import BaseModel


class LogisticScratchDiffOnlyModel(BaseModel):
    name = "logistic_scratch_diff_only"

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.feature_builder = diff_only_feature_set
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients_: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        intercept = np.ones((len(X), 1), dtype=float)
        return np.column_stack([intercept, X])

    def fit(self, X, y) -> None:
        X_matrix = self._add_intercept(X.to_numpy(dtype=float))
        y_vector = y.to_numpy(dtype=float)
        coefficients = np.zeros(X_matrix.shape[1], dtype=float)

        for _ in range(self.max_iter):
            linear_term = X_matrix @ coefficients
            probabilities = self._sigmoid(linear_term)
            weights = probabilities * (1.0 - probabilities)
            score = X_matrix.T @ (y_vector - probabilities)
            information = X_matrix.T @ (weights[:, None] * X_matrix)

            try:
                step = np.linalg.solve(information, score)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(information) @ score

            updated = coefficients + step
            if np.max(np.abs(updated - coefficients)) < self.tol:
                coefficients = updated
                break
            coefficients = updated

        self.coefficients_ = coefficients

    def predict_proba(self, X) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")

        X_matrix = self._add_intercept(X.to_numpy(dtype=float))
        return self._sigmoid(X_matrix @ self.coefficients_)
