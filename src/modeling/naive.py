"""Baseline model that always predicts 0.5."""

import numpy as np

from ..features import ground_zero_feature_set
from .base import BaseModel


class NaiveModel(BaseModel):
    name = "naive"

    def __init__(self) -> None:
        # Instance attribute so feature_builder is not bound as a method (class
        # attribute would pass self as the first positional arg to the callable).
        self.feature_builder = ground_zero_feature_set

    def fit(self, X, y) -> None:
        pass

    def predict_proba(self, X):
        return np.full(len(X), 0.5, dtype=float)
