"""Baseline model that always predicts Red wins."""

import numpy as np

from ..features import ground_zero_feature_set
from .base import BaseModel


class BaselineModel(BaseModel):
    name = "baseline"

    def __init__(self) -> None:
        self.feature_builder = ground_zero_feature_set

    def fit(self, X, y) -> None:
        pass

    def predict_proba(self, X) -> np.ndarray:
        return np.full(len(X), 0.6, dtype=float)
