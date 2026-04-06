"""Logistic regression on a small educational diff-only feature subset."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from ..features import diff_only_feature_set
from .base import BaseModel


class LogisticEducationalModel(BaseModel):
    name = "logistic_educational"

    def __init__(self) -> None:
        self.feature_builder = diff_only_feature_set
        self.active_feature_names = [
            "AgeDiff",
            "CurrentLoseStreakDiff",
            "CurrentWinStreakDiff",
            "HeightCmsDiff",
            "ReachCmsDiff",
            "WeightLbsDiff",
        ]
        self.pipeline = Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )

    def fit(self, X, y) -> None:
        X = X[self.active_feature_names]
        self.pipeline.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        X = X[self.active_feature_names]
        return self.pipeline.predict_proba(X)[:, 1]
