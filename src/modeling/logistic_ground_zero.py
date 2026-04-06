"""Logistic regression using the shared ground-zero feature set."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ..features import ground_zero_feature_set
from .base import BaseModel


class LogisticGroundZeroModel(BaseModel):
    name = "logistic_ground_zero"

    def __init__(self) -> None:
        self.feature_builder = ground_zero_feature_set
        self.pipeline = Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )

    def fit(self, X, y) -> None:
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
