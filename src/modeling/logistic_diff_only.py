"""Logistic regression using only corner-difference fighter features."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ..features import diff_only_feature_set
from .base import BaseModel


class LogisticDiffOnlyModel(BaseModel):
    name = "logistic_diff_only"

    def __init__(self) -> None:
        self.feature_builder = diff_only_feature_set
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
