"""Logistic regression with weight-class main effects on the default feature set."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ..features import weightclass_main_effect_feature_set
from .base import BaseModel


class LogisticWeightclassMainEffectModel(BaseModel):
    name = "logistic_weightclass_main_effect"

    def __init__(self) -> None:
        self.feature_builder = weightclass_main_effect_feature_set
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=4000, random_state=42)),
            ]
        )

    def fit(self, X, y) -> None:
        self.pipeline.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]
