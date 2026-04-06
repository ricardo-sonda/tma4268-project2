"""ElasticNet logistic regression with interaction terms on diff features.

Uses diff_imputed_feature_set (all 6,275 rows). Median-imputes NaNs in the
pipeline so sklearn can handle the missing striking/grappling stats.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from ..features import diff_imputed_feature_set
from .base import BaseModel

# Strongest diff predictors (by correlation magnitude with target).
KEY_FEATURES = [
    "AgeDiff",
    "CurrentWinStreakDiff",
    "CurrentLoseStreakDiff",
    "LossesDiff",
    "AvgTDLandedDiff",
    "AvgSigStrPctDiff",
    "AvgTDPctDiff",
    "ReachCmsDiff",
    "AvgSigStrLandedDiff",
    "AvgSubAttDiff",
    "TotalRoundsFoughtDiff",
    "WinsDiff",
]


class ElasticNetDiffOnlyModel(BaseModel):
    name = "elasticnet_diff_only"

    def __init__(self) -> None:
        self.feature_builder = diff_imputed_feature_set
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        C=0.1,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, X, y) -> None:
        X = X[KEY_FEATURES]
        self.pipeline.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        X = X[KEY_FEATURES]
        return self.pipeline.predict_proba(X)[:, 1]
