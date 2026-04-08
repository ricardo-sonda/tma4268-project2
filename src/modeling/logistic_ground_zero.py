"""Logistic regression on the default ground-zero feature set."""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features import ground_zero_feature_set
from .base import BaseModel


class LogisticGroundZeroModel(BaseModel):
    name = "logistic_ground_zero"

    def __init__(self) -> None:
        self.feature_builder = ground_zero_feature_set
        self.pipeline = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            (
                                "num",
                                Pipeline(
                                    [
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                selector(dtype_include=["number", "bool"]),
                            ),
                            (
                                "cat",
                                Pipeline(
                                    [
                                        (
                                            "impute",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "encode",
                                            OneHotEncoder(
                                                handle_unknown="ignore",
                                                sparse_output=False,
                                            ),
                                        ),
                                    ]
                                ),
                                selector(dtype_exclude=["number", "bool"]),
                            ),
                        ]
                    ),
                ),
                ("model", LogisticRegression(max_iter=4000, random_state=42)),
            ]
        )

    def fit(self, X, y) -> None:
        self.pipeline.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]
