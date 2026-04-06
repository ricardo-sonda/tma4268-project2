"""Random forest on fighter-difference features with in-pipeline imputation."""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..features import diff_imputed_feature_set
from .base import BaseModel


class RandomForestDiffImputedModel(BaseModel):
    name = "random_forest_diff_imputed"

    def __init__(self) -> None:
        self.feature_builder = diff_imputed_feature_set
        self.pipeline = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            (
                                "num",
                                Pipeline(
                                    [("impute", SimpleImputer(strategy="median"))]
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
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=10,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X, y) -> None:
        self.pipeline.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]
