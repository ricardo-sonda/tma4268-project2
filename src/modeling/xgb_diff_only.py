"""XGBoost gradient-boosted trees on the diff-imputed feature set.

Uses diff_imputed_feature_set which preserves all 6,275 rows (no dropna).
XGBoost handles NaN natively by learning optimal split directions for missing values.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ..features import diff_imputed_feature_set
from .base import BaseModel

CATEGORICAL_COLUMNS = ["TitleBout", "WeightClass", "Gender"]


class XGBDiffOnlyModel(BaseModel):
    name = "xgb_diff_only"

    def __init__(self) -> None:
        self.feature_builder = diff_imputed_feature_set
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            eval_metric="logloss",
            enable_categorical=True,
        )

    def _encode(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        X = X.copy()
        for col in CATEGORICAL_COLUMNS:
            if col not in X.columns:
                continue
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X[col] = le.transform(X[col].astype(str))
            X[col] = X[col].astype("category")
        return X

    def fit(self, X, y) -> None:
        X = self._encode(X, fit=True)
        self.model.fit(X, y)

    def predict_proba(self, X) -> np.ndarray:
        X = self._encode(X, fit=False)
        return self.model.predict_proba(X)[:, 1]
