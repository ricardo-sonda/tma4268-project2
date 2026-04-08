"""Interpretable statsmodels logistic regression on a small complete-case diff set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..features import diff_complete_feature_set
from .base import BaseModel


class LogisticStatsmodelsInterpretableModel(BaseModel):
    """Statsmodels logistic regression with outputs tailored for interpretation."""

    name = "logistic_statsmodels_interpretable"

    def __init__(self) -> None:
        self.feature_builder = diff_complete_feature_set
        self.active_feature_names = [
            "AgeDiff",
            "CurrentLoseStreakDiff",
            "CurrentWinStreakDiff",
            "HeightCmsDiff",
            "ReachCmsDiff",
            "WeightLbsDiff",
        ]
        self.result_: sm.discrete.discrete_model.BinaryResultsWrapper | None = None
        self.training_columns_: list[str] | None = None

    def _require_fit(self) -> sm.discrete.discrete_model.BinaryResultsWrapper:
        if self.result_ is None:
            raise RuntimeError("Model must be fit before accessing statsmodels results.")
        return self.result_

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.active_feature_names) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required features for {self.name}: {missing}")
        return X.loc[:, self.active_feature_names].astype(float)

    def _build_design_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        design = sm.add_constant(
            self._select_features(X),
            prepend=True,
            has_constant="add",
        )
        if self.training_columns_ is not None:
            design = design.reindex(columns=self.training_columns_, fill_value=0.0)
        return design

    def fit(self, X, y) -> None:
        design = self._build_design_matrix(X)
        self.training_columns_ = design.columns.tolist()
        self.result_ = sm.Logit(y.astype(float), design).fit(
            method="lbfgs",
            disp=False,
            maxiter=1000,
        )

    def predict_proba(self, X) -> np.ndarray:
        design = self._build_design_matrix(X)
        return np.asarray(self._require_fit().predict(design), dtype=float)

    def coefficient_frame(self) -> pd.DataFrame:
        result = self._require_fit()
        conf_int = result.conf_int()
        table = pd.DataFrame(
            {
                "feature": result.params.index,
                "coefficient": result.params.to_numpy(),
                "odds_ratio": np.exp(result.params.to_numpy()),
                "ci_low": np.exp(conf_int[0].to_numpy()),
                "ci_high": np.exp(conf_int[1].to_numpy()),
                "p_value": result.pvalues.to_numpy(),
            }
        )
        return table
