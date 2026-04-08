"""Pedagogical Elastic Net logistic regression with inner cross-validation.

This model is intentionally written to make the statistical workflow visible:

1. The shared evaluator in ``src/evaluation.py`` creates the outer train/test split.
2. Inside ``fit()``, GridSearchCV runs the inner cross-validation loop.
3. Every inner fold refits the full sklearn pipeline from scratch:
   - impute missing values
   - scale numeric variables
   - one-hot encode categorical variables
   - fit penalized logistic regression

That gives a simple nested-CV structure:
- outer split: honest evaluation handled by the repository evaluator
- inner CV: tuning handled inside this model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features import diff_feature_set
from .base import BaseModel


class ElasticNetNestedCVDiffModel(BaseModel):
    """Elastic Net logistic regression on the default diff feature set.

    Why this feature set?
    - It keeps the model focused on pre-fight matchup differences.
    - It preserves all rows, so the imputer has a clear role.
    - It stays small enough to be easy to reason about in a course project.
    """

    name = "elasticnet_nested_cv_diff"

    def __init__(self) -> None:
        self.feature_builder = diff_feature_set

        # Numeric columns need median imputation and scaling.
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

        # Categorical columns are imputed separately, then one-hot encoded.
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                ),
            ]
        )

        # ColumnTransformer keeps preprocessing inside the pipeline, so each
        # cross-validation fold learns its own medians, scaling constants,
        # and category levels from training data only.
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    make_column_selector(dtype_include=["number", "bool"]),
                ),
                (
                    "cat",
                    categorical_pipeline,
                    make_column_selector(dtype_include=["object", "string", "category"]),
                ),
            ]
        )

        # Elastic Net is a practical compromise between Ridge and Lasso.
        # It can zero out some coefficients while staying more stable when
        # predictors are correlated.
        base_pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Inner CV lives here. The repository evaluator supplies the outer split.
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # C controls overall regularization strength.
        # Smaller C = stronger regularization.
        # l1_ratio controls the Ridge/Lasso mix:
        # 0.0 -> mostly Ridge, 1.0 -> pure Lasso.
        self.search = GridSearchCV(
            estimator=base_pipeline,
            param_grid={
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
            },
            scoring="neg_log_loss",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        self.best_pipeline_: Pipeline | None = None
        self.best_params_: dict[str, float] | None = None
        self.selected_feature_names_: list[str] = []
        self.selected_coefficients_: pd.Series | None = None

    def fit(self, X, y) -> None:
        # GridSearchCV now performs the inner loop:
        # for each hyperparameter pair, it runs 5-fold CV on the provided
        # training set and keeps the setting with the best mean log loss.
        self.search.fit(X, y)

        self.best_pipeline_ = self.search.best_estimator_
        self.best_params_ = self.search.best_params_

        # After refitting on the full training set, inspect which coefficients
        # survived the penalty. Those are the selected features.
        preprocess = self.best_pipeline_.named_steps["preprocess"]
        model = self.best_pipeline_.named_steps["model"]

        feature_names = preprocess.get_feature_names_out()
        coefficients = pd.Series(model.coef_[0], index=feature_names)
        selected = coefficients[coefficients.abs() > 1e-8].sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )

        self.selected_feature_names_ = selected.index.tolist()
        self.selected_coefficients_ = selected

    def predict_proba(self, X) -> np.ndarray:
        if self.best_pipeline_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.best_pipeline_.predict_proba(X)[:, 1]
