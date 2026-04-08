"""Complete-case logistic regression with weight-class interactions."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from ..features import weightclass_interaction_complete_feature_set
from .base import BaseModel


class LogisticWeightclassInteractionCompleteModel(BaseModel):
    name = "logistic_weightclass_interaction_complete"

    def __init__(self) -> None:
        self.feature_builder = weightclass_interaction_complete_feature_set
        self.pipeline = Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=4000, random_state=42)),
            ]
        )

    def fit(self, X, y) -> None:
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
